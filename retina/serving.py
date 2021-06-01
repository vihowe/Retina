from torch import multiprocessing
from retina.worker.server_worker import ServerWorker
from retina import Worker
import time
import torch
import random
from multiprocessing import Process,Pipe
from retina.option import Option, ServingConfig
import csv

class Request(object):
    def __init__(self, id, sleep_duration, cur_load) -> None:
        super().__init__()
        self._id = id
        self._sleep_duration = sleep_duration
        self._cur_load = cur_load
        
class Serving(object):
    def __init__(self, run_config:Option, serve_config:ServingConfig) -> None:
        super().__init__()
        self._run_config = run_config
        self._serve_config = serve_config
    
    def start_up(self, model_id=0): # start server worker
        parent_conn, child_conn = Pipe()
        barrier = multiprocessing.Barrier(2)
        self._barrier = barrier
        server = ServerWorker(
            model_id,
            "resnet152",    #TODO model id2name
            [1, 2, 4, 8, 16, 32, 64, 128],
            child_conn,
            barrier,
            self._run_config,
            self._serve_config
        )
        server.start()
        self.servers_list = []
        self.servers_list.append((parent_conn, server))

    def prepare_test_queries(self, start_id, lambd):
        total_queries = self._serve_config.total_queries
        # self._test_queries = []
        for i in range(total_queries):
            sleep_duration = random.expovariate(lambd)
            self._test_queries.append(Request(start_id+i, sleep_duration, lambd))

    def querying(self) -> None:
        # forming request according to a distribution and send it to server worker
        self.start_up()
        send_pipe, server = self.servers_list[0]

        supported_load = self._serve_config.supported_load
        self._test_queries = []
        start_id = 0
        for load in supported_load:
            self.prepare_test_queries(start_id ,load)
            start_id += self._serve_config.total_queries

        # start_time = {}
        self._barrier.wait()
        for query in self._test_queries:
            id = query._id
            sleep_duration = query._sleep_duration
            start_stamp = time.time() * 1000
            # start_time.update({id:start_stamp})
            query._start_time = start_stamp
            send_pipe.send(id)
            # send_pipe.send(query)
            time.sleep(sleep_duration)
        send_pipe.send(-1) # terminate

        end_time = send_pipe.recv()     # req_id -> (infer_bs, end_stamp)
            
        server.join()
        # req_ids = list(start_time.keys())
        # elapsed_time = {}
        # for id in req_ids:
        #     elapsed_stamp = end_time[id][1] - start_time[id]
        #     elapsed_time.update({id:elapsed_stamp})
        for query in self._test_queries:
            query._infer_bs = end_time[query._id][0]
            query._end_time = end_time[query._id][1]

        return self._test_queries
    
    def record_latency(self, queries, filename):
        """
        args:
            queries: (id, load, infer_bs, start_time, end_time)
        return:
            csv file
        """
        with open(filename, "w+", newline='') as f:
            csv_wr = csv.writer(f)
            csv_wr.writerow(['id', 'load', 'bs', 'reward'])
            for query in queries:
                id = query._id
                load = query._cur_load
                infer_bs = query._infer_bs
                reward = query._start_time - query._end_time
                csv_wr.writerow([id, load, infer_bs, reward])