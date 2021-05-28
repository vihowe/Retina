from torch import multiprocessing
from retina.worker.server_worker import ServerWorker
from retina import Worker
import time
import torch
import random
from multiprocessing import Process,Pipe
from retina import option
import csv

class Request(object):
    def __init__(self, id, sleep_duration) -> None:
        super().__init__()
        self._id = id
        self._sleep_duration = sleep_duration
        
    
class Serving(object):
    def __init__(self, run_config:option) -> None:
        super().__init__()
        self._run_config = run_config
    
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
            self._run_config
        )
        server.start()
        self.servers_list = []
        self.servers_list.append((parent_conn, server))

    def prepare_test_queries(self, lambd=200):
        total_queries = self._run_config.total_queries
        self._test_queries = []
        for i in range(total_queries):
            sleep_duration = random.expovariate(lambd)
            self._test_queries.append(Request(i, sleep_duration))

    def querying(self) -> None:
        # forming request according to a distribution and send it to server worker
        self.start_up()
        send_pipe, server = self.servers_list[0]
        self.prepare_test_queries()
        start_time = {}
        self._barrier.wait()
        for query in self._test_queries:
            id = query._id
            sleep_duration = query._sleep_duration
            start_stamp = time.time() * 1000
            start_time.update({id:start_stamp})
            send_pipe.send(id)
            time.sleep(sleep_duration)
        send_pipe.send(-1) # terminate

        end_time = send_pipe.recv()
            
        server.join()
        req_ids = list(start_time.keys())
        elapsed_time = {}
        for id in req_ids:
            elapsed_stamp = end_time[id] - start_time[id]
            elapsed_time.update({id:elapsed_stamp})

        return elapsed_time
    
    def record_latency(self, latency, filename):
        """
        args:
            latenct: dic: id->latency(ms)
        return:
            csv file
        """
        with open(filename, "w+", newline='') as f:
            csv_wr = csv.writer(f)
            req_ids = list(latency.keys())
            req_latencies = []
            for req_id in req_ids:
                req_latencies.append(latency[req_id])
            csv_wr.writerow(req_ids)
            csv_wr.writerow(req_latencies)


        