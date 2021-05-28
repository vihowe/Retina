from retina.worker.server_worker import ServerWorker
from retina import Worker
import time
import torch
import random
from multiprocessing import Process,Pipe
from retina import option

class Request(object):
    def __init__(self, id, sleep_duration) -> None:
        super().__init__()
        self._id = id
        self.sleep_duration = sleep_duration
        
    
class Serving(object):
    def __init__(self, run_config:option) -> None:
        super().__init__()
        self._run_config = run_config
    
    def start_up(self, model_id=0): # start server worker
        parent_conn, child_conn = Pipe()
        server = ServerWorker(
            model_id,
            list(self._run_config.model_list[model_id].keys())[0],
            [1, 2, 4, 8, 16, 32, 64, 128],
            child_conn,
            self._run_config
        )
        server.start()
        self._pipe = parent_conn

    def prepare_test_queries(self, total_queries=10000, lambd=20):
        self._test_queries = []
        for i in range(total_queries):
            sleep_duration = random.expovariate(lambd)
            self._test_queries.append(Request(i, sleep_duration))

    def querying(self) -> None:
        # forming request according to a distribution and send it to server worker
        self.start_up()
        self.prepare_test_queries()
        start_time = []
        for id, sleep_duration in self._test_queries:
            start_stamp = time.time() * 1000
            start_time.append({id:start_stamp})
            self._pipe.send(id)
            time.sleep(sleep_duration)
        end_time = []
        while True:
            end_stamp = self._pipe.recv()
            if end_stamp == -1:
                break
            else:
                end_time.extend(end_stamp)

        return start_time, end_time






