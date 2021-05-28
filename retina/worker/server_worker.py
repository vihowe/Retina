import multiprocessing
from retina import Worker
import torch
import queue
import time

class ServerWorker(Worker):
    def __init__(self, worker_id, model_name, supported_bs: list, recv_pipe, barrier, run_config):
        super().__init__(worker_id, model_name, supported_bs, recv_pipe)
        self._barrier = barrier
        self._run_config = run_config

    def run(self):
        device = torch.device(self._run_config.device)

        # deploy submodels
        models = self._run_config.models_list["resnet152"]().get_submodules()

        for model in models:
            model = model.to(device)
            model.eval()
        
        # prewarming
        dump_input = torch.randn((4, 3, 224, 224)).to(device)
        for epoch in range(100):
            infer_input = dump_input.clone()
            for model in models:
                infer_input = model(infer_input)
        
        # receive req from client and serving, using dump input data to ignore the data transport time
        infer_bs = self._run_config.infer_bs
        dump_input = torch.randn((infer_bs, 3, 224, 224)).to(device)

        with torch.no_grad():
            requeue = queue.Queue()
            end_time = {}
            self._barrier.wait()
            while True:
                req_id = self._recv_pipe.recv()
                if req_id == -1:
                    if not requeue.empty():
                        reqs_id = []
                        while not requeue.empty():
                            reqs_id.append(requeue.get())
                        dump_input = torch.randn((len(reqs_id), 3, 224, 224)).to(device)
                        for model in models:
                            dump_input = model(dump_input)
                        end_stamp = time.time() * 1000
                        for id in reqs_id:
                            end_time[id] = end_stamp

                    self._recv_pipe.send(end_time)
                    break
                else:
                    requeue.put(req_id)
                if requeue.qsize() >= infer_bs:
                    reqs_id = []
                    for i in range(infer_bs):
                        reqs_id.append(requeue.get())
                    infer_data = dump_input.clone()
                    for model in models:
                        infer_data = model(infer_data)
                        torch.cuda.synchronize(device)
                    end_stamp = time.time() * 1000 #ms
                    for id in reqs_id:
                        end_time[id] = end_stamp
                    
                    


        