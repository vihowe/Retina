from torch.multiprocessing import Process


class Worker(Process):
    def __init__(
            self,
            worker_id,
            model_name,     # which model you are runing
            supported_bs: list,
            recv_pipe,
    ):
        super().__init__()
        self._worker_id = worker_id
        self._recv_pipe = recv_pipe
        self._supported_bs = supported_bs
        self._model_name = model_name
