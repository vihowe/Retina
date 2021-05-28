import multiprocessing as mp

from retina import Worker
import torch


class ProfileWorker(Worker):
    def __init__(
            self,
            worker_id,
            model_name: str,
            supported_bs,
            recv_pipe: mp.Pipe(),
            barrier,  # synchronization for different profile worker
            run_config,
    ):
        super(ProfileWorker, self).__init__(
            worker_id,
            model_name,
            supported_bs,
            recv_pipe,
        )
        self._barrier = barrier
        self._run_config = run_config

    def run(self) -> None:
        device = torch.device("cuda")
        model_fc = self._run_config.models_list[self._model_name]().to(device)

        # prepare input data
        if self._model_name[0:6] == 'resnet':
            self._input = {
                k: torch.rand((k, 3, 224, 224)).to(device) for k in self._supported_bs
            }
        else:
            self._input = {
                k: torch.rand((k, 1, 100, 100)).to(device) for k in self._supported_bs
            }

        # pre warming
        for bs in self._supported_bs:
            model_fc(self._input[bs])

        self._barrier.wait()

        with torch.no_grad():
            while True:
                bs = self._recv_pipe.recv()
                if bs == -1:
                    break

                model_fc(self._input[bs])

                torch.cuda.synchronize(device)
                self._barrier.wait()
