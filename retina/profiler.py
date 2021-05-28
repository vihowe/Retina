import multiprocessing as mp
import time
import csv
import os
from tqdm import tqdm
from retina.option import Option
from retina.worker.profile_worker import ProfileWorker


class Profiler(object):
    def __init__(
            self,
            runtime_config,
    ):
        self._runtime_config = runtime_config
        self._worker_list = []

    def profile(self):
        profile_data_path = os.path.join(self._runtime_config.data_path, 'profile_data')
        os.makedirs(profile_data_path, exist_ok=True)

        models = list(self._runtime_config.models_list.keys())

        barrier = mp.Barrier(len(models) + 1)

        profile_filename = models[0]
        for i, model_name in enumerate(models):
            profile_filename += '_'
            profile_filename += model_name

            parent, child = mp.Pipe()
            worker = ProfileWorker(
                i,
                model_name,
                self._runtime_config.supported_bs,
                child,
                barrier,
                self._runtime_config,
            )
            self._worker_list.append((worker, parent))
            worker.start()
        barrier.wait()

        profile_filename += '_' + str(self._runtime_config.cur_bs) + '.csv'
        profile_file = open(os.path.join(profile_data_path, profile_filename), 'w+')
        csv_wr = csv.writer(profile_file)
        csv_wr.writerow(['latency(ms)'])

        with tqdm(range(self._runtime_config.test_loop)) as t:
            for test_i in range(self._runtime_config.test_loop):
                start_time = time.time()
                for i in range(len(self._worker_list)):
                    _, pipe = self._worker_list[i]
                    pipe.send(self._runtime_config.cur_bs)

                barrier.wait()
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000  # ms
                # print("test_{}: elapsed_time = {}".format(test_i, elapsed_time))
                csv_wr.writerow([elapsed_time])
                t.set_postfix(elapsed_time=elapsed_time)
                t.update(1)
        for i in range(len(self._worker_list)):
            _, pipe = self._worker_list[i]
            pipe.send(-1)
        for i in range(len(self._worker_list)):
            worker, _ = self._worker_list[i]
            worker.join()


if __name__ == "__main__":
    runtime_config = Option()
    profiler = Profiler(runtime_config)
    profiler.profile()
