from random import choices
import torchvision.models as models
import os
import argparse
from retina.network.resnet_splited import resnet152


class Option(object):
    def __init__(self, args):
        self.task = args.task
        self.device = args.device

        self.models_name = ["resnet152"]
        self.models_list = {
            "resnet152":resnet152
        }

        self.supported_bs = [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
        ]

        self.path = "/home/vihowe/courses/nn/Retina"
        self.data_path = os.path.join(self.path, 'data')
        os.makedirs(self.data_path, exist_ok=True)

        self.cur_bs = 16

        self.test_loop = 100

        self.infer_bs = 8
        self.total_queries = 100

def parse_options():
    parser = argparse.ArgumentParser(description="Retina")

    parser.add_argument(
        "--task",
        type=str,
        default="serve",
        required=True,
        choices=["profile", "serve"],
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"]
    )

    args = parser.parse_args()

    run_config = Option(args=args)
    return run_config
