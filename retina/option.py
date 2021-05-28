from random import choices
import torchvision.models as models
import os
import argparse


class Option(object):
    def __init__(self, args):
        self.task = args.task
        self.device = args.device

        self.model_list = [
            # "resnet50": models.resnet50(),
            # "resnet152": models.resnet152(),
            {
                "resnet152": {'path':'/home/vihowe/courses/nn/Retina/models/', 'total_segs': 52}
            }
        ]

        self.supported_bs = [
            2,
            4,
            8,
            16,
            32
        ]

        self.path = "/home/vihowe/courses/nn/Retina"
        self.data_path = os.path.join(self.path, 'data')
        os.makedirs(self.data_path, exist_ok=True)

        self.cur_bs = 32

        self.test_loop = 100

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
