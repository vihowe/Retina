import torchvision.models as models
import os


class Option(object):
    def __init__(
            self,
    ):
        self.model_list = {
            # "resnet50": models.resnet50(),
            # "resnet152": models.resnet152(),
            "resnet152": {'path':'../models/', 'total_segs': 52}
        }

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
