from resnet_splited import *
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse


def generate_splited_model(model_name_list, model_list, input_tensor, seg_num=1):
    for model, model_name in zip(model_list, model_name_list):
        model.cuda()
        model.eval()
        x = input_tensor.cuda()
        y = model(x)
        print("Start generating " + model_name + ":")
        cnt = 1
        submodules = model.get_submodules()
        # print(submodules)
        num_submodules = len(submodules)
        print('num_submodules={}'.format(num_submodules))

        seg_num = seg_num if num_submodules > seg_num else num_submodules

        module_step = (num_submodules + seg_num - 1) // seg_num
        for i in range(0, num_submodules, module_step):
            start_id = i
            stop_id = (
                i + module_step
                if num_submodules > (i + module_step)
                else num_submodules
            )
            submodule = []
            for j in range(start_id, stop_id):
                submodule.append(submodules[j])

            submodule = nn.Sequential(*submodule)
            # print(submodule)

            print(
                "Generating {}_{}_{}.onnx, containing {} ~ {} modules".format(
                    model_name, seg_num, cnt, start_id, stop_id
                )
            )

            model_dir = '../models/'
            torch.save(submodule, model_dir+model_name+'_'+str(seg_num)+'_'+str(cnt)+'.pth')

            cnt = cnt + 1
            x = submodule(x)
            # print(x.shape)
        print(torch.mean(y - x))
        print("End generating " + model_name + ":")


def main():
    # modify generate_resnet to be true if you want to generate resnet-onnxfile
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="specify the model for breaking down",
        type=str,
        default="resnet",
        choices=["resnet", "vgg", "inception", "unet"],
    )
    parser.add_argument(
        "-s", "--seg", help="specify the number of segments", type=int, default=1
    )
    args = parser.parse_args()
    cudnn.benchmark = True
    if args.model == "resnet":
        generate_splited_model(
            model_name_list=[

             #   "resnet_18",
             #   "resnet_34",
                # "resnet_50",
            #    "resnet_101",
               "resnet152",
             #   "resnext50_32x4d",
             #   "resnext101_32x8d",
             #   "wide_resnet50_2",
             #   "wide_resnet101_2",
            ],
            model_list=[
            #    resnet18(),
             #   resnet34(),
                # resnet50(),
            #    resnet101(),
               resnet152(),
             #   resnext50_32x4d(),
             #   resnext101_32x8d(),
             #   wide_resnet50_2(),
             #   wide_resnet101_2(),
            ],

            input_tensor=torch.randn(1, 3, 224, 224, requires_grad=False),

            seg_num=args.seg,
        )

    # elif args.model == "vgg":
    #     generate_splited_model(
    #         model_name_list=[
    #             # "vgg_16",
    #             "vgg_19"
    #         ],
    #         model_list=[
    #             # vgg16(),
    #             vgg19()
    #         ],
    #         input_tensor=torch.randn(1, 3, 224, 224, requires_grad=False),
    #         seg_num=args.seg,
    #     )
    # elif args.model == "inception":
    #     generate_splited_model(
    #         model_name_list=["inception"],
    #         model_list=[inception_v3()],
    #         input_tensor=torch.randn(1, 3, 299, 299, requires_grad=False),
    #         seg_num=args.seg,
    #     )
    #
    # elif args.model == 'unet':
    #     generate_splited_model(
    #         model_name_list=['unet'],
    #         model_list=[Unet()],
    #         input_tensor=torch.randn(1,3,256,256),
    #         seg_num=args.seg,
    #     )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
    # print(resnet18)
