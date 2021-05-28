import sys
sys.path.append("..")
from retina.network.resnet_splited import resnet152
import torch 

device = torch.device("cuda")
model = resnet152().to("cuda")
input = torch.randn((1, 3, 224, 224)).to(device)
output = model(input)
print(output.shape)