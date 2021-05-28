import torch

models = []
for i in range(1, 53):
    model = torch.load('../models/resnet152_52_'+str(i)+'.pth')
    models.append(model)

input_data = torch.randn((1, 3, 224, 224))
for model in models:
    input_data = model(input_data)

print(input_data.shape)