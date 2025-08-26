from torchvision.models import resnet101, ConvNeXt_Small_Weights
import torch
model = resnet101(weights = None)

# print(model)
layers = [*model.children()][:-1]
model = torch.nn.Sequential(*layers)
print(model)
print(type(model))
