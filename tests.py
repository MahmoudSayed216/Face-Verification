from torchvision.models import convnext_small, ConvNeXt_Small_Weights
import torch
model = convnext_small(weights = None)

# print(model)
layers = [*model.children()]
model = torch.nn.Sequential(*layers)
print(model)
print(type(model))
