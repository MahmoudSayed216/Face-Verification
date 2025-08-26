from torchvision.models import resnet50, ResNet50_Weights
import torch
model = resnet50(weights = None)

# print(model)
layers = [*model.children()]
model = torch.nn.Sequential(*layers)
print(model)
print(type(model))
