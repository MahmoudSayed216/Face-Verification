from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch
# model = densenet161(weights = None)

model = convnext_base(weights = None).features

# print(model[-1][-1].)


# for param in model.features.denseblock4.parameters():
    # param.requires_grad = True
# # print(model)
# layers = [*model.children()][:-1]
# model = torch.nn.Sequential(*layers)
# print(model)
# print(type(model))


# from model import CringeNet
# model = CringeNet()