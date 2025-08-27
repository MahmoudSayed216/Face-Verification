# from torchvision.models import densenet161, DenseNet161_Weights
# import torch
# model = densenet161(weights = None)
# for param in model.features.denseblock4.parameters():
#     param.requires_grad = True
# # # print(model)
# # layers = [*model.children()][:-1]
# # model = torch.nn.Sequential(*layers)
# # print(model)
# # print(type(model))


from model import CringeNet
model = CringeNet()