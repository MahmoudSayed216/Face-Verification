from model import CringeNet
import torch


image = torch.rand((10, 3, 224, 224))


model = CringeNet()


print(model(image).shape)