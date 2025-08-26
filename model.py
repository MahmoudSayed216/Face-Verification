from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import resnet101, ResNet101_Weights

import torch.nn as nn
import configs

class CringeNet(nn.Module):
    def __init__(self):
        super(CringeNet, self).__init__()
        # self.backbone = .features
        layers = [*resnet101(weights = ResNet101_Weights).children()][:-1]
        self.backbone = nn.Sequential(*layers)
        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.embedder = nn.Linear(2048, configs.EMBEDDING_DIM)
        
        for p in self.backbone.parameters():
            p.requires_grad = False

        for name, param in self.backbone.named_parameters():
            if "7" in name or "6" in name or "5" in name:
                param.requires_grad = True


        # this tweak was suggested by GPT
        nn.init.xavier_uniform_(self.embedder.weight)
        if self.embedder.bias is not None:
            nn.init.zeros_(self.embedder.bias)


    def forward(self, images):  # (B, N, C, H, W)
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)        
        x = self.backbone(x)                   
        x = self.adaptive_avg_pooling(x)       
        x = x.view(B * N, -1)                  
        x = self.embedder(x)                   
        x = nn.functional.normalize(x, p=2, dim=1)
        x = x.view(B, N, -1)                   
        return x