from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import densenet161, DenseNet161_Weights

import torch.nn as nn
import configs

class CringeNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CringeNet, self).__init__()
        densenet = densenet161(weights=None)
        self.backbone = densenet.features  # this keeps the real DenseNet features
        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.flatter = nn.Flatten()
        self.embedder = nn.Linear(2208, embedding_dim)

        # Freeze everything
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Unfreeze only last dense block
        for p in self.backbone.denseblock4.parameters():
            p.requires_grad = True


        # this tweak was suggested by GPT
        nn.init.xavier_uniform_(self.embedder.weight)
        if self.embedder.bias is not None:
            nn.init.zeros_(self.embedder.bias)


    def forward(self, images):  # (B, N, C, H, W)
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)        
        x = self.backbone(x)   
        # x = self.flatter(x)                
        x = self.adaptive_avg_pooling(x)       
        x = x.view(B * N, -1)                  
        x = self.embedder(x)                   
        x = nn.functional.normalize(x, p=2, dim=1)
        x = x.view(B, N, -1)                   
        return x