from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch.nn as nn
import configs

class CringeNet(nn.Module):
    def __init__(self):
        super(CringeNet, self).__init__()
        self.backbone = convnext_base(weights = None).features
        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.embedder = nn.Linear(1024, configs.EMBEDDING_DIM)
        

        # this tweak was suggested by GPT
        nn.init.xavier_uniform_(self.embedder.weight)
        if self.embedder.bias is not None:
            nn.init.zeros_(self.embedder.bias)


    def forward(self, images):  # (B, N, C, H, W)
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)        # flatten batch + images
        x = self.backbone(x)                   # (B*N, 1024, H', W')
        x = self.adaptive_avg_pooling(x)       # (B*N, 1024, 1, 1)
        x = x.view(B * N, -1)                  # (B*N, 1024)
        x = self.embedder(x)                   # (B*N, EMBEDDING_DIM)
        x = nn.functional.normalize(x, p=2, dim=1)
        x = x.view(B, N, -1)                   # (B, N, EMBEDDING_DIM)
        return x