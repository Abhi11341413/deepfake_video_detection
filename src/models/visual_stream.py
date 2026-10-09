import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class VisemeEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(VisemeEncoder, self).__init__()
        
        # Load pre-trained ResNet-18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        """
        Input: [Batch, Time, Channels, Height, Width]
        We need to process each frame. So we fold Time into the Batch dimension.
        Output: [Batch, 512] (after averaging over the Time dimension)
        """
        b, t, c, h, w = x.size()

        x_folded = x.view(b * t, c, h, w)
        features = self.backbone(x_folded)
        
        features = features.view(b, t, -1)
        
        out = torch.mean(features, dim=1)
        
        return out
