import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PhonemeEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(PhonemeEncoder, self).__init__()
        
        # Load pre-trained ResNet-18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        
        # Remove the final fully connected layer to output the 512-dim embedding
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        """
        Input: [Batch, Channels (3), MFCC (40), Time (20)]
        Output: [Batch, 512]
        """
        # The input is treated as a 3-channel image
        features = self.backbone(x)
        return features
