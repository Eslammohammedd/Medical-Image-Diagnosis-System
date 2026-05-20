import torch
import torch.nn as nn
from torchvision import models

class ResNet50Model(nn.Module):
    """
    ResNet50 architecture for Chest X-Ray classification.
    """
    def __init__(self, num_classes=14, pretrained=True):
        super(ResNet50Model, self).__init__()
        
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Replace the fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=14, pretrained=True):
    return ResNet50Model(num_classes=num_classes, pretrained=pretrained)
