import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Model(nn.Module):
    """
    DenseNet121 architecture optimized for Chest X-Ray classification.
    Supports multi-label classification (typically 14 classes).
    """
    def __init__(self, num_classes=14, pretrained=True):
        super(DenseNet121Model, self).__init__()
        
        # Load pre-trained DenseNet121
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=weights).features
        
        # Adaptive pooling to handle various input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        num_features = 1024  # DenseNet121 features in the last block
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.pool(features)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits

def get_model(num_classes=14, pretrained=True):
    return DenseNet121Model(num_classes=num_classes, pretrained=pretrained)
