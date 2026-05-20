import torch
import torch.nn as nn
import timm

class ViTBaseModel(nn.Module):
    """
    Vision Transformer (ViT) Base architecture for Chest X-Ray.
    Requires 'timm' library.
    """
    def __init__(self, num_classes=14, pretrained=True):
        super(ViTBaseModel, self).__init__()
        
        self.backbone = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=14, pretrained=True):
    return ViTBaseModel(num_classes=num_classes, pretrained=pretrained)
