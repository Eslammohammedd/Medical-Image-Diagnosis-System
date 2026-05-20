import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical imaging datasets.
    Down-weights well-classified examples so the model focuses on hard ones.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with class weights to handle imbalanced datasets.
    Useful when one class (e.g. Normal) dominates the dataset.
    """
    def __init__(self, class_weights=None, device='cpu'):
        super(WeightedCrossEntropyLoss, self).__init__()
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        else:
            weights = None
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)


def get_loss_fn(loss_name: str, **kwargs):
    """
    Factory function to return the correct loss by name.
    Supported: 'CrossEntropyLoss', 'FocalLoss', 'WeightedCrossEntropyLoss'
    """
    loss_name = loss_name.strip()
    if loss_name == 'FocalLoss':
        return FocalLoss(**kwargs)
    elif loss_name == 'WeightedCrossEntropyLoss':
        return WeightedCrossEntropyLoss(**kwargs)
    else:
        return nn.CrossEntropyLoss()
