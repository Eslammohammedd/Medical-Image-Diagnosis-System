import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for multi-class classification."""
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = targets.cpu().numpy()
    return accuracy_score(labels, preds)


def compute_f1(outputs: torch.Tensor, targets: torch.Tensor, average='macro') -> float:
    """Macro F1-score across all classes."""
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = targets.cpu().numpy()
    return f1_score(labels, preds, average=average, zero_division=0)


def compute_precision_recall(outputs: torch.Tensor, targets: torch.Tensor, average='macro'):
    """Returns (precision, recall) tuple."""
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = targets.cpu().numpy()
    prec = precision_score(labels, preds, average=average, zero_division=0)
    rec  = recall_score(labels, preds, average=average, zero_division=0)
    return prec, rec


def compute_auc(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    """
    One-vs-Rest AUC for multi-class problems.
    probs  : (N, num_classes) softmax probabilities (numpy)
    targets: (N,) integer class labels (numpy)
    """
    try:
        from sklearn.preprocessing import label_binarize
        targets_bin = label_binarize(targets, classes=list(range(num_classes)))
        auc = roc_auc_score(targets_bin, probs, multi_class='ovr', average='macro')
        return auc
    except Exception:
        return 0.0


def compute_confusion_matrix(outputs: torch.Tensor, targets: torch.Tensor):
    """Returns the confusion matrix as a numpy array."""
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = targets.cpu().numpy()
    return confusion_matrix(labels, preds)


def get_classification_report(outputs: torch.Tensor, targets: torch.Tensor, class_names: list) -> str:
    """Full sklearn classification report string."""
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = targets.cpu().numpy()
    return classification_report(labels, preds, target_names=class_names, zero_division=0)


class MetricTracker:
    """
    Accumulates batch-level outputs and targets, then computes
    epoch-level metrics in one shot.
    """
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()

    def reset(self):
        self.all_outputs = []
        self.all_targets = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        self.all_outputs.append(outputs.detach().cpu())
        self.all_targets.append(targets.detach().cpu())

    def compute(self) -> dict:
        outputs = torch.cat(self.all_outputs)
        targets = torch.cat(self.all_targets)

        probs   = torch.softmax(outputs, dim=1).numpy()
        labels  = targets.numpy()

        acc  = compute_accuracy(outputs, targets)
        f1   = compute_f1(outputs, targets)
        prec, rec = compute_precision_recall(outputs, targets)
        auc  = compute_auc(probs, labels, self.num_classes)

        return {
            'accuracy':  round(acc,  4),
            'f1_macro':  round(f1,   4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'auc_macro': round(auc,  4),
        }
