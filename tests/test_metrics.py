import numpy as np
import torch

from training.metrics import (
    compute_accuracy,
    compute_auc,
    compute_confusion_matrix,
    compute_f1,
    get_classification_report,
)


def _outputs_and_targets():
    outputs = torch.tensor([
        [4.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 1.0, 2.0],
        [2.0, 0.0, 1.0],
    ])
    targets = torch.tensor([0, 1, 2, 0])
    return outputs, targets


def test_compute_accuracy():
    outputs, targets = _outputs_and_targets()
    assert compute_accuracy(outputs, targets) == 1.0


def test_compute_f1():
    outputs, targets = _outputs_and_targets()
    score = compute_f1(outputs, targets, average="macro")
    assert 0.0 <= score <= 1.0
    assert score > 0.6


def test_compute_confusion_matrix_shape():
    outputs, targets = _outputs_and_targets()
    matrix = compute_confusion_matrix(outputs, targets)
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 4


def test_compute_auc():
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
    ])
    targets = np.array([0, 1, 2, 0])
    assert 0.0 <= compute_auc(probs, targets, num_classes=3) <= 1.0


def test_classification_report_contains_class_names():
    outputs, targets = _outputs_and_targets()
    report = get_classification_report(outputs, targets, ["A", "B", "C"])
    assert "A" in report
    assert "macro avg" in report
