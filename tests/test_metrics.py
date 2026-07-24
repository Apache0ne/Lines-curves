import torch

from lines_curves.utils import binary_counts, binary_metrics, binary_metrics_from_counts


def test_global_binary_metrics_and_threshold_behavior():
    probabilities = torch.tensor([[[[0.30, 0.40, 0.80, 0.10]]]])
    logits = torch.logit(probabilities)
    target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])

    counts_035 = binary_counts(logits, target, 0.35)
    assert counts_035 == (1, 1, 1)
    assert binary_metrics_from_counts(*counts_035)["f1"] == 0.5

    metrics_025 = binary_metrics(logits, target, 0.25)
    assert metrics_025["recall"] == 1.0
    assert metrics_025["precision"] == 2 / 3
    assert metrics_025["f1"] > binary_metrics(logits, target, 0.50)["f1"]
