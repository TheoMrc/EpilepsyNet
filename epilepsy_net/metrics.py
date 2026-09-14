"""Script to compute metrics for the EpilepsyNet model."""

import numpy as np


def compute_f1_metrics(
    predictions: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 score from binary labels and predictions.

    Args:
        predictions (np.ndarray): Predicted labels (0 or 1).
        labels (np.ndarray): Ground truth labels (0 or 1).

    Returns:
        tuple: (precision, recall, f1) as floats.
    """
    true_positives = predictions[labels].sum()
    false_positives = predictions[~labels].sum()
    if true_positives == 0:
        return 0.0, 0.0, 0.0

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / labels.sum()
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_latency(moving_states: np.ndarray, min_duration: int = 5) -> float:
    """Calculate latency as the first index where moving_states=1 for at least min_duration consecutive frames.

    Args:
        moving_states (np.ndarray): 1D array where 1=moving, 0=not_moving.
        min_duration (int): Minimum number of consecutive moving frames (default: 5).

    Returns:
        float: Index of the first valid moving frame, or np.nan if none.
    """
    n_frames = len(moving_states)
    for i in range(n_frames - 1):
        if moving_states[i] == 1:
            if i + min_duration > n_frames - 1:
                continue
            if all(moving_states[j] == 1 for j in range(i, i + min_duration)):
                return float(i + 1)
    return np.nan
