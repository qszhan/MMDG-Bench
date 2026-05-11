"""
Face Anti-Spoofing Evaluation Metrics

Implements standard FAS metrics:
- AUC: Area Under ROC Curve
- EER: Equal Error Rate
- HTER: Half Total Error Rate
- APCER/BPCER: ISO/IEC 30107-3 standard metrics
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from typing import Dict, Tuple, Optional


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER)

    EER is the point where False Acceptance Rate (FAR) equals False Rejection Rate (FRR).
    This is commonly used in biometric systems evaluation.

    Args:
        y_true: True binary labels [N] (0=live, 1=spoof)
        y_scores: Prediction scores [N] (higher = more likely spoof)

    Returns:
        eer: Equal Error Rate (as a fraction, not percentage)
        threshold: Threshold at which EER occurs
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # False Negative Rate = 1 - True Positive Rate

    # Find threshold where FPR ≈ FNR
    # This is the Equal Error Rate point
    abs_diffs = np.abs(fnr - fpr)
    min_index = np.nanargmin(abs_diffs)

    eer = fpr[min_index]
    threshold = thresholds[min_index] if min_index < len(thresholds) else thresholds[-1]

    return float(eer), float(threshold)


def calculate_hter(y_true: np.ndarray,
                   y_scores: np.ndarray,
                   threshold: Optional[float] = None) -> float:
    """
    Calculate Half Total Error Rate (HTER)

    HTER = (FAR + FRR) / 2

    Where:
    - FAR (False Acceptance Rate): Proportion of attacks accepted as genuine
    - FRR (False Rejection Rate): Proportion of genuine samples rejected

    Args:
        y_true: True binary labels (0=live, 1=spoof)
        y_scores: Prediction scores (higher = more likely spoof)
        threshold: Decision threshold (if None, use EER threshold)

    Returns:
        hter: Half Total Error Rate
    """
    if threshold is None:
        # Use EER threshold
        _, threshold = calculate_eer(y_true, y_scores)

    # Binarize predictions
    y_pred = (y_scores >= threshold).astype(int)

    # Separate live and spoof samples
    live_mask = (y_true == 0)
    spoof_mask = (y_true == 1)

    num_live = live_mask.sum()
    num_spoof = spoof_mask.sum()

    if num_live == 0 or num_spoof == 0:
        return float('nan')

    # FAR: Proportion of spoofs incorrectly classified as live
    # (spoof labeled as live = false acceptance)
    far = (y_pred[spoof_mask] == 0).sum() / num_spoof

    # FRR: Proportion of lives incorrectly classified as spoof
    # (live labeled as spoof = false rejection)
    frr = (y_pred[live_mask] == 1).sum() / num_live

    hter = (far + frr) / 2.0

    return float(hter)


def calculate_apcer_bpcer(y_true: np.ndarray,
                          y_scores: np.ndarray,
                          threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Calculate APCER and BPCER according to ISO/IEC 30107-3 standard

    APCER (Attack Presentation Classification Error Rate):
        Proportion of attack presentations incorrectly classified as bona fide
        = Number of incorrectly classified attack presentations / Total number of attack presentations
        = FP / (TN + FP) when attack=1, live=0

    BPCER (Bona Fide Presentation Classification Error Rate):
        Proportion of bona fide presentations incorrectly classified as attacks
        = Number of incorrectly classified bona fide presentations / Total number of bona fide presentations
        = FN / (FN + TP) when attack=1, live=0

    ACER (Average Classification Error Rate):
        = (APCER + BPCER) / 2

    Args:
        y_true: True binary labels (0=live/bonafide, 1=spoof/attack)
        y_scores: Prediction scores (higher = more likely spoof)
        threshold: Decision threshold (if None, use EER threshold)

    Returns:
        Dictionary with 'apcer', 'bpcer', 'acer'
    """
    if threshold is None:
        _, threshold = calculate_eer(y_true, y_scores)

    # Binarize predictions
    y_pred = (y_scores >= threshold).astype(int)

    # Separate by ground truth
    live_mask = (y_true == 0)
    spoof_mask = (y_true == 1)

    num_live = live_mask.sum()
    num_spoof = spoof_mask.sum()

    if num_live == 0 or num_spoof == 0:
        return {
            'apcer': float('nan'),
            'bpcer': float('nan'),
            'acer': float('nan')
        }

    # APCER: attacks (y_true=1) classified as live (y_pred=0)
    apcer = (y_pred[spoof_mask] == 0).sum() / num_spoof

    # BPCER: lives (y_true=0) classified as spoof (y_pred=1)
    bpcer = (y_pred[live_mask] == 1).sum() / num_live

    # ACER: Average of APCER and BPCER
    acer = (apcer + bpcer) / 2.0

    return {
        'apcer': float(apcer),
        'bpcer': float(bpcer),
        'acer': float(acer)
    }


def evaluate_fas(y_true: np.ndarray,
                 y_scores: np.ndarray,
                 threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Comprehensive FAS evaluation

    Computes all standard FAS metrics in one go.

    Args:
        y_true: True binary labels (0=live, 1=spoof)
        y_scores: Prediction scores (higher = more likely spoof)
        threshold: Optional threshold (if None, EER threshold is used)

    Returns:
        Dictionary of metrics:
            - auc: Area Under ROC Curve
            - eer: Equal Error Rate
            - threshold: EER threshold
            - hter: Half Total Error Rate at EER threshold
            - apcer: Attack Presentation Classification Error Rate
            - bpcer: Bona Fide Presentation Classification Error Rate
            - acer: Average Classification Error Rate
            - accuracy: Accuracy at EER threshold
    """
 
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()
    y_true = y_true.flatten()
    y_scores = y_scores.flatten()

 
    if len(y_true) == 0 or len(y_scores) == 0:
        return {key: float('nan') for key in [
            'auc', 'eer', 'hter', 'apcer', 'bpcer', 'acer', 'accuracy', 'threshold'
        ]}

 
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class present in y_true: {unique_labels}")
        return {key: float('nan') for key in [
            'auc', 'eer', 'hter', 'apcer', 'bpcer', 'acer', 'accuracy', 'threshold'
        ]}

    auc = roc_auc_score(y_true, y_scores)
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    if threshold is None:
        threshold = eer_threshold
    hter = calculate_hter(y_true, y_scores, threshold)
    apcer_bpcer = calculate_apcer_bpcer(y_true, y_scores, threshold)

    # Calculate accuracy at the threshold
    y_pred = (y_scores >= threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)

    return {
        'auc': float(auc),
        'eer': float(eer),
        'hter': float(hter),
        'apcer': apcer_bpcer['apcer'],
        'bpcer': apcer_bpcer['bpcer'],
        'acer': apcer_bpcer['acer'],
        'accuracy': float(accuracy),
        'threshold': float(threshold)
    }


def print_fas_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print FAS metrics

    Args:
        metrics: Dictionary of metrics from evaluate_fas()
        prefix: Prefix for print statements (e.g., "Val: " or "Test: ")
    """
    print(f"\n{'='*60}")
    print(f"{prefix}FAS Evaluation Metrics")
    print(f"{'='*60}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"EER:         {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    print(f"HTER:        {metrics['hter']:.4f} ({metrics['hter']*100:.2f}%)")
    print(f"APCER:       {metrics['apcer']:.4f} ({metrics['apcer']*100:.2f}%)")
    print(f"BPCER:       {metrics['bpcer']:.4f} ({metrics['bpcer']*100:.2f}%)")
    print(f"ACER:        {metrics['acer']:.4f} ({metrics['acer']*100:.2f}%)")
    print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Threshold:   {metrics['threshold']:.4f}")
    print(f"{'='*60}\n")


# Example usage
if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)

    # Generate dummy predictions
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)  # Random binary labels
    y_scores = np.random.rand(n_samples)  # Random scores

    # Make scores somewhat correlated with labels
    y_scores[y_true == 1] += 0.3  # Spoofs tend to have higher scores

    # Clip to [0, 1]
    y_scores = np.clip(y_scores, 0, 1)

    # Evaluate
    metrics = evaluate_fas(y_true, y_scores)

    # Print results
    print_fas_metrics(metrics, prefix="Test: ")