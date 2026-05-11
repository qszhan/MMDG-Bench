"""
Mixup for Domain Generalization
"""

import torch
import numpy as np


def multimodal_mixup(modality_features0, fusion_feat0, labels0,
                     modality_features1, fusion_feat1, labels1,
                     alpha=1.0, num_classes=None):
    """
    Apply mixup augmentation to multimodal features from two domains.

    Mixup creates virtual training examples by linearly interpolating between
    pairs of examples and their labels:
        mixed_x = λ * x_i + (1 - λ) * x_j
        mixed_y = λ * y_i + (1 - λ) * y_j

    where λ ~ Beta(α, α)

    Args:
        modality_features0: Dict of per-modality features from domain 0
                           {modality_name: tensor [batch_size, feature_dim]}
        fusion_feat0: Fused features from domain 0 [batch_size, fusion_dim]
                     Can be None for early fusion scenario where fusion hasn't happened yet
        labels0: Labels from domain 0 [batch_size]
        modality_features1: Dict of per-modality features from domain 1
        fusion_feat1: Fused features from domain 1 [batch_size, fusion_dim]
                     Can be None for early fusion scenario where fusion hasn't happened yet
        labels1: Labels from domain 1 [batch_size]
        alpha: Beta distribution parameter (default: 1.0)
               - alpha = 1.0: uniform distribution
               - alpha > 1.0: prefer mixing ratio close to 0.5
               - alpha < 1.0: prefer mixing ratio close to 0 or 1
        num_classes: Number of classes for one-hot encoding (optional)

    Returns:
        mixed_modality_features: Dict of mixed per-modality features
        mixed_fusion_feat: Mixed fused features (None if fusion_feat0/1 are None)
        mixed_labels: Mixed labels (soft labels)
        lam: The mixing coefficient used
    """
    # Determine batch size from domain 0
    batch_size0 = None
    for mod_name, feat in modality_features0.items():
        if isinstance(feat, torch.Tensor):
            batch_size0 = feat.shape[0]
            break

    if batch_size0 is None and fusion_feat0 is not None:
        batch_size0 = fusion_feat0.shape[0]

    if batch_size0 is None:
        raise ValueError("Cannot determine batch size from domain 0")

    # Determine batch size from domain 1
    batch_size1 = None
    for mod_name, feat in modality_features1.items():
        if isinstance(feat, torch.Tensor):
            batch_size1 = feat.shape[0]
            break

    if batch_size1 is None and fusion_feat1 is not None:
        batch_size1 = fusion_feat1.shape[0]

    if batch_size1 is None:
        raise ValueError("Cannot determine batch size from domain 1")

    # Use the minimum batch size if they don't match
    batch_size = min(batch_size0, batch_size1)

    # Truncate features and labels to minimum batch size if needed
    if batch_size0 > batch_size:
        modality_features0 = {k: v[:batch_size] for k, v in modality_features0.items()}
        if fusion_feat0 is not None:
            fusion_feat0 = fusion_feat0[:batch_size]
        labels0 = labels0[:batch_size]

    if batch_size1 > batch_size:
        modality_features1 = {k: v[:batch_size] for k, v in modality_features1.items()}
        if fusion_feat1 is not None:
            fusion_feat1 = fusion_feat1[:batch_size]
        labels1 = labels1[:batch_size]

    # Sample mixing coefficient from Beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Determine device from first available tensor
    device = labels0.device

    # Generate random permutation indices for mixing
    indices = torch.randperm(batch_size).to(device)

    # Mix per-modality features
    mixed_modality_features = {}
    for modality in modality_features0.keys():
        if modality in modality_features1:
            feat0 = modality_features0[modality]
            feat1 = modality_features1[modality]

            if isinstance(feat0, torch.Tensor) and isinstance(feat1, torch.Tensor):
                mixed_modality_features[modality] = (
                    lam * feat0 + (1 - lam) * feat1[indices]
                )

    # Mix fusion features (only if provided)
    mixed_fusion_feat = None
    if fusion_feat0 is not None and fusion_feat1 is not None:
        mixed_fusion_feat = lam * fusion_feat0 + (1 - lam) * fusion_feat1[indices]

    # Mix labels (create soft labels for mixup)
    # Convert labels to one-hot if needed for proper mixing
    if len(labels0.shape) == 1:
        # Labels are class indices, need to create soft labels
        # Use provided num_classes or infer from labels
        if num_classes is None:
            num_classes = max(labels0.max().item(), labels1.max().item()) + 1

        labels0_onehot = torch.zeros(batch_size, num_classes).to(labels0.device)
        labels0_onehot.scatter_(1, labels0.long().unsqueeze(1), 1)

        labels1_onehot = torch.zeros(batch_size, num_classes).to(labels1.device)
        labels1_onehot.scatter_(1, labels1.long().unsqueeze(1), 1)

        mixed_labels = lam * labels0_onehot + (1 - lam) * labels1_onehot[indices]
    else:
        # Labels are already soft (one-hot or probabilistic)
        mixed_labels = lam * labels0 + (1 - lam) * labels1[indices]

    return mixed_modality_features, mixed_fusion_feat, mixed_labels, lam


def mixup_criterion(predictions, mixed_labels, criterion):
    """
    Compute loss for mixup training.

    For mixup, we compute loss against soft labels:
        L = λ * L(pred, y_i) + (1 - λ) * L(pred, y_j)

    This is equivalent to computing cross-entropy with the mixed soft labels.

    Args:
        predictions: Model predictions [batch_size, num_classes]
        mixed_labels: Mixed soft labels [batch_size, num_classes]
        criterion: Loss function (should support soft labels, e.g., cross-entropy)

    Returns:
        loss: Scalar loss value
    """
    # If mixed_labels are soft labels (probabilities), use soft cross-entropy
    if len(mixed_labels.shape) == 2 and mixed_labels.shape[1] > 1:
        # Soft cross-entropy: -sum(target * log_softmax(pred))
        log_probs = torch.nn.functional.log_softmax(predictions, dim=1)
        loss = -(mixed_labels * log_probs).sum(dim=1).mean()
    else:
        # Standard criterion with hard labels
        loss = criterion(predictions, mixed_labels.long())

    return loss