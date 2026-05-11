"""
Modal Contrastive Loss for Multi-Modal Domain Generalization

This module implements contrastive loss computation for different modalities (video, flow, audio, combined).
The contrastive loss treats samples from the same class as positive pairs and samples from different classes as negative pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalContrastiveLoss(nn.Module):
    """
    Contrastive loss for modal features where samples from the same class are positive pairs
    and samples from different classes are negative pairs.
    """

    def __init__(self, temperature=0.1, eps=1e-8):
        """
        Args:
            temperature (float): Temperature parameter for softmax normalization
            eps (float): Small value to prevent division by zero
        """
        super(ModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        """
        Compute contrastive loss for given features and labels

        Args:
            features (torch.Tensor): Feature tensor of shape [N, D] where N is batch size, D is feature dimension
            labels (torch.Tensor): Label tensor of shape [N] containing class labels

        Returns:
            torch.Tensor: Scalar contrastive loss
        """
        if features is None or features.size(0) == 0:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)

        # L2 normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create positive mask (same class)
        labels_expanded = labels.unsqueeze(1).expand(-1, labels.size(0))
        positive_mask = (labels_expanded == labels_expanded.T).float()

        # Remove diagonal (self-similarity)
        positive_mask = positive_mask - torch.eye(labels.size(0), device=labels.device)

        # Compute exponential similarities
        exp_sim = torch.exp(similarity_matrix)

        # Denominator: sum of similarities to all samples except self
        denominator = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim)

        # Numerator: sum of similarities to positive samples
        numerator = torch.sum(exp_sim * positive_mask, dim=1)

        # Compute contrastive loss
        contrastive_loss = -torch.log(numerator / (denominator + self.eps) + self.eps)

        # Only consider samples that have positive pairs
        valid_samples = (torch.sum(positive_mask, dim=1) > 0)
        if valid_samples.sum() > 0:
            return contrastive_loss[valid_samples].mean()
        else:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)


def compute_modal_contrastive_losses(modal_features, labels, temperature=0.1):
    """
    Compute contrastive losses for all available modalities

    Args:
        modal_features (dict or torch.Tensor): Dictionary containing modal features
                                              Keys: 'video', 'flow', 'audio', 'combined'
                                              Values: torch.Tensor of shape [N, D]
                                              OR a single torch.Tensor of shape [N, D]
        labels (torch.Tensor): Class labels of shape [N]
        temperature (float): Temperature parameter for contrastive loss

    Returns:
        dict: Dictionary containing contrastive loss for each modality
        float: Total contrastive loss (sum of all modal losses)
    """
    modal_contrastive_losses = {}
    contrastive_loss_fn = ModalContrastiveLoss(temperature=temperature)

    # Handle tensor input (when use_modal_specific_dg == False)
    if isinstance(modal_features, torch.Tensor):
        loss = contrastive_loss_fn(modal_features, labels)
        modal_contrastive_losses['combined'] = loss
        return modal_contrastive_losses, loss

    # Handle dictionary input (when use_modal_specific_dg == True)
    total_loss = 0.0
    valid_modalities = 0

    for modality, features in modal_features.items():
        if features is not None and features.size(0) > 0:
            loss = contrastive_loss_fn(features, labels)
            modal_contrastive_losses[modality] = loss
            total_loss += loss
            valid_modalities += 1

    # Average loss across modalities
    if valid_modalities > 0:
        avg_loss = total_loss / valid_modalities
    else:
        avg_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)

    return modal_contrastive_losses, avg_loss


class WeightedModalContrastiveLoss(nn.Module):
    """
    Weighted contrastive loss that can assign different weights to different modalities
    """

    def __init__(self, temperature=0.1, modal_weights=None):
        """
        Args:
            temperature (float): Temperature parameter
            modal_weights (dict): Dictionary of weights for each modality
                                 Default: {'video': 1.0, 'flow': 1.0, 'audio': 1.0, 'combined': 1.0}
        """
        super(WeightedModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.modal_weights = modal_weights or {
            'video': 1.0, 'flow': 1.0, 'audio': 1.0, 'combined': 1.0
        }
        self.contrastive_loss_fn = ModalContrastiveLoss(temperature=temperature)

    def forward(self, modal_features, labels):
        """
        Compute weighted contrastive losses

        Args:
            modal_features (dict): Dictionary containing modal features
            labels (torch.Tensor): Class labels

        Returns:
            dict: Individual modal losses
            torch.Tensor: Weighted total loss
        """
        modal_losses = {}
        total_loss = 0.0
        total_weight = 0.0

        for modality, features in modal_features.items():
            if features is not None and features.size(0) > 0:
                loss = self.contrastive_loss_fn(features, labels)
                weight = self.modal_weights.get(modality, 1.0)

                modal_losses[modality] = loss
                total_loss += weight * loss
                total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            total_loss = total_loss / total_weight
        else:
            total_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)

        return modal_losses, total_loss