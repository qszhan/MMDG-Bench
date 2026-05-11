"""
Contrastive loss functions for bridging modal gaps in multi-modal learning

These losses help align features from different modalities by:
1. Pulling together features from the same class (across modalities)
2. Pushing apart features from different classes
 
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, sample_weights=None):
        """
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
            sample_weights: tensor of shape [bsz] with weights for each sample.

        Returns:
            A loss scalar.
        """
        device = torch.device('cuda' if features.is_cuda else 'cpu')

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # Add epsilon to prevent log(0) which causes -inf/NaN
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # compute mean of log-likelihood over positive
        # Add epsilon to prevent division by zero
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)

        # Apply sample weights if provided
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)
            # Repeat weights for each anchor (if using multiple views)
            weights_expanded = sample_weights.unsqueeze(0).repeat(anchor_count, 1)
            loss = loss * weights_expanded

        loss = loss.mean()

        return loss


class HardSupConLoss(SupConLoss):
    """
    Hard Supervised Contrastive Loss
    Focus on hard negative samples
    """
    def __init__(self, temperature=0.07, base_temperature=0.07, hard_neg_ratio=0.5):
        super().__init__(temperature, 'all', base_temperature)
        self.hard_neg_ratio = hard_neg_ratio

    def forward(self, features, labels=None):
        """
        Compute hard supervised contrastive loss

        Args:
            features: [bsz, n_views, dim]
            labels: [bsz]

        Returns:
            Loss scalar
        """
        device = features.device
        batch_size = features.shape[0]
        contrast_count = features.shape[1]

        # Flatten features
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, dim]
        anchor_feature = contrast_feature

        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(contrast_count, contrast_count)

        # Compute similarities
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )

        # Find hard negatives
        neg_mask = (1 - mask) * logits_mask  # Negative samples
        neg_logits = anchor_dot_contrast * neg_mask

        # Select top-k hardest negatives
        num_negatives = int(neg_mask.sum(1).max().item())
        k = max(1, int(num_negatives * self.hard_neg_ratio))

        # For each anchor, find k hardest negatives
        hard_neg_mask = torch.zeros_like(neg_mask)
        for i in range(anchor_dot_contrast.shape[0]):
            if neg_mask[i].sum() > 0:
                _, hard_indices = torch.topk(neg_logits[i], k=min(k, int(neg_mask[i].sum())))
                hard_neg_mask[i, hard_indices] = 1

        # Combine positive mask with hard negative mask
        combined_mask = mask + hard_neg_mask
        combined_mask = combined_mask * logits_mask

        # Compute loss
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * combined_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean over positive samples
        pos_mask = mask * logits_mask
        mask_sum = pos_mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / mask_sum

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
