"""
Domain Generalization Methods

This module contains various domain generalization methods for addressing domain gap:
- MIRO: Mutual-Information Regularization
- Domain Contrastive Loss: Contrastive learning for domain-invariant features
- MMD: Maximum Mean Discrepancy
"""

from .domain_contrastive_loss import (
    ModalContrastiveLoss,
    compute_modal_contrastive_losses,
    WeightedModalContrastiveLoss
)

from .mmd import (
    compute_mmd_loss,
    compute_multi_kernel_mmd,
    MMDLoss
)

from . import miro

__all__ = [
    # Domain Contrastive Loss
    'ModalContrastiveLoss',
    'compute_modal_contrastive_losses',
    'WeightedModalContrastiveLoss',
    # MMD
    'compute_mmd_loss',
    'compute_multi_kernel_mmd',
    'MMDLoss',
    # MIRO
    'miro',
]