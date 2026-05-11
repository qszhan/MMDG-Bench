"""
Projection heads and cross-modal translation modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectHead(nn.Module):
    """
    Projection head to map features to a common embedding space

    Used to project different modalities to the same dimension for:
    - Contrastive learning
    - Cross-modal alignment
    - Fusion

    Architecture: 3-layer MLP with BatchNorm and L2 normalization
    """
    def __init__(self, input_dim, hidden_dim=2048, out_dim=128):
        super(ProjectHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        # Project to embedding space and L2 normalize
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x