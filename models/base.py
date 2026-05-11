"""
Base model classes for MMDG-Bench
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseMultiModalModel(nn.Module, ABC):
    """
    Base class for multi-modal models

    Handles multiple input modalities and produces task-specific outputs
    """

    def __init__(
        self,
        modalities: List[str],
        num_classes: int,
        task_type: str,
        **kwargs
    ):
        """
        Args:
            modalities: List of input modalities
            num_classes: Number of output classes
            task_type: Type of task ('action_recognition', etc.)
        """
        super().__init__()

        self.modalities = modalities
        self.num_classes = num_classes
        self.task_type = task_type

        # Modality-specific backbones (to be initialized by subclasses)
        self.backbones = nn.ModuleDict()

        # Classification head
        self.head = None

    @abstractmethod
    def forward_modality(
        self,
        modality_name: str,
        modality_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for a single modality

        Args:
            modality_name: Name of the modality
            modality_data: Input tensor for this modality

        Returns:
            Feature tensor
        """
        pass

    @abstractmethod
    def fuse_features(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse features from multiple modalities

        Args:
            modality_features: Dict mapping modality names to feature tensors

        Returns:
            Fused feature tensor
        """
        pass

    def forward(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the model

        Args:
            modality_inputs: Dict mapping modality names to input tensors
            return_features: Whether to return intermediate features

        Returns:
            If return_features=False:
                predictions (batch_size, num_classes)
            If return_features=True:
                (predictions, feature_dict) where feature_dict contains:
                    - modality-specific features
                    - fused features
        """
        modality_features = {}

        # Extract features from each modality
        for modality_name in self.modalities:
            if modality_name in modality_inputs:
                modality_data = modality_inputs[modality_name]

                # Handle placeholder values (0 for missing modalities)
                if isinstance(modality_data, int) or \
                   (isinstance(modality_data, torch.Tensor) and modality_data.numel() == 1):
                    continue

                features = self.forward_modality(modality_name, modality_data)
                modality_features[modality_name] = features

        # Fuse features
        fused_features = self.fuse_features(modality_features)

        # Classification
        predictions = self.head(fused_features)

        if return_features:
            feature_dict = {
                **modality_features,
                'fused': fused_features
            }
            return predictions, feature_dict
        else:
            return predictions

    def get_feature_dim(self, modality: str) -> int:
        """Get feature dimension for a specific modality"""
        raise NotImplementedError("Subclasses should implement get_feature_dim")


class SimpleClassificationHead(nn.Module):
    """Simple linear classification head"""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


class MLPClassificationHead(nn.Module):
    """MLP classification head with feature decomposition"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.5,
        use_feature_split: bool = False
    ):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension (if None, uses input_dim // 2)
            dropout: Dropout probability
            use_feature_split: Whether to split features into specific/shared
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim // 2

        self.use_feature_split = use_feature_split

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input features
            return_embeddings: Whether to return embeddings before classification

        Returns:
            If return_embeddings=False: predictions
            If return_embeddings=True: (predictions, embeddings)
        """
        embeddings = self.mlp(x)
        predictions = self.classifier(embeddings)

        if return_embeddings:
            if self.use_feature_split:
                # Split embeddings: first half = specific, second half = shared
                split_point = embeddings.shape[1] // 2
                specific = embeddings[:, :split_point]
                shared = embeddings[:, split_point:]
                return predictions, {'specific': specific, 'shared': shared, 'full': embeddings}
            else:
                return predictions, embeddings
        else:
            return predictions
