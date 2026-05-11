"""
Action Recognition Model Implementation

Multi-modal model for action recognition with RGB, flow, and audio modalities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

from .base import BaseMultiModalModel, MLPClassificationHead
from .factory import BackboneFactory, FusionFactory


class ActionRecognitionModel(BaseMultiModalModel):
    """
    Multi-modal action recognition model

    Supports RGB video, optical flow, and audio modalities
    """

    def __init__(
        self,
        modalities: List[str],
        num_classes: int,
        model_config: Dict[str, Any]
    ):
        """
        Args:
            modalities: List of modalities (['rgb', 'flow', 'audio'])
            num_classes: Number of action classes
            model_config: Model configuration containing:
                - backbone_configs: Dict of backbone configurations per modality
                - fusion_type: Type of fusion ('concat', 'attention', etc.)
                - head_config: Classification head configuration
        """
        super().__init__(
            modalities=modalities,
            num_classes=num_classes,
            task_type='action_recognition'
        )

        self.model_config = model_config

        # Initialize backbones for each modality
        self._init_backbones()

        # Get feature dimensions
        self.feature_dims = self._get_feature_dims()

        # Initialize fusion module
        fusion_type = model_config.get('fusion_type', 'concat')
        fusion_config = model_config.get('fusion_config', {})
        self.fusion = FusionFactory.create_fusion_module(
            fusion_type=fusion_type,
            modality_dims=self.feature_dims,
            fusion_config=fusion_config
        )

        # Initialize classification head
        head_config = model_config.get('head_config', {})
        fused_dim = self.fusion.get_output_dim()

        self.head = MLPClassificationHead(
            input_dim=fused_dim,
            num_classes=num_classes,
            **head_config
        )

    def _init_backbones(self):
        """Initialize backbone networks for each modality"""
        backbone_configs = self.model_config.get('backbone_configs', {})

        for modality in self.modalities:
            if modality == 'rgb':
                backbone_cfg = backbone_configs.get('rgb', {})
                self.backbones['rgb'] = BackboneFactory.create_video_backbone(
                    backbone_name=backbone_cfg.get('name', 'slowfast'),
                    pretrained=backbone_cfg.get('pretrained', True),
                    config_path=backbone_cfg.get('config_path', None)
                )

            elif modality == 'flow':
                backbone_cfg = backbone_configs.get('flow', {})
                self.backbones['flow'] = BackboneFactory.create_video_backbone(
                    backbone_name=backbone_cfg.get('name', 'slowonly'),
                    pretrained=backbone_cfg.get('pretrained', True),
                    config_path=backbone_cfg.get('config_path', None)
                )

            elif modality == 'audio':
                backbone_cfg = backbone_configs.get('audio', {})
                self.backbones['audio'] = BackboneFactory.create_audio_backbone(
                    backbone_name=backbone_cfg.get('name', 'vggsound'),
                    pretrained=backbone_cfg.get('pretrained', True),
                    pretrained_path=backbone_cfg.get('pretrained_path', None)
                )

    def _get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each modality"""
        feature_dims = {}

        if 'rgb' in self.modalities:
            # SlowFast typically outputs 2304 (slow: 2048, fast: 256)
            # SlowOnly outputs 2048
            feature_dims['rgb'] = self.model_config.get('rgb_feature_dim', 2304)

        if 'flow' in self.modalities:
            # SlowOnly for flow outputs 2048
            feature_dims['flow'] = self.model_config.get('flow_feature_dim', 2048)

        if 'audio' in self.modalities:
            # VGGSound outputs 512
            feature_dims['audio'] = self.model_config.get('audio_feature_dim', 512)

        return feature_dims

    def forward_modality(
        self,
        modality_name: str,
        modality_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features from a single modality

        Args:
            modality_name: Name of modality ('rgb', 'flow', 'audio')
            modality_data: Input tensor for this modality

        Returns:
            Feature tensor
        """
        backbone = self.backbones[modality_name]

        with torch.set_grad_enabled(self.training):
            if modality_name == 'audio':
                # Audio model returns (feat_before_avgpool, feat, class_output)
                _, features, _ = backbone(modality_data.unsqueeze(1))  # Add channel dim
                return features
            else:
                # Video models (RGB/Flow)
                # Use backbone's feature extraction method
                features = backbone.backbone.get_feature(modality_data)

                # Handle SlowFast (returns tuple of slow and fast features)
                if isinstance(features, tuple):
                    # Concatenate slow and fast features
                    features = torch.cat([f.flatten(1) for f in features], dim=1)
                else:
                    # Single pathway (SlowOnly)
                    features = features.flatten(1)

                return features

    def fuse_features(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse features from multiple modalities

        Args:
            modality_features: Dict of modality features

        Returns:
            Fused feature tensor
        """
        if len(modality_features) == 0:
            raise ValueError("No modality features provided")
        import pdb
        pdb.set_trace()

        return self.fusion(modality_features)

    def get_feature_dim(self, modality: str) -> int:
        """Get feature dimension for a modality"""
        return self.feature_dims.get(modality, 0)


# Simpler model for testing without full backbones
class SimpleActionRecognitionModel(BaseMultiModalModel):
    """
    Simplified action recognition model for testing

    Uses simple CNNs instead of full backbones
    """

    def __init__(
        self,
        modalities: List[str],
        num_classes: int,
        hidden_dim: int = 256
    ):
        super().__init__(
            modalities=modalities,
            num_classes=num_classes,
            task_type='action_recognition'
        )

        self.hidden_dim = hidden_dim

        # Simple encoders for each modality
        if 'rgb' in modalities:
            self.backbones['rgb'] = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(64, hidden_dim)
            )

        if 'flow' in modalities:
            self.backbones['flow'] = nn.Sequential(
                nn.Conv3d(2, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(64, hidden_dim)
            )

        if 'audio' in modalities:
            self.backbones['audio'] = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, hidden_dim)
            )

        # Simple concatenation fusion
        fused_dim = hidden_dim * len(modalities)
        self.head = MLPClassificationHead(fused_dim, num_classes)

    def forward_modality(self, modality_name: str, modality_data: torch.Tensor):
        return self.backbones[modality_name](modality_data)

    def fuse_features(self, modality_features: Dict[str, torch.Tensor]):
        features = [modality_features[mod] for mod in sorted(modality_features.keys())]
        return torch.cat(features, dim=1)

    def get_feature_dim(self, modality: str) -> int:
        return self.hidden_dim
