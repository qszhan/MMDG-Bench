"""
Model factory for creating models based on task and configuration
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

# Add third_party to path for accessing external dependencies
_models_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_models_dir)
_third_party = os.path.join(_project_root, 'third_party')
if _third_party not in sys.path:
    sys.path.insert(0, _third_party)

from .base import BaseMultiModalModel, SimpleClassificationHead, MLPClassificationHead


class MultiModalModelFactory:
    """Factory for creating multi-modal models"""

    @staticmethod
    def create_model(
        task_type: str,
        modalities: List[str],
        num_classes: int,
        model_config: Dict[str, Any]
    ) -> BaseMultiModalModel:
        """
        Create a multi-modal model based on task and config

        Args:
            task_type: Task type ('action_recognition', etc.)
            modalities: List of modalities to use
            num_classes: Number of output classes
            model_config: Model configuration dict

        Returns:
            Instantiated model
        """
        if task_type == 'action_recognition':
            return MultiModalModelFactory._create_action_recognition_model(
                modalities, num_classes, model_config
            )
        elif task_type == 'face_antispoofing':
            raise NotImplementedError(
                "Face anti-spoofing models not yet implemented"
            )
        elif task_type == 'emotion_recognition':
            raise NotImplementedError(
                "Emotion recognition models not yet implemented"
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @staticmethod
    def _create_action_recognition_model(
        modalities: List[str],
        num_classes: int,
        model_config: Dict[str, Any]
    ) -> 'ActionRecognitionModel':
        """Create action recognition model"""
        from .action_recognition import ActionRecognitionModel

        return ActionRecognitionModel(
            modalities=modalities,
            num_classes=num_classes,
            model_config=model_config
        )


class BackboneFactory:
    """Factory for creating backbone networks"""

    @staticmethod
    def create_video_backbone(
        backbone_name: str,
        pretrained: bool = True,
        config_path: Optional[str] = None
    ) -> nn.Module:
        """
        Create video backbone (SlowFast, ResNet3D, etc.)

        Args:
            backbone_name: Name of backbone ('slowfast', 'slowonly', etc.)
            pretrained: Whether to use pretrained weights
            config_path: Path to MMAction2 config file

        Returns:
            Backbone model
        """
        # Import MMAction2 models
        from mmaction.models import build_model
        from mmcv import Config

        if config_path is None:
            # Use default config paths
            base_path = '/path/to/configs'

            if backbone_name == 'slowfast':
                config_path = f'{base_path}/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py'
            elif backbone_name == 'slowonly':
                config_path = f'{base_path}/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py'
            else:
                raise ValueError(f"Unknown video backbone: {backbone_name}")

        cfg = Config.fromfile(config_path)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

        return model

    @staticmethod
    def create_audio_backbone(
        backbone_name: str = 'vggsound',
        pretrained: bool = True,
        pretrained_path: Optional[str] = None
    ) -> nn.Module:
        """
        Create audio backbone

        Args:
            backbone_name: Name of audio backbone
            pretrained: Whether to use pretrained weights
            pretrained_path: Path to pretrained weights

        Returns:
            Audio backbone model
        """
        if backbone_name == 'vggsound':
            # Import VGGSound model (from third_party)
            from VGGSound.models.resnet import AudioAttGenModule

            model = AudioAttGenModule()

            if pretrained and pretrained_path:
                import torch
                checkpoint = torch.load(pretrained_path)
                model.load_state_dict(checkpoint, strict=False)

            return model
        else:
            raise ValueError(f"Unknown audio backbone: {backbone_name}")


class FusionFactory:
    """Factory for creating fusion modules"""

    @staticmethod
    def create_fusion_module(
        fusion_type: str,
        modality_dims: Dict[str, int],
        fusion_config: Optional[Dict[str, Any]] = None
    ) -> nn.Module:
        """
        Create fusion module

        Args:
            fusion_type: Type of fusion ('concat', 'attention', 'late', etc.)
            modality_dims: Dict mapping modality names to feature dimensions
            fusion_config: Additional fusion configuration

        Returns:
            Fusion module
        """
        if fusion_config is None:
            fusion_config = {}

        if fusion_type == 'concat':
            return ConcatFusion(modality_dims)
        elif fusion_type == 'attention':
            return AttentionFusion(modality_dims, **fusion_config)
        elif fusion_type == 'late':
            # Late fusion with separate classifiers
            return LateFusion(modality_dims, **fusion_config)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")


class ConcatFusion(nn.Module):
    """Simple concatenation fusion"""

    def __init__(self, modality_dims: Dict[str, int]):
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = sum(modality_dims.values())

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate features from all modalities"""
        import torch
        features = [modality_features[mod] for mod in sorted(modality_features.keys())]
        return torch.cat(features, dim=1)

    def get_output_dim(self) -> int:
        return self.output_dim


class AttentionFusion(nn.Module):
    """Attention-based fusion"""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 4
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim

        # Project each modality to common dimension
        self.projections = nn.ModuleDict({
            mod: nn.Linear(dim, hidden_dim)
            for mod, dim in modality_dims.items()
        })

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.output_dim = hidden_dim

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse features using attention"""
        import torch

        # Project to common dimension
        projected = []
        for mod_name in sorted(modality_features.keys()):
            feat = self.projections[mod_name](modality_features[mod_name])
            projected.append(feat.unsqueeze(1))  # (batch, 1, hidden)

        # Stack modalities
        stacked = torch.cat(projected, dim=1)  # (batch, num_modalities, hidden)

        # Self-attention
        attended, _ = self.attention(stacked, stacked, stacked)

        # Mean pooling across modalities
        fused = attended.mean(dim=1)

        return fused

    def get_output_dim(self) -> int:
        return self.output_dim


class LateFusion(nn.Module):
    """Late fusion with averaging predictions"""

    def __init__(
        self,
        modality_dims: Dict[str, int],
        num_classes: int,
        learnable_weights: bool = False
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.num_classes = num_classes

        # Separate classifier for each modality
        self.classifiers = nn.ModuleDict({
            mod: nn.Linear(dim, num_classes)
            for mod, dim in modality_dims.items()
        })

        # Optional learnable fusion weights
        if learnable_weights:
            import torch
            self.fusion_weights = nn.Parameter(
                torch.ones(len(modality_dims)) / len(modality_dims)
            )
        else:
            self.fusion_weights = None

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        return_logits: bool = True
    ):
        """
        Late fusion by averaging predictions

        Args:
            modality_features: Dict of modality features
            return_logits: If True, return fused logits; else return dict of per-modality logits
        """
        import torch

        logits_dict = {}
        for mod_name in sorted(modality_features.keys()):
            logits = self.classifiers[mod_name](modality_features[mod_name])
            logits_dict[mod_name] = logits

        if return_logits:
            # Average logits
            if self.fusion_weights is not None:
                # Weighted average
                weights = torch.softmax(self.fusion_weights, dim=0)
                fused = sum(
                    weights[i] * logits
                    for i, logits in enumerate(logits_dict.values())
                )
            else:
                # Uniform average
                fused = sum(logits_dict.values()) / len(logits_dict)

            return fused
        else:
            return logits_dict