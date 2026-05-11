"""
Face Anti-Spoofing Model

Multi-modal model for face anti-spoofing using ViT backbones.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path

from .base import BaseMultiModalModel
from .factory import FusionFactory


class FaceAntiSpoofingModel(BaseMultiModalModel):
    """
    Multi-modal Face Anti-Spoofing Model

    Architecture:
        Input: Dict of modality tensors
            RGB [B, 3, 224, 224]
            Depth [B, 3, 224, 224]
            IR [B, 3, 224, 224]
                    ↓
        Per-Modality ViT-Base Backbones
            RGB → ViT → [B, 768]
            Depth → ViT → [B, 768]
            IR → ViT → [B, 768]
                    ↓
        Fusion Module (Concat/Attention/Late)
            Fused features [B, fusion_dim]
                    ↓
        Binary Classification Head
            [B, 1] logit (for BCEWithLogitsLoss)

    Note: This model can work with either:
        1. Pre-extracted features [B, 768] per modality
        2. Raw images [B, 3, 224, 224] per modality
    """

    def __init__(self,
                 modalities: List[str],
                 num_classes: int,
                 model_config: Dict,
                 task_name: str = 'face_antispoofing',
                 use_pretrained_features: bool = False,
                 backbones_only: bool = False,
                 shared_backbone: bool = False):
        """
        Args:
            modalities: List of modalities (e.g., ['rgb', 'depth', 'ir'])
            num_classes: Number of classes (typically 2 for binary: live=0, spoof=1)
            model_config: Configuration dict containing:
                - backbone_configs: Config for each modality's backbone
                - fusion_type: 'concat', 'attention', or 'late'
                - head_config: Config for classification head
                - use_pretrained_vit: Whether to use pretrained ViT (default True)
                - shared_backbone: Whether to use a single shared ViT for all modalities
            task_name: Task name
            use_pretrained_features: If True, skip backbone (input is already features)
            backbones_only: If True, only create backbones (for early fusion with EarlyFusionDG)
            shared_backbone: If True, use single shared ViT for all modalities (like Flex-Modal-FAS)
        """

        super().__init__(modalities, num_classes, task_name)

        self.model_config = model_config
        self.use_pretrained_features = use_pretrained_features
        self.backbones_only = backbones_only
        self.shared_backbone = shared_backbone or model_config.get('shared_backbone', False)

        # Initialize backbones (only if not using pre-extracted features)
        if not use_pretrained_features:
            self._init_backbones()
        else:
            print("Using pre-extracted features, skipping backbone initialization")

        # Initialize fusion module and head only if not in backbones_only mode
        if not backbones_only:
            # Initialize fusion module
            self._init_fusion()

            # Initialize classification head
            self._init_head()
        else:
            print("Backbones-only mode: Skipping fusion and head initialization")
            print("  (Fusion and classification will be handled by EarlyFusionDG algorithm)")

    def _init_backbones(self):
        """Initialize ViT-Base backbone (shared or per-modality) using timm"""
        import timm

        use_pretrained_vit = self.model_config.get('use_pretrained_vit', True)
        local_pretrained_path = self.model_config.get('local_pretrained_path', None)

        print(f"Initializing ViT-Base backbones for modalities: {self.modalities}")
        print(f"  Pretrained: {use_pretrained_vit}")
        print(f"  Shared backbone: {self.shared_backbone}")
        if local_pretrained_path:
            print(f"  Local pretrained path: {local_pretrained_path}")

        # Determine num_classes for ViT backbone
        # For backbones_only mode (Early/Late Fusion), use num_classes to enable per-modality classification
        # - Early Fusion: doesn't use per-modality logits, but harmless
        # - Late Fusion: needs per-modality logits for computing losses
        num_classes_backbone = self.num_classes if self.backbones_only else 0

        print(f"  ViT num_classes: {num_classes_backbone} ({'per-modality heads' if num_classes_backbone > 0 else 'features only'})")

        if self.shared_backbone:
            # Create single shared ViT for all modalities (like Flex-Modal-FAS)
            if use_pretrained_vit:
                # Create ViT model using timm
                shared_vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes_backbone)

                # Load local pretrained weights if provided
                if local_pretrained_path and os.path.exists(local_pretrained_path):
                    print(f"  Loading pretrained weights from: {local_pretrained_path}")
                    state_dict = torch.load(local_pretrained_path, map_location='cpu')
                    # Remove 'head' weights if present (head will be randomly initialized for FAS task)
                    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
                    missing, unexpected = shared_vit.load_state_dict(state_dict, strict=False)
                    print(f"  ✓ Loaded pretrained weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
                else:
                    print(f"  Warning: No local pretrained weights found, using random initialization")
            else:
                # Initialize ViT from scratch
                shared_vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes_backbone)

            # Store reference to shared backbone
            self.shared_vit = shared_vit
            print(f"  ✓ Shared backbone initialized (1 ViT for {len(self.modalities)} modalities)")

        else:
            # Create separate ViT for each modality
            for modality in self.modalities:
                if use_pretrained_vit:
                    # Create ViT model using timm
                    vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes_backbone)

                    # Load local pretrained weights if provided
                    if local_pretrained_path and os.path.exists(local_pretrained_path):
                        print(f"  Loading pretrained weights for {modality} from: {local_pretrained_path}")
                        state_dict = torch.load(local_pretrained_path, map_location='cpu')
                        # Remove 'head' weights if present (head will be randomly initialized for FAS task)
                        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
                        missing, unexpected = vit.load_state_dict(state_dict, strict=False)
                        print(f"  ✓ Loaded pretrained weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
                    else:
                        print(f"  Warning: No local pretrained weights found for {modality}, using random initialization")
                else:
                    # Initialize ViT from scratch
                    vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes_backbone)

                self.backbones[modality] = vit

            print(f"  ✓ Backbones initialized ({len(self.modalities)} separate ViTs)")

    def _init_fusion(self):
        """Initialize fusion module"""
        fusion_type = self.model_config.get('fusion_type', 'concat')

        # Feature dimensions (ViT-Base outputs 768-dim class token)
        feature_dims = {m: 768 for m in self.modalities}

        fusion_config = self.model_config.get('fusion_config', {})

        print(f"Initializing fusion module: {fusion_type}")

        self.fusion = FusionFactory.create_fusion_module(
            fusion_type,
            feature_dims,
            fusion_config=fusion_config
        )

        fusion_output_dim = self.fusion.get_output_dim()
        print(f"  Fusion output dim: {fusion_output_dim}")

    def _init_head(self):
        """Initialize classification head"""
        head_config = self.model_config.get('head_config', {})

        input_dim = self.fusion.get_output_dim()
        dropout = head_config.get('dropout', 0.5)
        hidden_dim = head_config.get('hidden_dim', 256)

        print(f"Initializing classification head:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {self.num_classes}")
        print(f"  Dropout: {dropout}")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes)
        )

    def forward_modality(self, modality_name: str, modality_data: torch.Tensor):
        """
        Extract features from a single modality using ViT (timm version)

        Args:
            modality_name: 'rgb', 'depth', or 'ir'
            modality_data: [B, 3, 224, 224] for raw images
                          OR [B, 768] for pre-extracted features

        Returns:
            If backbones_only=True (for EarlyFusionDG):
                (predict, features): Tuple of (None, [B, 768])
            If backbones_only=False:
                features: [B, 768]
        """
        if self.use_pretrained_features:
            # Input is already features [B, 768]
            features = modality_data
        else:
            # Extract features using ViT (shared or modality-specific)
            if self.shared_backbone:
                # Use shared ViT for all modalities
                vit = self.shared_vit
            else:
                # Use modality-specific ViT
                vit = self.backbones[modality_name]

            # timm ViT forward pass
            # Behavior depends on num_classes:
            # - num_classes=0: vit(x) returns features [B, 768]
            # - num_classes>0: vit(x) returns logits [B, num_classes]
            #                  vit.forward_features(x) returns features [B, 768]
            if hasattr(vit, 'head') and vit.head is not None and hasattr(vit, 'forward_features'):
                # ViT with classification head (num_classes > 0)
                features = vit.forward_features(modality_data)  # [B, 768]
                if features.dim() ==3:
                    features = features[:, 0]
                logits = vit.head(features)  # [B, num_classes]
            else:
                # ViT without head (num_classes = 0)
                features = vit(modality_data)  # [B, 768]
                logits = None

        # Return format depends on mode
        if self.backbones_only:
            # For EarlyFusionDG/LateFusionDG: return (logits, features)
            # - Early Fusion: ignores logits (uses _ to receive it)
            # - Late Fusion: uses logits for per-modality classification loss
            return logits, features
        else:
            # For standalone use: just return features
            return features

    def fuse_features(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple modalities

        Args:
            modality_features: Dict of {modality_name: [B, 768]}

        Returns:
            fused_features: [B, fusion_dim]
        """
        return self.fusion(modality_features)

    def forward(self,
                modality_inputs: Dict[str, torch.Tensor],
                return_features: bool = False):
        """
        Forward pass

        Args:
            modality_inputs: Dict of {modality_name: tensor}
                - If use_pretrained_features=False: [B, 3, 224, 224] per modality
                - If use_pretrained_features=True: [B, 768] per modality
            return_features: If True, return features before classification

        Returns:
            logits: [B, 1] for BCEWithLogitsLoss
            features (optional): [B, fusion_dim] if return_features=True
        """
        if self.backbones_only:
            raise RuntimeError(
                "Cannot call forward() in backbones_only mode. "
                "Use forward_modality() instead, or pass backbones_only=False."
            )

        # Extract features from each modality
        modality_features = {}
        for modality_name, modality_data in modality_inputs.items():
            if modality_name in self.modalities:
                features = self.forward_modality(modality_name, modality_data)
                modality_features[modality_name] = features

        # Fuse features
        fused_features = self.fuse_features(modality_features)

        # Classification
        logits = self.head(fused_features)

        if return_features:
            return logits, fused_features
        else:
            return logits

    def predict_proba(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get probability predictions (apply sigmoid to logits)

        Args:
            modality_inputs: Dict of modality tensors

        Returns:
            probs: [B, 1] probabilities (0-1 range)
        """
        logits = self.forward(modality_inputs)
        probs = torch.sigmoid(logits)
        return probs

    def predict(self, modality_inputs: Dict[str, torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
        """
        Get binary predictions

        Args:
            modality_inputs: Dict of modality tensors
            threshold: Decision threshold (default 0.5)

        Returns:
            predictions: [B, 1] binary predictions (0=live, 1=spoof)
        """
        probs = self.predict_proba(modality_inputs)
        predictions = (probs >= threshold).long()
        return predictions