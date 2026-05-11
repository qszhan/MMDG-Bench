"""
Early Fusion Domain Generalization Algorithm with VIT-Base

Key differences from Late Fusion:
1. Domain gap mitigation PER MODALITY first
   - Each modality learns domain-invariant features
   - Per-modality DG classifiers
2. Modal gap mitigation AFTER domain gap
   - Cross-modal translation on domain-invariant features
   - Supervised contrastive learning
   - Fusion of aligned features

Training flow:
1. Extract features from each modality (D0 and D1 separately)
2. Per-modality domain gap mitigation:
   - Concat D0 and D1 features for each modality
   - Per-modality DG classifier on concatenated features
3. Project to common space
4. Modal gap mitigation:
   - Cross-modal translation
   - Supervised contrastive learning
5. Fusion and final classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from .base import BaseAlgorithm
from .MMmethods.modal_contrastive_loss import SupConLoss
from .algorithm_utils.projectors import ProjectHead
from .MMmethods.CMT import CrossModalTranslation
from .MMmethods.mmfusion import AttentionFusion, BottleneckFusion 
from .algorithm_utils.sam import SAM
from .MMmethods import GBL
from .DG_methods.mixup import multimodal_mixup, mixup_criterion
from .DG_methods.domain_contrastive_loss import compute_modal_contrastive_losses
from .DG_methods import miro
from .DG_methods.mmd import compute_mmd_loss


# ============================================================================
# VideoMAEv2 Layer-wise LR Decay Utilities
# ============================================================================

def get_num_layer_for_vit(var_name, num_max_layer):
    """Get layer ID for ViT/VideoMAEv2 parameters (for layer-wise LR decay)"""
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    """Assign layer-wise learning rate scales for VideoMAEv2"""

    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    """
    Cosine scheduler with warmup (step-level)

    Args:
        base_value: Peak value after warmup
        final_value: Final value at end of training
        epochs: Total epochs
        niter_per_ep: Number of iterations per epoch
        warmup_epochs: Number of warmup epochs
        start_warmup_value: Starting value for warmup
        warmup_steps: Alternative to warmup_epochs (uses steps instead)

    Returns:
        schedule: Array of length (epochs * niter_per_ep) with scheduled values
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps

    if warmup_epochs > 0 or warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([
        final_value + 0.5 * (base_value - final_value) *
        (1 + math.cos(math.pi * i / len(iters))) for i in iters
    ])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


class Encoder(nn.Module):
    """Simple encoder for per-modality DG classifiers"""

    def __init__(self, input_dim=2304, out_dim=8, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        return self.enc_net(feat)


class EarlyFusionDG(BaseAlgorithm):
    """
    Early Fusion Domain Generalization Algorithm

    Architecture (Domain-then-Modal):
    1. Extract features from each modality (D0 and D1)
    2. Per-modality domain gap mitigation:
       - Concat [feat_D0, feat_D1] per modality
       - Per-modality DG classifier
    3. Project domain-invariant features to common space
    4. Modal gap mitigation:
       - Cross-modal translation
       - Supervised contrastive learning
    5. Bottleneck fusion
    6. Final classification

    Losses:
    - Per-modality DG classification losses
    - Fused classification loss
    - Cross-modal translation loss (on domain-invariant features)
    - Supervised contrastive loss (on domain-invariant features)
    """

    def __init__(
            self,
            model,
            config,
            modalities: List[str],
            feature_dims: Dict[str, int],
            num_classes: int
    ):
        """
        Args:
            model: Multi-modal model (contains backbones)
            config: Algorithm configuration
            modalities: List of modality names
            feature_dims: Dict mapping modality to feature dimension
            num_classes: Number of output classes
        """
        super().__init__(model, config)

        self.modalities = modalities
        self.feature_dims = feature_dims
        self.num_classes = num_classes
        self.vit_pretrained_path = config.get('vit_pretrained_path', None)

        # Hyperparameters
        self.hidden_dim = config.get('hidden_dim', 2048)
        self.proj_dim = config.get('proj_dim', 128)
        self.alpha_contrast = config.get('alpha_contrast', 2.0)
        self.alpha_trans = config.get('alpha_trans', 1.0)
        self.temperature = config.get('temperature', 0.1)
        self.use_contras = config.get('use_contras', True)
        self.use_modtrans = config.get('use_modtrans', True)
        self.ld = config.get('ld', 0.05)  # Weight for modal gap losses

        # Optimizer configuration
        self.optimizer_type = config.get('optimizer_type', 'sam')
        self.use_scheduler = config.get('use_scheduler', True)
        self.sam_rho = config.get('sam_rho', 0.05)
        self.sam_adaptive = config.get('sam_adaptive', False)

        # ViT freezing configuration
        self.freeze_vit_backbone = config.get('freeze_vit_backbone', False)
        # freeze_vit_layers options:
        #   -1: freeze all layers (complete freeze)
        #   0-11: freeze layers 0 to N (ViT-Base has 12 layers)
        #   None/False: no freezing
        self.freeze_vit_layers = config.get('freeze_vit_layers', -1 if self.freeze_vit_backbone else None)

        # VideoMAEv2 optimizer configuration
        self.videomae_v2_pretrained_path = config.get('videomae_v2_pretrained_path', None)
        self.use_videomae_optim = config.get('use_videomae_optim', True)  # Use VideoMAEv2 optimizer strategy
        self.layer_decay = config.get('layer_decay', 0.75)  # Layer-wise LR decay
        self.warmup_epochs = config.get('warmup_epochs', 5)  # Warmup epochs
        self.min_lr = config.get('min_lr', 1e-6)  # Minimum learning rate
        self.warmup_lr = config.get('warmup_lr', 1e-8)  # Warmup start learning rate

        # GBL configuration
        self.use_gblend = config.get('use_gblend', False)
        self.gbl_momentum = config.get('gbl_momentum', 0.9)

        # Mixup configuration (per-modality for early fusion)
        self.use_mixup = config.get('use_mixup', False)
        self.mixup_alpha = config.get('mixup_alpha', 0.2)
        self.mixup_weight = config.get('mixup_weight', 0.1)
        self.mixup_start_epoch = config.get('mixup_start_epoch', 5)

        # Domain contrastive loss configuration (per-modality for DG)
        self.use_domain_contrastive = config.get('use_domain_contrastive', False)
        self.domain_contrastive_weight = config.get('domain_contrastive_weight', 0.05)
        self.domain_contrastive_temp = config.get('domain_contrastive_temp', 0.1)

        # MIRO loss configuration (per-modality for DG)
        self.use_miro = config.get('use_miro', False)
        self.miro_weight = config.get('miro_weight', 0.5)

        # MMD loss configuration (per-modality for DG)
        self.use_mmd = config.get('use_mmd', False)
        self.mmd_weight = config.get('mmd_weight', 0.05)

        # Initialize modality weights for GBL
        if self.use_gblend:
            num_weights = len(modalities) + 1  # per-modality + fusion
            equal_weight = 1.0 / num_weights

            self.modality_weights = {mod: equal_weight for mod in modalities}
            self.modality_weights['fusion'] = equal_weight

            # Loss history for GBL
            self.loss_history = {'train': {}, 'val': {}}

        # Fusion configuration
        self.fusion_type = config.get('fusion_type', 'bottleneck')
        self.fusion_hidden_dim = config.get('fusion_hidden_dim', 512)
        self.num_bottlenecks = config.get('num_bottlenecks', 4)

        # Automatically determine if projector is needed
        # If all modalities have the same feature dimension, skip projector
        unique_dims = set(feature_dims.values())
        self.use_projector = len(unique_dims) > 1

        if not self.use_projector:
            print(f"All modalities have same feature dim ({list(unique_dims)[0]}), skipping projectors")
            # Use feature dim as projection dim when projector is disabled
            self.proj_dim = list(unique_dims)[0]
        else:
            print(
                f"Different feature dimensions detected, using projectors: {feature_dims} -> proj_dim={self.proj_dim}")

        # Per-modality DG classifiers (operate on concatenated D0+D1 features)
        self.modality_dg_classifiers = nn.ModuleDict()
        for modality in modalities:
            self.modality_dg_classifiers[modality] = Encoder(
                input_dim=feature_dims[modality],
                out_dim=num_classes,
                hidden=512
            ).to(self.device)

        # Projection heads (project domain-invariant features to common space)
        # Only create if use_projector is True
        self.projectors = nn.ModuleDict()
        if self.use_projector:
            for modality in modalities:
                self.projectors[modality] = ProjectHead(
                    input_dim=feature_dims[modality],
                    hidden_dim=self.hidden_dim,
                    out_dim=self.proj_dim
                ).to(self.device)

        # MIRO encoders (if enabled)
        if self.use_miro:
            self.miro_mean_encoders = nn.ModuleDict()
            self.miro_var_encoders = nn.ModuleDict()
            # Each modality gets mean and variance encoders for projected features
            miro_shape = (1, self.proj_dim)  # Shape for projected features
            for modality in modalities:
                self.miro_mean_encoders[modality] = nn.ModuleList([
                    miro.MeanEncoder(miro_shape).to(self.device)
                ])
                self.miro_var_encoders[modality] = nn.ModuleList([
                    miro.VarianceEncoder(miro_shape).to(self.device)
                ])

        # Fusion module
        if self.fusion_type == 'bottleneck':
            input_dims = [feature_dims[mod] for mod in sorted(modalities)]
            self.fusion_module = BottleneckFusion(
                input_dims=input_dims,
                projected_dim=self.proj_dim,
                output_dim=num_classes,
                hidden_dim=self.fusion_hidden_dim,
                num_bottlenecks=self.num_bottlenecks
            ).to(self.device)
        elif self.fusion_type == 'transformer_bottleneck':
            input_dims = [feature_dims[mod] for mod in sorted(modalities)]
            self.fusion_module = TransformerBottleneckFusion(
                input_dims=input_dims,
                output_dim=num_classes,
                hidden_dim=self.fusion_hidden_dim,
                num_bottlenecks=self.num_bottlenecks,
                num_heads=config.get('fusion_num_heads', 8),
                num_layers=config.get('fusion_num_layers', 2),
                dropout=config.get('fusion_dropout', 0.1)
            ).to(self.device)

        elif self.fusion_type == 'attention':
            self.fusion_module = AttentionFusion(
                projected_dim=self.proj_dim,
                output_dim=num_classes,
                hidden_dim=self.fusion_hidden_dim
            ).to(self.device)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # Cross-modal translation
        self.cross_modal_trans = CrossModalTranslation(normalize=False)

        # Supervised contrastive loss
        if self.use_contras:
            self.contrastive_loss_fn = SupConLoss(temperature=self.temperature)

        # Classification loss
        self.criterion = nn.CrossEntropyLoss()

        # Setup optimizer
        self._setup_optimizer()

    def _freeze_vit_backbone(self):
        """
        Freeze ViT backbone layers based on configuration

        freeze_vit_layers options:
        - -1: Freeze all layers (complete freeze, only train classification heads)
        - 0-11: Freeze layers 0 to N (for ViT-Base with 12 transformer blocks)
        - None: No freezing
        """
        if not self.freeze_vit_backbone or self.freeze_vit_layers is None:
            print("[ViT Freezing] No freezing applied - all ViT parameters trainable")
            return

        print("\n" + "=" * 80)
        print("ViT BACKBONE FREEZING")
        print("=" * 80)

        # Helper function to freeze a ViT model
        def freeze_vit_model(vit_model, modality_name):
            """Freeze specific layers of a ViT model"""
            total_params = 0
            frozen_params = 0

            if self.freeze_vit_layers == -1:
                # Complete freeze: freeze everything except head
                print(f"[{modality_name}] Freezing ALL backbone layers (complete freeze)")

                for name, param in vit_model.named_parameters():
                    total_params += param.numel()
                    # Only keep head trainable
                    if not name.startswith('head'):
                        param.requires_grad = False
                        frozen_params += param.numel()
                    else:
                        param.requires_grad = True

            else:
                # Partial freeze: freeze layers 0 to freeze_vit_layers
                freeze_to_layer = self.freeze_vit_layers
                print(
                    f"[{modality_name}] Freezing layers 0-{freeze_to_layer} (keeping {11 - freeze_to_layer} layers + head trainable)")

                for name, param in vit_model.named_parameters():
                    total_params += param.numel()

                    # Freeze patch_embed and positional embeddings
                    if 'patch_embed' in name or 'pos_embed' in name or 'cls_token' in name:
                        param.requires_grad = False
                        frozen_params += param.numel()
                    # Freeze transformer blocks up to freeze_to_layer
                    elif 'blocks' in name:
                        # Extract block index (e.g., blocks.0, blocks.1, ...)
                        block_idx = int(name.split('blocks.')[1].split('.')[0])
                        if block_idx <= freeze_to_layer:
                            param.requires_grad = False
                            frozen_params += param.numel()
                        else:
                            param.requires_grad = True
                    # Keep head and norm trainable
                    else:
                        param.requires_grad = True

            frozen_pct = 100.0 * frozen_params / total_params if total_params > 0 else 0
            trainable_params = total_params - frozen_params
            print(f"  Total params: {total_params:,}")
            print(f"  Frozen params: {frozen_params:,} ({frozen_pct:.1f}%)")
            print(f"  Trainable params: {trainable_params:,} ({100 - frozen_pct:.1f}%)")

        # Freeze RGB ViT
        if hasattr(self.model, 'rgb_backbone'):
            freeze_vit_model(self.model.rgb_backbone, 'RGB ViT')

        # Freeze Flow ViT
        if hasattr(self.model, 'flow_backbone'):
            freeze_vit_model(self.model.flow_backbone, 'Flow ViT')

        # Freeze Audio ViT (might be wrapped in ASTWrapper)
        if hasattr(self.model, 'audio_backbone'):
            audio_model = self.model.audio_backbone
            # Check if it's wrapped
            if hasattr(audio_model, 'ast'):
                # AST wrapper
                freeze_vit_model(audio_model.ast, 'Audio AST')
            else:
                # Standard ViT
                freeze_vit_model(audio_model, 'Audio ViT')

        print("=" * 80 + "\n")

    def _get_videomae_parameter_groups(self, model, weight_decay, layer_decay):
        """
        Create parameter groups with layer-wise LR decay and weight decay skipping
        (VideoMAEv2 style)

        Returns:
            parameter_groups: List of parameter dicts with 'params', 'weight_decay', 'lr_scale'
            skip_list: Set of parameter names to skip weight decay
        """
        # Get skip list from model (pos_embed, cls_token, etc.)
        skip_list = set()
        for backbone_name in ['rgb_backbone', 'flow_backbone', 'audio_backbone']:
            if hasattr(model, backbone_name):
                backbone = getattr(model, backbone_name)
                if hasattr(backbone, 'no_weight_decay'):
                    skip_list.update(backbone.no_weight_decay())

        # Determine number of layers (VideoMAEv2 ViT-Base has 12 transformer blocks)
        num_layers = 12  # ViT-Base default
        for backbone_name in ['rgb_backbone', 'flow_backbone']:
            if hasattr(model, backbone_name):
                backbone = getattr(model, backbone_name)
                if hasattr(backbone, 'get_num_layers'):
                    num_layers = backbone.get_num_layers()
                    break

        # Create layer-wise LR decay assigner
        if layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(
                list(layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            )
            print(f"[Optimizer] Using layer-wise LR decay (layer_decay={layer_decay}, num_layers={num_layers})")
            print(f"  Layer scales: {[f'{v:.4f}' for v in assigner.values]}")
        else:
            assigner = None
            print(f"[Optimizer] Layer-wise LR decay disabled (layer_decay={layer_decay})")

        # Create parameter groups
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters

            # Determine if weight decay should be applied
            # Skip weight decay for: 1D params (norms), biases, and special parameters (pos_embed, cls_token)
            if len(param.shape) == 1 or name.endswith(".bias") or name.endswith(".scale") or name in skip_list:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay

            # Get layer ID for layer-wise LR decay
            if assigner is not None:
                layer_id = assigner.get_layer_id(name)
                group_name = f"layer_{layer_id}_{group_name}"
            else:
                layer_id = None

            # Create group if not exists
            if group_name not in parameter_group_names:
                if assigner is not None:
                    scale = assigner.get_scale(layer_id)
                else:
                    scale = 1.

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        print(f"[Optimizer] Created {len(parameter_group_vars)} parameter groups")
        return list(parameter_group_vars.values()), skip_list

    def _setup_optimizer(self):
        """Setup optimizer with VideoMAEv2-style layer-wise LR decay"""
        # Get hyperparameters - Use VideoMAEv2 defaults (lr=1e-3, wd=0.05)
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.05)

        # Freeze ViT backbones if configured
        is_vit_based = self.vit_pretrained_path or self.videomae_v2_pretrained_path
        if is_vit_based:
            self._freeze_vit_backbone()

        # Use VideoMAEv2 optimizer strategy if enabled
       
        if self.use_videomae_optim and is_vit_based:
            print("=" * 80)
            print("[Optimizer] Using VideoMAEv2 optimization strategy")
            print(f"  - Layer-wise LR decay: {self.layer_decay}")
            print(f"  - Weight decay skipping: Enabled (pos_embed, cls_token, biases, norms)")
            print(f"  - Base LR: {lr}")
            print(f"  - Weight decay: {weight_decay}")
            print("=" * 80 + "\n")

            # Create all modules in a single model wrapper for unified parameter grouping
            param_groups, skip_list = self._get_videomae_parameter_groups(
                self.model, weight_decay, self.layer_decay
            )

            # Add other module parameters (per-modality DG classifiers, fusion, projectors, MIRO)
            other_modules = nn.ModuleDict({
                'fusion': self.fusion_module
            })
            # Per-modality DG classifiers
            for mod_name, classifier in self.modality_dg_classifiers.items():
                other_modules[f'dg_classifier_{mod_name}'] = classifier
            # Projectors
            if self.use_projector:
                for mod_name, projector in self.projectors.items():
                    other_modules[f'projector_{mod_name}'] = projector
            # MIRO encoders
            if self.use_miro:
                for mod_name, mean_enc in self.miro_mean_encoders.items():
                    other_modules[f'miro_mean_{mod_name}'] = mean_enc
                for mod_name, var_enc in self.miro_var_encoders.items():
                    other_modules[f'miro_var_{mod_name}'] = var_enc

            # Add other module parameters to param groups (full lr, with weight decay)
            other_param_group = {
                'params': [],
                'weight_decay': weight_decay,
                'lr_scale': 1.0  # Full learning rate for head/fusion modules
            }
            for module in other_modules.values():
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        # Skip weight decay for biases and norms
                        if len(param.shape) == 1 or name.endswith(".bias"):
                            wd = 0.
                        else:
                            wd = weight_decay

                        # Find or create group for other params
                        if len(other_param_group['params']) == 0:
                            other_param_group['weight_decay'] = wd
                        other_param_group['params'].append(param)

            if len(other_param_group['params']) > 0:
                param_groups.append(other_param_group)
                print(f"[Optimizer] Added {len(other_param_group['params'])} parameters from other modules")

        else:
            # Legacy optimizer strategy (backward compatibility)
            print("=" * 80)
            print("[Optimizer] Using legacy optimization strategy")
            print("=" * 80 + "\n")

            backbone_lr_scale = self.config.get('backbone_lr_scale', 0.1)
            backbone_params = []
            other_params = []

            if is_vit_based:
                trainable_model_params = [p for p in self.model.parameters() if p.requires_grad]
                backbone_params.extend(trainable_model_params)
                model_type = "VideoMAEv2" if self.videomae_v2_pretrained_path else "ViT"
                print(
                    f"[Optimizer] Collected {len(trainable_model_params)} trainable parameter groups from {model_type} backbones")
            else:
                # Action Recognition model structure
                if 'rgb' in self.modalities and hasattr(self.model, 'rgb_backbone'):
                    rgb_trainable = [p for p in self.model.rgb_backbone.parameters() if p.requires_grad]
                    backbone_params.extend(rgb_trainable)
                if 'flow' in self.modalities and hasattr(self.model, 'flow_backbone'):
                    flow_trainable = [p for p in self.model.flow_backbone.parameters() if p.requires_grad]
                    backbone_params.extend(flow_trainable)
                if 'audio' in self.modalities and hasattr(self.model, 'audio_cls'):
                    audio_trainable = [p for p in self.model.audio_cls.parameters() if p.requires_grad]
                    backbone_params.extend(audio_trainable)

            # Collect other module parameters
            # Per-modality DG classifiers
            for classifier in self.modality_dg_classifiers.values():
                other_params.extend(list(classifier.parameters()))
            # Projectors
            if self.use_projector:
                for projector in self.projectors.values():
                    other_params.extend(list(projector.parameters()))
            # Fusion module
            other_params.extend(list(self.fusion_module.parameters()))
            # MIRO encoders
            if self.use_miro:
                for mean_enc_list in self.miro_mean_encoders.values():
                    other_params.extend(list(mean_enc_list.parameters()))
                for var_enc_list in self.miro_var_encoders.values():
                    other_params.extend(list(var_enc_list.parameters()))

            # Create parameter groups
            if len(backbone_params) > 0:
                param_groups = [
                    {'params': other_params, 'lr': lr, 'lr_scale': 1.0},
                    {'params': backbone_params, 'lr': lr * backbone_lr_scale, 'lr_scale': backbone_lr_scale}
                ]
            else:
                param_groups = [{'params': other_params, 'lr': lr, 'lr_scale': 1.0}]

        # Create optimizer
        if self.optimizer_type == 'sam':
            self.optimizer = SAM(
                param_groups,
                base_optimizer=torch.optim.AdamW,
                lr=lr,
                weight_decay=weight_decay,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive
            )
        elif self.optimizer_type in ('adamw', 'adam'):
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=0.,  # weight_decay handled in param_groups
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            self.optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=0.9,
                weight_decay=0.  # weight_decay handled in param_groups
            )

        # Setup step-level scheduler (VideoMAEv2 style)
        self.scheduler = None
        self.lr_schedule_values = None
        self.wd_schedule_values = None

        if self.use_scheduler and self.use_videomae_optim:
            # Will be initialized in train() when num_training_steps_per_epoch is known
            print("[Optimizer] Step-level scheduler will be initialized during training")
        elif self.use_scheduler:
            # Legacy epoch-level scheduler
            from torch.optim.lr_scheduler import CosineAnnealingLR
            base_opt = self.optimizer.base_optimizer if self.optimizer_type == 'sam' else self.optimizer
            num_epochs = self.config.get('num_epochs', 15)
            self.scheduler = CosineAnnealingLR(base_opt, T_max=num_epochs, eta_min=self.min_lr)

        # Print optimizer summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n[Optimizer] Summary:")
        print(f"  Optimizer type: {self.optimizer_type}")
        print(f"  Base learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
        print(
            f"  Frozen parameters: {total_params - trainable_params:,} ({100 * (1 - trainable_params / total_params):.1f}%)")
        print(f"  Use VideoMAEv2 optim: {self.use_videomae_optim}")
        print(f"  Use scheduler: {self.use_scheduler}")
        if self.freeze_vit_backbone:
            model_type = "VideoMAEv2" if self.videomae_v2_pretrained_path else "ViT"
            print(f"  {model_type} freezing: Enabled (freeze_vit_layers={self.freeze_vit_layers})")
        print()

    def extract_features(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features from all modalities

        Args:
            modality_inputs: Dict of modality inputs

        Returns:
            modality_features: Dict of raw features per modality
        """
        modality_features = {}

        # Extract features using model's per-modality forward
        for modality in self.modalities:
            if modality in modality_inputs:
                # Use model's forward_modality to get features
                _, features = self.model.forward_modality(
                    modality, modality_inputs[modality]
                )
                modality_features[modality] = features

        return modality_features

    def extract_pretrained_features(
            self,
            modality_inputs0: Dict[str, torch.Tensor],
            modality_inputs1: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Extract pretrained features for MIRO (frozen backbone features)

        Args:
            modality_inputs0: Modality inputs from domain 0
            modality_inputs1: Modality inputs from domain 1

        Returns:
            pretrained_features: Dict of concatenated pretrained projected features per modality
        """
        pretrained_features = {}

        with torch.no_grad():
            # Extract features from both domains
            modality_features0 = self.extract_features(modality_inputs0)
            modality_features1 = self.extract_features(modality_inputs1)
            # Project and concatenate for each modality
            for modality in self.modalities:
                if modality in modality_features0 and modality in modality_features1:
                    if self.use_projector:
                        # Project features using frozen projector
                        proj_feat0 = self.projectors[modality](modality_features0[modality])
                        proj_feat1 = self.projectors[modality](modality_features1[modality])
                    else:
                        # Use features directly (with normalization) when projector is disabled
                        proj_feat0 = F.normalize(modality_features0[modality], dim=1)
                        proj_feat1 = F.normalize(modality_features1[modality], dim=1)
                    # Concatenate from both domains
                    pretrained_features[modality] = torch.cat([proj_feat0, proj_feat1], dim=0)

        return pretrained_features

    def compute_per_modality_dg_losses(
            self,
            modality_features0: Dict[str, torch.Tensor],
            modality_features1: Dict[str, torch.Tensor],
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute per-modality DG classification losses

        Args:
            modality_features0: Features from domain 0
            modality_features1: Features from domain 1
            labels: Combined labels (D0 + D1)

        Returns:
            total_dg_loss: Sum of per-modality DG losses
            loss_dict: Dict of individual losses (as tensors for GBL)
            modality_features_combined: Dict of concatenated features (D0+D1) per modality
        """
        total_loss = 0
        loss_dict = {}
        modality_features_combined = {}

        for modality in self.modalities:
            if modality in modality_features0 and modality in modality_features1:
                # Concatenate features from both domains
                features_combined = torch.cat([
                    modality_features0[modality],
                    modality_features1[modality]
                ], dim=0)

                # Store combined features for reuse
                modality_features_combined[modality] = features_combined

                # Per-modality DG classifier
                logits = self.modality_dg_classifiers[modality](features_combined)

                loss = self.criterion(logits, labels.long())

                total_loss += loss
                # Store as tensor (not .item()) for GBL weighted combination
                loss_dict[f'{modality}_dg_loss'] = loss

        return total_loss, loss_dict, modality_features_combined

    def compute_modal_gap_losses(
            self,
            modality_features0: Dict[str, torch.Tensor],
            modality_features1: Dict[str, torch.Tensor],
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute modal gap losses (CMT + SupCon) on projected domain-invariant features

        Args:
            modality_features0:  features from D0
            modality_features1:  features from D1
            labels: Combined labels (D0 + D1)

        Returns:
            modal_gap_loss: Combined CMT + SupCon loss
            loss_dict: Dict of individual losses
        """
        modality_projected0 = {}
        modality_projected1 = {}

        if self.use_projector:
            # Use projectors to map features to common space
            for modality in self.modalities:
                if modality in modality_features0:
                    modality_projected0[modality] = self.projectors[modality](modality_features0[modality])
                if modality in modality_features1:
                    modality_projected1[modality] = self.projectors[modality](modality_features1[modality])
        else:
            # Skip projection, use features directly (when all dims are same)
            modality_projected0 = {k: F.normalize(v, dim=1) for k, v in modality_features0.items()}
            modality_projected1 = {k: F.normalize(v, dim=1) for k, v in modality_features1.items()}

        # Combine projections from both domains
        modality_projected = {}
        for modality in self.modalities:
            if modality in modality_projected0 and modality in modality_projected1:
                modality_projected[modality] = torch.cat([
                    modality_projected0[modality],
                    modality_projected1[modality]
                ], dim=0)

        # Cross-modal translation loss
        translation_loss = 0
        if self.use_modtrans and len(modality_projected) >= 2:
            translation_loss = self.cross_modal_trans(modality_projected)

        # Supervised contrastive loss
        contrastive_loss = 0
        if self.use_contras and len(modality_projected) >= 2:
            # Stack modalities as different views
            modality_list = sorted(modality_projected.keys())
            features = torch.stack([modality_projected[mod] for mod in modality_list], dim=1)
            contrastive_loss = self.contrastive_loss_fn(features, labels)

        # Total modal gap loss
        modal_gap_loss = self.alpha_trans * translation_loss + self.alpha_contrast * contrastive_loss

        loss_dict = {
            'translation_loss': translation_loss if isinstance(translation_loss,
                                                               (int, float)) else translation_loss.item(),
            'contrastive_loss': contrastive_loss if isinstance(contrastive_loss,
                                                               (int, float)) else contrastive_loss.item(),
        }

        return modal_gap_loss, loss_dict, modality_projected0, modality_projected1, modality_projected

    def compute_fusion_loss(
            self,
            modality_features_combined: Dict[str, torch.Tensor],
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fusion classification loss on domain-invariant features

        Args:
            modality_features_combined: Concatenated features (D0+D1) per modality
            labels: Combined labels (D0 + D1)

        Returns:
            fusion_loss: Classification loss on fused features
            fusion_predictions: Predictions from fusion
        """
        # Fusion forward (sorted order)
        feature_list = [modality_features_combined[mod] for mod in sorted(modality_features_combined.keys())]
        fusion_predictions, _ = self.fusion_module(feature_list)

        # Classification loss
        fusion_loss = self.criterion(fusion_predictions, labels.long())

        return fusion_loss, fusion_predictions

    def _compute_all_losses(
            self,
            modality_inputs0: Dict[str, torch.Tensor],
            labels0: torch.Tensor,
            modality_inputs1: Dict[str, torch.Tensor],
            labels1: torch.Tensor,
            current_epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """
        Compute all losses for two domains

        Process (domain-then-modal):
        1. Extract features from D0 and D1
        2. Per-modality DG losses (domain gap)
        3. Modal gap losses (CMT + SupCon)
        4. Fusion loss

        Args:
            modality_inputs0: Modality inputs from domain 0
            labels0: Labels from domain 0
            modality_inputs1: Modality inputs from domain 1
            labels1: Labels from domain 1

        Returns:
            total_loss: Combined loss
            loss_dict: Dict of individual losses
            predictions: Final predictions
        """
        # 1. Extract features from both domains
        modality_features0 = self.extract_features(modality_inputs0)
        modality_features1 = self.extract_features(modality_inputs1)

        # Combine labels
        labels_combined = torch.cat([labels0, labels1], dim=0)

        # 2. Per-modality domain gap mitigation
        # Returns combined features to avoid recomputation
        dg_loss, dg_loss_dict, modality_features_combined = self.compute_per_modality_dg_losses(
            modality_features0, modality_features1, labels_combined
        )
        # 3. Modal gap mitigation (on projected domain-invariant features)
        # Pass pre-computed projected features to avoid redundant computation

        modal_gap_loss, _, modality_projected0, modality_projected1, modality_projected = self.compute_modal_gap_losses(
            modality_features0, modality_features1, labels_combined
        )

        # 4. Fusion loss
        # Reuses precomputed combined features
        fusion_loss, fusion_predictions = self.compute_fusion_loss(
            modality_features_combined, labels_combined
        )

        # 2.1. MMD loss (Maximum Mean Discrepancy) - Per-modality domain alignment
        # we can achive this ether on projected features or on in variant feature space
        mmd_loss = 0
        if self.use_mmd:
            # Compute MMD loss for each modality's projected features between D0 and D1
            # This aligns domain distributions in the projected embedding space
            for modality in self.modalities:
                if modality in modality_features0 and modality in modality_features1:
                    modality_mmd_loss = compute_mmd_loss(
                        modality_features0[modality],
                        modality_features1[modality]
                    )

                    # Apply GBL weights if enabled
                    if self.use_gblend:
                        weight = self.modality_weights.get(modality, 0)
                        mmd_loss += weight * modality_mmd_loss
                    else:
                        mmd_loss += modality_mmd_loss

        # 2.2. Per-modality domain contrastive loss (if enabled)
        # we can also try modality_projected instead of modality_features_combined
        domain_contrastive_loss = 0

        if self.use_domain_contrastive:
            per_modality_contrastive_losses = {}
            # Compute per-modality contrastive losses on domain-invariant features
            for modality in self.modalities:
                if modality in modality_features_combined:
                    _, modality_contrastive_loss = compute_modal_contrastive_losses(
                        modality_features_combined[modality],
                        labels_combined,
                        temperature=self.domain_contrastive_temp
                    )
                    per_modality_contrastive_losses[modality] = modality_contrastive_loss

            # Aggregate using GBL weights if enabled
            if self.use_gblend and len(per_modality_contrastive_losses) > 0:
                # Weighted sum using GBL modality weights
                for modality, contrastive_loss in per_modality_contrastive_losses.items():
                    weight = self.modality_weights.get(modality, 0)
                    domain_contrastive_loss += weight * contrastive_loss
            elif len(per_modality_contrastive_losses) > 0:
                # Simple average if GBL is not enabled
                domain_contrastive_loss = sum(per_modality_contrastive_losses.values()) / len(
                    per_modality_contrastive_losses)

        # 2.3. Per-modality Mixup (if enabled and after start epoch)
        mixup_loss = 0
        if self.use_mixup and current_epoch >= self.mixup_start_epoch:
            per_modality_mixup_losses = {}
            # Apply mixup to per-modality features (fusion_feat is None for early fusion)

            mixed_modality_features, _, mixed_labels, _ = multimodal_mixup(
                modality_features0, None, labels0.long(),
                modality_features1, None, labels1.long(), num_classes=self.num_classes,
                alpha=self.mixup_alpha
            )

            # Compute per-modality DG predictions on mixed features
            for modality in self.modalities:
                if modality in mixed_modality_features:
                    mixed_feat = mixed_modality_features[modality]
                    mixed_predictions = self.modality_dg_classifiers[modality](mixed_feat)

                    # Compute mixup loss with soft labels
                    modality_mixup_loss = mixup_criterion(mixed_predictions, mixed_labels, self.criterion)
                    per_modality_mixup_losses[modality] = modality_mixup_loss

            # Aggregate mixup losses using GBL weights if enabled
            if self.use_gblend and len(per_modality_mixup_losses) > 0:
                # Weighted sum using GBL modality weights
                for modality, modality_mixup_loss in per_modality_mixup_losses.items():
                    weight = self.modality_weights.get(modality, 0)
                    mixup_loss += weight * modality_mixup_loss
            elif len(per_modality_mixup_losses) > 0:
                # Simple average if GBL is not enabled
                mixup_loss = sum(per_modality_mixup_losses.values()) / len(per_modality_mixup_losses)

        # 2.5 MIRO loss (if enabled)
        miro_loss = 0

        if self.use_miro:
            per_modality_miro_losses = {}
            # Extract pretrained features (frozen)
            pretrained_features = self.extract_pretrained_features(modality_inputs0, modality_inputs1)
            # Compute MIRO loss for each modality
            for modality in self.modalities:
                if modality in modality_features_combined:
                    # Project current trainable features
                    # current_proj_feat0 = self.projectors[modality](modality_features0[modality])
                    # current_proj_feat1 = self.projectors[modality](modality_features1[modality])
                    # current_proj_feat = torch.cat([current_proj_feat0, current_proj_feat1], dim=0)

                    # Compute MIRO loss using mean and variance encoders
                    pretrained_feat = pretrained_features[modality]
                    current_proj_feat = modality_projected[modality]
                    for f, pre_f, mean_enc, var_enc in miro.zip_strict(
                            [current_proj_feat], [pretrained_feat],
                            self.miro_mean_encoders[modality], self.miro_var_encoders[modality]
                    ):
                        mean = mean_enc(f)
                        var = var_enc(f)
                        # Variational lower bound
                        vlb = (mean - pre_f).pow(2).div(var) + (var + 1).log()
                        modality_miro_loss = vlb.mean() / 2.0
                        per_modality_miro_losses[modality] = modality_miro_loss

            # Aggregate MIRO losses using GBL weights if enabled
            if self.use_gblend and len(per_modality_miro_losses) > 0:
                # Weighted sum using GBL modality weights
                for modality, modality_miro_loss in per_modality_miro_losses.items():
                    weight = self.modality_weights.get(modality, 0)
                    miro_loss += weight * modality_miro_loss
            elif len(per_modality_miro_losses) > 0:
                # Simple average if GBL is not enabled
                miro_loss = sum(per_modality_miro_losses.values()) / len(per_modality_miro_losses)

        # Total loss
        if self.use_gblend:
            # Apply GBL weights (directly use tensors from loss_dict)
            weighted_loss = 0
            for modality in self.modalities:
                modality_loss_key = f'{modality}_dg_loss'
                if modality_loss_key in dg_loss_dict:
                    weight = self.modality_weights.get(modality, 0)
                    modality_loss_tensor = dg_loss_dict[modality_loss_key]
                    weighted_loss += weight * modality_loss_tensor

            # Add weighted fusion loss
            weighted_loss += self.modality_weights['fusion'] * fusion_loss

            # Add modal gap loss, mixup loss, domain contrastive loss, MIRO loss, and MMD loss
            total_loss = weighted_loss + self.ld * modal_gap_loss + self.mixup_weight * mixup_loss + self.domain_contrastive_weight * domain_contrastive_loss + self.miro_weight * miro_loss + self.mmd_weight * mmd_loss
        else:
            # Uniform weighting
            total_loss = dg_loss + fusion_loss + self.ld * modal_gap_loss + self.mixup_weight * mixup_loss + self.domain_contrastive_weight * domain_contrastive_loss + self.miro_weight * miro_loss + self.mmd_weight * mmd_loss

        # Prepare loss dict
        loss_dict = {
            'dg_loss': dg_loss.item(),
            'fusion_loss': fusion_loss.item(),
            'modal_gap_loss': modal_gap_loss.item(),
            'mixup_loss': mixup_loss.item() if isinstance(mixup_loss, torch.Tensor) else mixup_loss,
            'domain_contrastive_loss': domain_contrastive_loss.item() if isinstance(domain_contrastive_loss,
                                                                                    torch.Tensor) else domain_contrastive_loss,
            'miro_loss': miro_loss.item() if isinstance(miro_loss, torch.Tensor) else miro_loss,
            'mmd_loss': mmd_loss.item() if isinstance(mmd_loss, torch.Tensor) else mmd_loss,
            'total_loss': total_loss.item()
        }
        loss_dict.update(dg_loss_dict)
        # loss_dict.update(modal_gap_dict)

        # Add per-modality losses for GBL tracking
        if self.use_gblend:
            loss_dict['modality_losses'] = {}
            for modality in self.modalities:
                loss_key = f'{modality}_dg_loss'
                if loss_key in dg_loss_dict:
                    tensor_loss = dg_loss_dict[loss_key]
                    loss_dict['modality_losses'][modality] = tensor_loss.item() if isinstance(tensor_loss,
                                                                                              torch.Tensor) else tensor_loss
                loss_dict['modality_losses']['fusion'] = fusion_loss.item()

        return total_loss, loss_dict, fusion_predictions

    def train_step_two_domains(
            self,
            modality_inputs0: Dict[str, torch.Tensor],
            labels0: torch.Tensor,
            modality_inputs1: Dict[str, torch.Tensor],
            labels1: torch.Tensor,
            current_epoch: int = 0
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        Training step for two domains

        Args:
            modality_inputs0: Modality inputs from domain 0
            labels0: Labels from domain 0
            modality_inputs1: Modality inputs from domain 1
            labels1: Labels from domain 1

        Returns:
            loss_dict: Dict of losses
            predictions: Final predictions
        """
        # Set to training mode
        self.model.train()
        for classifier in self.modality_dg_classifiers.values():
            classifier.train()
        for projector in self.projectors.values():
            projector.train()
        self.fusion_module.train()

        if self.optimizer_type == 'sam':
            # SAM two-step update
            # First step
            total_loss, loss_dict, predictions = self._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1, current_epoch
            )
            total_loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # Second step
            total_loss_new, _, _ = self._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1, current_epoch
            )
            total_loss_new.backward()
            self.optimizer.second_step(zero_grad=True)

            if self.scheduler is not None:
                self.scheduler.step()

        else:
            # Standard optimizer
            total_loss, loss_dict, predictions = self._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1, current_epoch
            )
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

        return loss_dict, predictions

    def predict(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Make predictions

        Args:
            modality_inputs: Dict of modality inputs

        Returns:
            Predictions from fusion module
        """
        self.model.eval()
        for classifier in self.modality_dg_classifiers.values():
            classifier.eval()
        if self.use_projector:
            for projector in self.projectors.values():
                projector.eval()
        self.fusion_module.eval()

        with torch.no_grad():
            # Extract features
            modality_features = self.extract_features(modality_inputs)

            # Fusion
            feature_list = [modality_features[mod] for mod in sorted(modality_features.keys())]
            predictions, _ = self.fusion_module(feature_list)

        return predictions

    def update_loss_history(self, epoch: int, split: str, losses: Dict[str, float]):
        """
        Update loss history for GBL

        Args:
            epoch: Current epoch
            split: 'train' or 'val'
            losses: Dict of modality losses
        """
        if not self.use_gblend:
            return

        if epoch not in self.loss_history[split]:
            self.loss_history[split][epoch] = {}

        self.loss_history[split][epoch] = losses.copy()

    def update_gbl_weights(self, epoch: int) -> Optional[Dict[str, float]]:
        """
        Update GBL weights based on loss history

        Args:
            epoch: Current epoch (must be > 0)

        Returns:
            Updated weights dict
        """
        if not self.use_gblend or epoch == 0:
            return None

        modality_list = list(self.modalities) + ['fusion']

        # Compute GBL coefficients
        modal_weights = []
        for modality in modality_list:
            prev_train = self.loss_history['train'].get(epoch - 1, {}).get(modality, 0)
            prev_val = self.loss_history['val'].get(epoch - 1, {}).get(modality, 0)
            curr_train = self.loss_history['train'].get(epoch, {}).get(modality, 0)
            curr_val = self.loss_history['val'].get(epoch, {}).get(modality, 0)

            coef = GBL.compute_gblend_coef(curr_train, curr_val, prev_train, prev_val)
            modal_weights.append(coef)

        # Normalize
        normed_weights = GBL.normalize_weights(modal_weights)

        # Apply exponential smoothing
        for idx, modality in enumerate(modality_list):
            old_weight = self.modality_weights[modality]
            new_weight = old_weight * self.gbl_momentum + normed_weights[idx] * (1 - self.gbl_momentum)
            self.modality_weights[modality] = new_weight

        return self.modality_weights.copy()

    def update(self, loss: torch.Tensor):
        """Update model parameters (for compatibility with base class)"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()