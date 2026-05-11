"""
Late Fusion Domain Generalization Algorithm for FAS and ACtion recognition with CNN backbone
 

Key components:
1. Per-modality classification with independent backbones
2. Feature projection to common space
3. Attention-based fusion of projected features
4. Cross-modal translation for robustness
5. Supervised contrastive learning
6. Domain generalization classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List

from .base import BaseAlgorithm
from .MMmethods.modal_contrastive_loss import SupConLoss
from .algorithm_utils.projectors import ProjectHead
from .MMmethods.CMT import CrossModalTranslation
from .MMmethods.mmfusion import AttentionFusion, BottleneckFusion, create_fusion_model 
from .algorithm_utils.sam import SAM
from .MMmethods import GBL
from .DG_methods.mixup import multimodal_mixup, mixup_criterion
from .DG_methods.domain_contrastive_loss import compute_modal_contrastive_losses
from .DG_methods import miro
from .DG_methods.mmd import compute_mmd_loss

class LateFusionDG(BaseAlgorithm):
    """
    Late Fusion Domain Generalization Algorithm

    Architecture:
    1. Extract features from each modality independently
    2. Per-modality classification
    3. Project features to common embedding space
    4. Fuse projected features with attention
    5. Final classification on fused features

    Losses:
    - Per-modality classification losses
    - Fused classification loss
    - Cross-modal translation loss
    - Supervised contrastive loss
    - Domain generalization loss
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

        # Hyperparameters
        self.hidden_dim = config.get('hidden_dim', 2048)
        self.proj_dim = config.get('proj_dim', 128)
        self.alpha_contrast = config.get('alpha_contrast',3)
        self.alpha_trans = config.get('alpha_trans', 1)
        self.temperature = config.get('temperature', 0.1)
        self.use_contras = config.get('use_contras', True)
        self.use_modtrans = config.get('use_modtrans', True)
        self.ld = config.get('ld', 0.05)

        # Domain contrastive loss configuration
        self.use_domain_contrastive = config.get('use_domain_contrastive', False)
        self.dg_contrastive_weight = config.get('dg_contrastive_weight', 0.05)
        self.dg_contrastive_temp = config.get('dg_contrastive_temp', 0.1)

        # MMD loss configuration
        self.use_mmd = config.get('use_mmd', False)
        self.mmd_weight = config.get('mmd_weight', 0.05)
        self.mmd_kernel_num = config.get('mmd_kernel_num', 5)
        self.mmd_kernel_mul = config.get('mmd_kernel_mul', 2.0)

        # Mixup configuration
        self.use_mixup = config.get('use_mixup', False)
        self.mixup_alpha = config.get('mixup_alpha', 1.0)
        self.mixup_weight = config.get('mixup_weight', 0.5)

        # MIRO configuration
        self.use_miro = config.get('use_miro', False)
        self.miro_weight = config.get('miro_weight', 1.0)
        self.miro_var_init = config.get('miro_var_init', 0.1)

        # Optimizer configuration
        self.optimizer_type = config.get('optimizer_type', 'sam')  # 'sgd' or 'sam'
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

        # GBL (Generalized Blending) configuration
        self.use_gblend = config.get('use_gblend', False)
        self.gbl_momentum = config.get('gbl_momentum', 0.9)

        # Initialize modality weights for each domain (for GBL)
        # Weights for per-modality losses + fusion loss
        if self.use_gblend:
            num_weights_per_domain = len(modalities) + 1  # per-modality + fusion
            # equal_weight = 1.0 / num_weights_per_domain
            equal_weight = 0.2

            self.domain0_weights = {mod: equal_weight for mod in modalities}
            # self.domain0_weights['fusion'] = equal_weight
            self.domain0_weights['fusion'] = 0.4

            self.domain1_weights = {mod: equal_weight for mod in modalities}
            # self.domain1_weights['fusion'] = equal_weight
            self.domain1_weights['fusion'] = 0.4

            # Loss history for GBL weight computation
            self.loss_history = {
                'D0': {'train': {}, 'val': {}},
                'D1': {'train': {}, 'val': {}}
            }

        # Fusion configuration
        self.fusion_type = config.get('fusion_type', 'bottleneck')  # Default: 'bottleneck', also supports 'attention'
        self.fusion_hidden_dim = config.get('fusion_hidden_dim', 512)
        self.num_bottlenecks = config.get('num_bottlenecks', 4)  # For bottleneck fusion
        self.dg_hidden_dim = config.get('dg_hidden_dim', 256)  # For DG classifier
        self.vit_pretrained_path = config.get('vit_pretrained_path', None)
        #

        # MIRO encoders (if enabled)
        if self.use_miro:
            # Shape for fused features from fusion module
            miro_shape = (1, self.fusion_hidden_dim)
            self.miro_mean_encoder = miro.MeanEncoder(miro_shape).to(self.device)
            self.miro_var_encoder = miro.VarianceEncoder(
                miro_shape,
                init=self.miro_var_init
            ).to(self.device)

        # Create projection heads for each modality
        unique_dims = set(feature_dims.values())
        self.use_projector = len(unique_dims) > 1

        if not self.use_projector:
            print(f"All modalities have same feature dim ({list(unique_dims)[0]}), skipping projectors")
            # Use feature dim as projection dim when projector is disabled
            self.proj_dim = list(unique_dims)[0]
        else:
            print(
                f"Different feature dimensions detected, using projectors: {feature_dims} -> proj_dim={self.proj_dim}")

        # Create projection heads for each modality (only if needed)
        self.projectors = nn.ModuleDict()
        if self.use_projector:
            for modality in modalities:
                self.projectors[modality] = ProjectHead(
                    input_dim=feature_dims[modality],
                    hidden_dim=self.hidden_dim,
                    out_dim=self.proj_dim
                ).to(self.device)

        # Fusion module (configurable: bottleneck or attention fusion)
        if self.fusion_type == 'bottleneck':
            # Prepare input dimensions list for each modality (use sorted order)
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
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}. "
                             f"Supported types: 'bottleneck', 'attention'")

        # Domain generalization classifier (two-layer MLP)
        self.dg_classifier = nn.Sequential(
            nn.Linear(self.fusion_hidden_dim, self.dg_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.dg_hidden_dim, num_classes)
        ).to(self.device)

        # Cross-modal translation (on projected features)
        self.cross_modal_trans = CrossModalTranslation(
            normalize=False
        )

        # Supervised contrastive loss
        if self.use_contras:
            self.contrastive_loss_fn = SupConLoss(
                temperature=self.temperature
            )

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

        print("\n" + "="*80)
        print("ViT BACKBONE FREEZING")
        print("="*80)

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
                print(f"[{modality_name}] Freezing layers 0-{freeze_to_layer} (keeping {11 - freeze_to_layer} layers + head trainable)")

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
            print(f"  Trainable params: {trainable_params:,} ({100-frozen_pct:.1f}%)")

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

        print("="*80 + "\n")

    def _setup_optimizer(self):
        """Setup optimizer for all components"""
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        # Freeze ViT backbones if configured
        self._freeze_vit_backbone()

        # Collect all parameters (only trainable ones will be added)
        # Model parameters (backbones)
        params = []
        if self.vit_pretrained_path:
            # ViT-based model structure
            # Only collect trainable parameters (respects requires_grad flag set by _freeze_vit_backbone)
            trainable_model_params = [p for p in self.model.parameters() if p.requires_grad]
            params.extend(trainable_model_params)
            print(f"[Optimizer] Collected {len(trainable_model_params)} trainable parameter groups from ViT backbones")
        else:
            # if hasattr(self.model, 'rgb_backbone') or hasattr(self.model, 'flow_backbone'):
            # Action Recognition model structure
            if 'rgb' in self.modalities and hasattr(self.model, 'rgb_backbone'):
                params.extend(list(self.model.rgb_backbone.module.backbone.fast_path.layer4.parameters()))
                params.extend(list(self.model.rgb_backbone.module.backbone.slow_path.layer4.parameters()))
                params.extend(list(self.model.rgb_backbone.module.cls_head.parameters()))

            if 'flow' in self.modalities and hasattr(self.model, 'flow_backbone'):
                params.extend(list(self.model.flow_backbone.module.backbone.layer4.parameters()))
                params.extend(list(self.model.flow_backbone.module.cls_head.parameters()))

            if 'audio' in self.modalities and hasattr(self.model, 'audio_cls'):
                params.extend(list(self.model.audio_cls.parameters()))



        # Projector parameters
        if self.use_projector:
            for projector in self.projectors.values():
                params.extend(list(projector.parameters()))
        # Fusion module parameters
        params.extend(list(self.fusion_module.parameters()))

        # DG classifier parameters
        params.extend(list(self.dg_classifier.parameters()))

        # MIRO encoder parameters (if enabled)
        if self.use_miro:
            params.extend(list(self.miro_mean_encoder.parameters()))
            params.extend(list(self.miro_var_encoder.parameters()))

        # Create optimizer based on configuration
        if self.optimizer_type == 'sam':
            # SAM optimizer with AdamW as base optimizer
            self.optimizer = SAM(
                params,
                base_optimizer=torch.optim.AdamW,
                lr=lr,
                weight_decay=weight_decay,
                rho=self.sam_rho,
                adaptive=self.sam_adaptive
            )
        else:
            # Default SGD optimizer
            self.optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )

        # Setup scheduler if requested
        if self.use_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            # Get base optimizer for scheduler (SAM wraps the base optimizer)
            base_opt = self.optimizer.base_optimizer if self.optimizer_type == 'sam' else self.optimizer
            num_epochs = self.config.get('num_epochs', 15)
            min_lr = self.config.get('min_lr', 1e-6)

            self.scheduler = CosineAnnealingLR(
                base_opt,
                T_max=num_epochs,
                eta_min=min_lr
            )

        else:
            self.scheduler = None

        # Print optimizer summary
        total_trainable_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"\n[Optimizer] Summary:")
        print(f"  Optimizer type: {self.optimizer_type}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Total trainable parameters: {total_trainable_params:,}")
        print(f"  Use scheduler: {self.use_scheduler}")
        if self.freeze_vit_backbone:
            print(f"  ViT freezing: Enabled (freeze_vit_layers={self.freeze_vit_layers})")
        else:
            print(f"  ViT freezing: Disabled (all parameters trainable)")
        print()

    def extract_features(
            self,
            modality_inputs: Dict[str, torch.Tensor],
            labels: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Extract features from all modalities

        Args:
            modality_inputs: Dict of modality inputs

        Returns:
            modality_features: Dict of raw features per modality
            modality_projected: Dict of projected features per modality
        """
        total_loss = 0
        loss_dict = {}
        modality_features = {}
        modality_projected = {}

        # Extract features using model's per-modality forward
        with torch.set_grad_enabled(self.model.training):
            for modality in self.modalities:
                if modality in modality_inputs:
                    # Use model's forward_modality to get features
                    logits, features = self.model.forward_modality(
                        modality, modality_inputs[modality]
                    )
                    modality_features[modality] = features


                    loss = self.criterion(logits, labels)

                    total_loss += loss
                    # loss_dict[f'{modality}_loss'] = loss.item()
                    loss_dict[f'{modality}_loss'] = loss

                    # Project to common space
                    if self.use_projector:
                        projected = self.projectors[modality](features)
                        modality_projected[modality] = projected
                    else:
                        # Skip projection, use normalized features directly
                        modality_projected[modality] = F.normalize(features, dim=1)




        return modality_features, modality_projected, loss_dict, total_loss

    def extract_pretrained_features(
            self,
            modality_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract pretrained frozen features for MIRO

        This method extracts features using frozen backbones and fusion module,
        representing the pretrained model's output before fine-tuning.

        Args:
            modality_inputs: Dict of modality inputs

        Returns:
            fused_pretrained_features: Fused features from frozen backbones (for MIRO)
        """
        with torch.no_grad():
            modality_features = {}

            # Extract features from each modality (frozen)
            for modality in self.modalities:
                if modality in modality_inputs:
                    # Use model's forward_modality to get features
                    _, features = self.model.forward_modality(
                        modality, modality_inputs[modality]
                    )
                    modality_features[modality] = features.detach()

            # Fuse pretrained features using fusion module (frozen)
            feature_list = [modality_features[mod] for mod in sorted(modality_features.keys())]
            _, fused_features = self.fusion_module(feature_list)

        return fused_features.detach()

    def compute_fusion_loss(
            self,
            modality_features: Dict[str, torch.Tensor],
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute fusion classification loss

        Args:
            modality_features: Dict of raw features from backbones
            labels: Ground truth labels

        Returns:
            fusion_loss: Classification loss on fused features
            fusion_predictions: Predictions from fusion
            fused_features: Fused features
        """
        # Convert dict to list (fusion module expects list)
        # Both BottleneckFusion and AttentionFusion use raw features
        feature_list = [modality_features[mod] for mod in sorted(modality_features.keys())]

        # Fusion forward
        fusion_predictions, fused_features = self.fusion_module(feature_list)

        # Classification loss
        fusion_loss = self.criterion(fusion_predictions, labels)

        return fusion_loss, fusion_predictions, fused_features

    def compute_contrastive_loss(
            self,
            modality_projected: Dict[str, torch.Tensor],
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss on projected features

        Args:
            modality_projected: Dict of projected features
            labels: Ground truth labels

        Returns:
            contrastive_loss: Supervised contrastive loss
        """
        if not self.use_contras or len(modality_projected) < 2:
            return torch.tensor(0.0, device=self.device)

        # Stack modalities as different views
        # Shape: (batch_size, num_modalities, proj_dim)
        modality_list = sorted(modality_projected.keys())
        features = torch.stack([modality_projected[mod] for mod in modality_list], dim=1)

        # Compute supervised contrastive loss
        loss = self.contrastive_loss_fn(features, labels)

        return loss

    def compute_translation_loss(
            self,
            modality_projected: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute cross-modal translation loss

        Args:
            modality_projected: Dict of projected features

        Returns:
            translation_loss: Cross-modal alignment loss
        """
        if not self.use_modtrans or len(modality_projected) < 2:
            return torch.tensor(0.0, device=self.device)

        loss = self.cross_modal_trans(modality_projected)

        return loss

    def compute_dg_loss(
            self,
            fused_features: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute domain generalization loss

        Args:
            fused_features: Fused features
            labels: Ground truth labels

        Returns:
            dg_loss: Domain generalization classification loss
        """
        logits = self.dg_classifier(fused_features)
        loss = self.criterion(logits, labels)

        return loss

    def _compute_all_losses(
            self,
            modality_inputs0: Dict[str, torch.Tensor],
            labels0: torch.Tensor,
            modality_inputs1: Dict[str, torch.Tensor],
            labels1: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute all losses for two domains (used by both SGD and SAM)

        Args:
            modality_inputs0: Modality inputs from domain 0
            labels0: Labels from domain 0
            modality_inputs1: Modality inputs from domain 1
            labels1: Labels from domain 1

        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dict of individual losses for logging
        """

        # Forward pass for domain 0
        modality_features0, modality_projected0, modal_loss_dict0, modal_loss0 = self.extract_features(modality_inputs0,
                                                                                                       labels0)

        fusion_loss0, fusion_pred0, fusion_feat0 = self.compute_fusion_loss(modality_features0, labels0)
        # modal_loss_dict0['fusion'] = fusion_loss0.item()
        modal_loss_dict0['fusion'] = fusion_loss0

        # Forward pass for domain 1
        modality_features1, modality_projected1, modal_loss_dict1, modal_loss1 = self.extract_features(modality_inputs1,
                                                                                                       labels1)

        fusion_loss1, fusion_pred1, fusion_feat1 = self.compute_fusion_loss(modality_features1, labels1)
        # modal_loss_dict1['fusion'] = fusion_loss1.item()
        modal_loss_dict1['fusion'] = fusion_loss1

        # Cross-modal translation losses
        translation_loss0 = 0
        translation_loss1 = 0
        if self.use_modtrans:
            translation_loss0 = self.compute_translation_loss(modality_projected0)
            translation_loss1 = self.compute_translation_loss(modality_projected1)

        # Contrastive learning losses
        contrastive_loss0 = 0
        contrastive_loss1 = 0
        if self.use_contras:
            contrastive_loss0 = self.compute_contrastive_loss(modality_projected0, labels0)
            contrastive_loss1 = self.compute_contrastive_loss(modality_projected1, labels1)

        # Multi-modal losses (modal gap)
        mm_loss0 = self.alpha_trans * translation_loss0 + self.alpha_contrast * contrastive_loss0
        mm_loss1 = self.alpha_trans * translation_loss1 + self.alpha_contrast * contrastive_loss1
        # Domain contrastive loss (domain gap mitigation)
        # For each modality, concatenate features from both domains and compute contrastive loss
        dg_contrastive_loss = 0

        if self.use_domain_contrastive:
            # Prepare modal features for domain classification (concatenate D0 and D1 for each modality)
            modal_features_combined = {}

            # Concatenate features from both domains for each modality
            for modality in self.modalities:
                if modality in modality_features0 and modality in modality_features1:
                    modal_features_combined[modality] = torch.cat([
                        modality_features0[modality],
                        modality_features1[modality]
                    ], dim=0)

            # Add combined fusion features (use 'fusion' key to be consistent with GBL weights)
            modal_features_combined['fusion'] = torch.cat([fusion_feat0, fusion_feat1], dim=0)

            # Compute contrastive loss on combined domain features (helps learn domain-invariant representations)
            labels_combined = torch.cat([labels0, labels1], dim=0)
            _, dg_contrastive_loss = compute_modal_contrastive_losses(
                modal_features_combined, labels_combined, temperature=self.dg_contrastive_temp
            )

        # MMD loss (domain gap mitigation via distribution alignment)
        # Compute MMD between domain 0 and domain 1 for each modality and fusion
        total_mmd_loss = 0
        if self.use_mmd:
            mmd_losses = []

            # Per-modality MMD losses
            for modality in self.modalities:
                if modality in modality_features0 and modality in modality_features1:
                    mmd_loss = compute_mmd_loss(
                        modality_features0[modality],
                        modality_features1[modality],
                        kernel_num=self.mmd_kernel_num,
                        kernel_mul=self.mmd_kernel_mul
                    )
                    mmd_losses.append(mmd_loss)

            # Fusion MMD loss
            mmd_fusion_loss = compute_mmd_loss(
                fusion_feat0,
                fusion_feat1,
                kernel_num=self.mmd_kernel_num,
                kernel_mul=self.mmd_kernel_mul
            )
            mmd_losses.append(mmd_fusion_loss)

            # Total MMD loss (sum over all modalities + fusion)
            if len(mmd_losses) > 0:
                total_mmd_loss = sum(mmd_losses)
                # total_mmd_loss = sum(mmd_losses) / len(mmd_losses)

        # MIRO loss (Mutual Information Regularization)
        # Regularize fine-tuned features to stay close to pretrained features
        miro_reg_loss = 0

        if self.use_miro:
            # Fine-tuned fusion features (current model)
            miro_feat = torch.cat([fusion_feat0, fusion_feat1], dim=0)

            # Extract pretrained frozen features
            pretrained_feat0 = self.extract_pretrained_features(modality_inputs0)
            pretrained_feat1 = self.extract_pretrained_features(modality_inputs1)
            pretrained_feats = torch.cat([pretrained_feat0, pretrained_feat1], dim=0)

            # Compute MIRO variational lower bound (VLB)
            mean = self.miro_mean_encoder(miro_feat)  # Identity mapping
            var = self.miro_var_encoder(miro_feat)  # Learnable variance

            # VLB = (mean - pretrained)^2 / var + log(var + 1)
            vlb = (mean - pretrained_feats).pow(2).div(var) + (var + 1).log()
            miro_reg_loss = vlb.mean() / 2.0
        # Modal supervised losses (per-modality + fusion)
        if self.use_gblend:
            # Apply GBL weights to combine losses from both domains
            # Weighted combination: sum over (weight * domain_loss) for each modality
            modal_supervised_loss = 0

            # Domain 0: per-modality losses
            for modality in modal_loss_dict0.keys():
                mod_name = modality.replace('_loss', '')  # Remove '_loss' suffix
                weight = self.domain0_weights.get(mod_name, 0)
                modal_supervised_loss += weight * modal_loss_dict0[modality]

            # Domain 1: per-modality losses
            for modality in modal_loss_dict1.keys():
                mod_name = modality.replace('_loss', '')  # Remove '_loss' suffix
                weight = self.domain1_weights.get(mod_name, 0)
                modal_supervised_loss += weight * modal_loss_dict1[modality]

            # Note: Fusion losses are already included in modal_loss_dict0/1['fused']
            # and processed in the loops above, so no need to add them again
        else:
            # Uniform weighting (original)
            modal_supervised_loss = modal_loss0 + modal_loss1 + fusion_loss0 + fusion_loss1

        # Domain gap loss (ERM: combine both domains)
        labels_combined = torch.cat([labels0, labels1], dim=0)
        fusion_feat_combined = torch.cat([fusion_feat0, fusion_feat1], dim=0)
        dg_predictions = self.dg_classifier(fusion_feat_combined)
        dg_loss = self.criterion(dg_predictions, labels_combined)

        # Mixup loss (multimodal embedding mixup for data augmentation)
        mixup_loss = 0
        if self.use_mixup:
            # Apply mixup to features from both domains
            mixed_modality_features, mixed_fusion_feat, mixed_labels, lam = multimodal_mixup(
                modality_features0, fusion_feat0, labels0,
                modality_features1, fusion_feat1, labels1, num_classes = self.num_classes,
                alpha=self.mixup_alpha
            )

            # Compute DG classifier prediction on mixed features
            mixed_dg_predictions = self.dg_classifier(mixed_fusion_feat)

            # Compute mixup loss with soft labels
            mixup_loss = mixup_criterion(mixed_dg_predictions, mixed_labels, self.criterion)

        # Total loss
        total_loss = modal_supervised_loss + self.ld*(mm_loss0 + mm_loss1) + dg_loss + \
                     self.dg_contrastive_weight * dg_contrastive_loss + \
                     self.mmd_weight * total_mmd_loss + \
                     self.miro_weight * miro_reg_loss + \
                     self.mixup_weight * mixup_loss

        # Prepare loss dict for logging
        loss_dict = {
            'domain0_modal_loss': modal_loss0.item(),
            'domain1_modal_loss': modal_loss1.item(),
            'fusion_loss0': fusion_loss0.item(),
            'fusion_loss1': fusion_loss1.item(),
            'contrastive_loss0': contrastive_loss0 if isinstance(contrastive_loss0,
                                                                 (int, float)) else contrastive_loss0.item(),
            'contrastive_loss1': contrastive_loss1 if isinstance(contrastive_loss1,
                                                                 (int, float)) else contrastive_loss1.item(),
            'translation_loss0': translation_loss0 if isinstance(translation_loss0,
                                                                 (int, float)) else translation_loss0.item(),
            'translation_loss1': translation_loss1 if isinstance(translation_loss1,
                                                                 (int, float)) else translation_loss1.item(),
            'dg_loss': dg_loss.item(),
            'dg_contrastive_loss': dg_contrastive_loss if isinstance(dg_contrastive_loss,
                                                                     (int, float)) else dg_contrastive_loss.item(),
            'mmd_loss': total_mmd_loss if isinstance(total_mmd_loss, (int, float)) else total_mmd_loss.item(),
            'miro_loss': miro_reg_loss if isinstance(miro_reg_loss, (int, float)) else miro_reg_loss.item(),
            'mixup_loss': mixup_loss if isinstance(mixup_loss, (int, float)) else mixup_loss.item(),
            'total_loss': total_loss.item()
        }

        # Add detailed per-modality losses for GBL tracking
        if self.use_gblend:
            loss_dict['domain0_modality_losses'] = {k: v for k, v in modal_loss_dict0.items()}
            loss_dict['domain1_modality_losses'] = {k: v for k, v in modal_loss_dict1.items()}

        return total_loss, loss_dict, fusion_pred0, fusion_pred1, dg_predictions

    def train_step_two_domains(
            self,
            modality_inputs0: Dict[str, torch.Tensor],
            labels0: torch.Tensor,
            modality_inputs1: Dict[str, torch.Tensor],
            labels1: torch.Tensor
    ) -> Dict[str, float]:
        """
        Training step for two domains with support for both SGD and SAM optimizers

        Args:
            modality_inputs0: Modality inputs from domain 0
            labels0: Labels from domain 0
            modality_inputs1: Modality inputs from domain 1
            labels1: Labels from domain 1

        Returns:
            Dict of losses for logging
        """
        # Set all modules to training mode
        self.model.train()

        # Set projectors to training mode
        if self.use_projector:
            for projector in self.projectors.values():
                projector.train()


        # Set fusion module to training mode
        self.fusion_module.train()

        # Set DG classifier to training mode
        self.dg_classifier.train()

        if self.optimizer_type == 'sam':
            # SAM two-step update
            # First step: compute gradients and perturb weights
            total_loss, loss_dict, fusion_pred0, fusion_pred1, dg_predictions = self._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1
            )
            total_loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # Second step: recompute forward-backward with perturbed weights
            total_loss_new, _, _, _, _ = self._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1
            )
            total_loss_new.backward()
            self.optimizer.second_step(zero_grad=True)

            # Step scheduler if enabled
            if self.scheduler is not None:
                self.scheduler.step()

        else:
            # Standard SGD update
            total_loss, loss_dict, fusion_pred0, fusion_pred1, dg_predictions = self._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1
            )
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Step scheduler if enabled
            if self.scheduler is not None:
                self.scheduler.step()

        return loss_dict, fusion_pred0, fusion_pred1, dg_predictions

    def update(self, loss: torch.Tensor):
        """Update model parameters (for compatibility with base class)"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_loss_history(self, epoch: int, split: str, domain: str, losses: Dict[str, float]):
        """
        Update loss history for GBL weight computation

        Args:
            epoch: Current epoch number
            split: 'train' or 'val'
            domain: 'D0' or 'D1'
            losses: Dict of modality losses (e.g., {'rgb': 0.5, 'flow': 0.3, 'fusion': 0.4})
        """
        if not self.use_gblend:
            return

        if epoch not in self.loss_history[domain][split]:
            self.loss_history[domain][split][epoch] = {}

        self.loss_history[domain][split][epoch] = losses.copy()

    def update_gbl_weights(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """
        Update GBL weights based on loss history using exponential smoothing

        Args:
            epoch: Current epoch number (must be > 0 to compute weight updates)

        Returns:
            Dict of updated weights for both domains
        """
        if not self.use_gblend or epoch == 0:
            return None

        updated_weights = {}

        for domain in ['D0', 'D1']:
            # Get modality list for this domain (exclude 'fusion')
            modality_list = [mod for mod in self.modalities]
            modality_list.append('fusion')  # Add fusion

            # Compute GBL coefficients for each modality
            modal_weights = []
            for modality in modality_list:
                # Get losses for this modality
                prev_train = self.loss_history[domain]['train'].get(epoch - 1, {}).get(modality, 0)
                prev_val = self.loss_history[domain]['val'].get(epoch - 1, {}).get(modality, 0)
                curr_train = self.loss_history[domain]['train'].get(epoch, {}).get(modality, 0)
                curr_val = self.loss_history[domain]['val'].get(epoch, {}).get(modality, 0)

                # Compute GBL coefficient
                coef = GBL.compute_gblend_coef(curr_train, curr_val, prev_train, prev_val)
                modal_weights.append(coef)

            # Normalize weights
            normed_weights = GBL.normalize_weights(modal_weights)

            # Apply exponential smoothing
            if domain == 'D0':
                for idx, modality in enumerate(modality_list):
                    old_weight = self.domain0_weights[modality]
                    new_weight = old_weight * (self.gbl_momentum) + normed_weights[idx] * (1 - self.gbl_momentum)
                    self.domain0_weights[modality] = new_weight

                updated_weights['D0'] = self.domain0_weights.copy()
            else:
                for idx, modality in enumerate(modality_list):
                    old_weight = self.domain1_weights[modality]
                    new_weight = old_weight * (self.gbl_momentum) + normed_weights[idx] * (1 - self.gbl_momentum)
                    self.domain1_weights[modality] = new_weight

                updated_weights['D1'] = self.domain1_weights.copy()

        return updated_weights

    def predict(self, modality_inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Make predictions using fusion

        Args:
            modality_inputs: Dict of modality inputs

        Returns:
            Predictions (from fusion module)
        """
        # Set all modules to eval mode
        self.model.eval()

        # Set projectors to eval mode
        if self.use_projector:
            for projector in self.projectors.values():
                projector.eval()

        # Set fusion module to eval mode
        self.fusion_module.eval()

        # Set DG classifier to eval mode
        self.dg_classifier.eval()

        with torch.no_grad():
            # Extract features
            modality_features, _, _, _ = self.extract_features(modality_inputs, labels)

            # Fusion (use raw features)
            feature_list = [modality_features[mod] for mod in sorted(modality_features.keys())]
            _, fused_features  = self.fusion_module(feature_list)
            # 3. Get final predictions from the DG classifier
            predictions = self.dg_classifier(fused_features)

        return predictions

    def predict_all(self, modality_inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Make predictions using both fusion module and DG classifier

        Args:
            modality_inputs: Dict of modality inputs

        Returns:
            fusion_predictions: Predictions from fusion module
            dg_predictions: Predictions from DG classifier
        """
        # Set all modules to eval mode
        self.model.eval()
        if self.use_projector:
            for projector in self.projectors.values():
                projector.eval()

        self.fusion_module.eval()
        self.dg_classifier.eval()

        with torch.no_grad():
            # Extract features
            modality_features, _, _, _ = self.extract_features(modality_inputs, labels)

            # Fusion predictions
            feature_list = [modality_features[mod] for mod in sorted(modality_features.keys())]
            fusion_predictions, fused_features = self.fusion_module(feature_list)

            # DG classifier predictions
            dg_predictions = self.dg_classifier(fused_features)

        return fusion_predictions, dg_predictions

    def compute_val_losses(self, modality_inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute per-modality validation losses for GBL weight computation

        Args:
            modality_inputs: Dict of modality inputs
            labels: Ground truth labels

        Returns:
            Dict of per-modality losses (e.g., {'rgb': 0.5, 'flow': 0.3, 'audio': 0.4, 'fusion': 0.45})
        """
        # Set all modules to eval mode
        self.model.eval()
        if self.use_projector:
            for projector in self.projectors.values():
                projector.eval()

        self.fusion_module.eval()
        self.dg_classifier.eval()

        with torch.no_grad():
            # Extract features
            modality_features, _, val_losses, _ = self.extract_features(modality_inputs, labels)

            # Compute fusion loss
            feature_list = [modality_features[mod] for mod in sorted(modality_features.keys())]
            fusion_predictions, fused_features = self.fusion_module(feature_list)
            fusion_loss = self.criterion(fusion_predictions, labels)
            val_losses['fusion'] = fusion_loss.item()

        return val_losses