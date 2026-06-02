 

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm

# Add parent directory to path
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

# Add third_party to path
_third_party = _project_root / 'third_party'
if str(_third_party) not in sys.path:
    sys.path.insert(0, str(_third_party))
from mmcv import Config
from datasets.action_recognition import HACDataset, EPICKitchensDataset
from common_utils.config import get_config, print_config, save_config
from datasets.base import MultiModalCollator
from algorithms.early_fusion_dg_mae import EarlyFusionDG
from collections import OrderedDict
import torch.nn.functional as F


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def remap_checkpoint_keys(checkpoint):
    """
    Robustly extracts and MERGES state_dict from a 'split' checkpoint.

    Handles checkpoints where 'module' contains the blocks,
    but 'pos_embed' and 'cls_token' are at the top level.
    """
    top_level_keys = set(checkpoint.keys())
    if 'module' in top_level_keys and isinstance(checkpoint['module'], dict):
        print("  [remap_keys] Found 'module' container key. Using it as base.")
        state_dict = checkpoint['module'].copy()  # 使用 .copy()
    else:
        print("  [remap_keys] No 'module' container. Assuming flat structure.")
        state_dict = checkpoint.copy()

    
    keys_to_merge = ['pos_embed', 'cls_token']
    merged_count = 0
    for key in keys_to_merge:
        if key in top_level_keys and key not in state_dict:
            print(f"  [remap_keys] Merging top-level key '{key}' into state_dict.")
            state_dict[key] = checkpoint[key]
            merged_count += 1

    if merged_count > 0:
        print(f"  [remap_keys] Merged {merged_count} top-level keys.")

    
    all_keys = list(state_dict.keys())
    if not all_keys:
        print("  [remap_keys] ERROR: state_dict is empty.")
        return state_dict

    prefixes_to_strip = ['module.', 'backbone.', 'encoder.', '_orig_mod.']
    needs_stripping = any(key.startswith(p) for key in all_keys for p in prefixes_to_strip)

    if not needs_stripping:
        print("  [remap_keys] Keys are clean (no prefix). Returning as-is.")
        return state_dict  

 
    print("  [remap_keys] Stripping prefixes from keys...")
    new_dict = OrderedDict()
    for key in all_keys:
        stripped = False
        for prefix in prefixes_to_strip:
            if key.startswith(prefix):
                new_dict[key[len(prefix):]] = state_dict[key]
                stripped = True
                break
        if not stripped:
            new_dict[key] = state_dict[key]

    return new_dict


def interpolate_pos_embed_videomae(model, checkpoint_model):
 
    if 'pos_embed' not in checkpoint_model:
        print("  [pos_embed] 'pos_embed' not found in checkpoint. Skipping interpolation.")
        return checkpoint_model

    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    # Get target T (temporal) and P (spatial)
    target_num_frames = model.all_frames
    target_tubelet_size = model.patch_embed.tubelet_size[0]  # e.g., 2
    target_T = target_num_frames // target_tubelet_size  # e.g., 16 // 2 = 8
    target_P = int(model.patch_embed.grid_size[1])  # e.g., 14 (from 224//16)

    if target_T * target_P * target_P != num_patches:
        print(f"  [pos_embed] ERROR: Patch calculation mismatch. {target_T}*{target_P}*{target_P} != {num_patches}")
        return checkpoint_model

    # Infer original T and P from checkpoint shape
    L_orig = pos_embed_checkpoint.shape[1] - num_extra_tokens

    # Assume the checkpoint is from default VideoMAE v2 (16 frames, 224x224, tubelet 2)
    # T_orig = 8, P_orig = 14. L_orig = 8*14*14 = 1568
    orig_T = 8
    orig_P = 14

    if L_orig != (orig_T * orig_P * orig_P):
        print(
            f"  [pos_embed] Warning: Checkpoint pos_embed shape ({L_orig} patches) does not match assumed 16-frame (T=8, P=14) default ({orig_T * orig_P * orig_P}).")
        try:
            # Attempt to infer P_orig assuming T_orig=8
            orig_P = int((L_orig // orig_T) ** 0.5)
            if L_orig != (orig_T * orig_P * orig_P):
                raise ValueError("Non-integer P or T*P*P mismatch")
            print(f"  [pos_embed] Inferred orig_P={orig_P} assuming orig_T={orig_T}.")
        except Exception as e:
            print(f"  [pos_embed] Failed to infer original T and P. Skipping interpolation. Error: {e}")
            return checkpoint_model

    print(f"  [pos_embed] Orig (T,P): ({orig_T}, {orig_P}), Target (T,P): ({target_T}, {target_P})")

    # 1. Spatial Interpolation (if P_orig != P_target)
    if orig_P != target_P:
        print(f"  [pos_embed] Interpolating SPATIAL from {orig_P}x{orig_P} to {target_P}x{target_P}")

        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

        # B, L, C -> B, T, P, P, C -> (B*T), P, P, C
        pos_tokens = pos_tokens.reshape(-1, orig_T, orig_P, orig_P, embedding_size)
        pos_tokens = pos_tokens.reshape(-1, orig_P, orig_P, embedding_size)

        # (B*T), P, P, C -> (B*T), C, P, P
        pos_tokens = pos_tokens.permute(0, 3, 1, 2)

        # Interpolate
        pos_tokens = F.interpolate(
            pos_tokens,
            size=(target_P, target_P),
            mode='bicubic',
            align_corners=False
        )

        # (B*T), C, P_new, P_new -> (B*T), P_new, P_new, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1)

        # (B*T), P_new, P_new, C -> B, T_orig, P_new, P_new, C
        pos_tokens = pos_tokens.reshape(-1, orig_T, target_P, target_P, embedding_size)

        # B, T_orig, P_new, P_new, C -> B, (T_orig*P_new*P_new), C
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L_new_spatial, C

        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        # Update pos_embed_checkpoint for next step
        pos_embed_checkpoint = new_pos_embed

    # 2. Temporal Interpolation (if T_orig != T_target)
    
    if orig_T != target_T:
        print(f"  [pos_embed] Interpolating TEMPORAL from {orig_T} to {target_T}")

        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

        # B, (T_orig*P_new*P_new), C -> B, T_orig, P_new, P_new, C
        pos_tokens = pos_tokens.reshape(-1, orig_T, target_P, target_P, embedding_size)

        # B, T, P, P, C -> B, P, P, C, T
        pos_tokens = pos_tokens.permute(0, 2, 3, 4, 1)

        # B, P, P, C, T -> (B*P*P), C, T
        pos_tokens = pos_tokens.reshape(-1, embedding_size, orig_T)

        # Interpolate
        pos_tokens = F.interpolate(
            pos_tokens,
            size=target_T,
            mode='linear'
        )

        # (B*P*P), C, T_new -> B, P, P, C, T_new
        pos_tokens = pos_tokens.reshape(1, target_P, target_P, embedding_size, target_T)

        # B, P, P, C, T_new -> B, T_new, P, P, C
        pos_tokens = pos_tokens.permute(0, 4, 1, 2, 3)

        # B, T_new, P, P, C -> B, (T_new*P*P), C
        pos_tokens = pos_tokens.flatten(1, 3)  # B, L_new, C

        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

    return checkpoint_model


def _prepare_videomae_input(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure input is 5D [B, C, T, H, W] for VideoMAEv2.
    Accepts common variants:
      - [B, 1, C, T, H, W]  -> squeeze dim 1
      - [B, T, C, H, W]     -> permute to [B, C, T, H, W]
      - [B, H, W, T, C]     -> permute to [B, C, T, H, W]
    """
    # Squeeze a singleton "num_clips / num_views" dimension
    if x.dim() == 6 and x.size(1) == 1:
        x = x.squeeze(1)  # [B, C, T, H, W]

    if x.dim() != 5:
        raise ValueError(f"VideoMAEv2 expects 5D input [B, C, T, H, W], got shape {tuple(x.shape)}")

    # Try to infer layout & permute to [B, C, T, H, W]
    B, A, B2, C2, D2 = x.shape  # dummy read to remind dimensions

    if x.shape[1] in (1, 2, 3):  # likely [B, C, T, H, W] already
        pass
    elif x.shape[2] in (1, 2, 3):  # [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
    elif x.shape[-1] in (1, 2, 3):  # [B, H, W, T, C]
        x = x.permute(0, 4, 3, 1, 2).contiguous()
    else:
        # Last-ditch: assume [B, T, H, W, C]
        if x.shape[-1] <= 8:  # C is small
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        else:
            raise ValueError(f"Cannot infer channel dimension from shape {tuple(x.shape)}")

    # Type safety
    if x.dtype != torch.float32:
        x = x.float()
    return x


def get_dataset(config, split, domains):
    """
    Create dataset based on config.dataset.name

    Supports:
    - 'epic_kitchens': EPIC-Kitchens dataset
    - 'hac': HAC (Human-Animal-Computer) dataset
    """

    # Load MMAction2 configs if needed
    cfg_video = None
    cfg_flow = None

    if 'rgb' in config.dataset.modalities:
        cfg_video = Config.fromfile(config.dataset.cfg_video_path)

    if 'flow' in config.dataset.modalities:
        cfg_flow = Config.fromfile(config.dataset.cfg_flow_path)

    # Common dataset parameters
    dataset_params = {
        'split': split,
        'domains': domains,
        'modalities': config.dataset.modalities,
        'data_root': config.dataset.data_root,
        'cfg_video': cfg_video,
        'cfg_flow': cfg_flow,
        'audio_sample_rate': config.dataset.get('audio_sample_rate', 16000),
        'audio_segment_length': config.dataset.get('audio_segment_length', 160000),
        'load_train_split': config.dataset.get('load_train_split', False)
    }

    # Select dataset class based on config
    dataset_name = config.dataset.name.lower()

    if dataset_name == 'epic_kitchens':
        # EPIC-Kitchens specific parameter
        dataset_params['sample_dur'] = config.dataset.get('sample_dur', 10)
        return EPICKitchensDataset(**dataset_params)

    elif dataset_name == 'hac':
        # HAC dataset (no sample_dur parameter)
        return HACDataset(**dataset_params)

    else:
        raise ValueError(
            f"Unknown dataset: {config.dataset.name}. "
            f"Supported datasets: 'epic_kitchens', 'hac'"
        )


def create_base_model_videomae_v2(config):
    """
    Create base model with VideoMAEv2 backbones  

    Architecture:
        - RGB: VideoMAEv2 ViT-Base -> [B, 768]
        - Flow: VideoMAEv2 ViT-Base-> [B, 768]
        - Audio: AST or ViT-Base -> [B, 768]

    Input:
        - RGB: [B, 3, 16, 224, 224]
        - Flow: [B, 2, 16, 224, 224]
        - Audio: [B, 1, 128, time]
    """
    import sys
    from pathlib import Path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Add VideoMAEv2 to path
    videomae_v2_path = Path(config.model.get('videomae_v2_path',
                                             'MMDG_Bench/third_party/VideoMAEv2'))
    if str(videomae_v2_path) not in sys.path:
        sys.path.insert(0, str(videomae_v2_path))

    from models.modeling_finetune import vit_base_patch16_224

    backbones = {}
    feature_dims = {}
    num_classes = config.task.num_classes

    # Get VideoMAEv2 pretrained path
    videomae_v2_pretrained = config.model.get('videomae_pretrained_path', None)
    if videomae_v2_pretrained:
        videomae_v2_pretrained = str(_project_root / videomae_v2_pretrained)

    print("\n" + "=" * 80)
    print("Initializing VideoMAEv2-Based Multi-Modal Model")
    print("=" * 80)

    # ========== RGB: VideoMAEv2 ViT-Base  ==========
    if 'rgb' in config.dataset.modalities:
        print("\n[RGB] Initializing VideoMAEv2 ViT-Base (16 frames)...")

        rgb_videomae = vit_base_patch16_224(
            pretrained=False,
            num_classes=num_classes,
            all_frames=16,
            tubelet_size=2,
            drop_rate=0.,
            drop_path_rate=0.1,
            attn_drop_rate=0.,
            # with_cp=False,
            with_cp=True,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        target_input_size = 224
        # Load pretrained weights
        if videomae_v2_pretrained and os.path.exists(videomae_v2_pretrained):           
            checkpoint = torch.load(videomae_v2_pretrained, map_location='cpu')            
            state_dict = remap_checkpoint_keys(checkpoint)            
            print("  [RGB] Interpolating pos_embed...")
            state_dict = interpolate_pos_embed_videomae(
                model=rgb_videomae,
                checkpoint_model=state_dict
            )
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}            
            missing, unexpected = rgb_videomae.load_state_dict(state_dict, strict=False)
            print(f"Loaded VideoMAEv2 weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
        else:
            print(f"No pretrained weights, using random initialization")

        rgb_videomae = rgb_videomae.to(device)
        backbones['rgb'] = rgb_videomae
        feature_dims['rgb'] = 768
        print(f"  ✓ RGB VideoMAEv2 initialized: [B, 3, 16, 224, 224] -> 768-dim")

    # ========== Flow: VideoMAEv2 ViT-Base ==========
    if 'flow' in config.dataset.modalities:
        print("\n[Flow] Initializing VideoMAEv2 ViT-Base (16 frames, 2-channel)...")

        flow_videomae = vit_base_patch16_224(
            pretrained=False,
            num_classes=num_classes,
            all_frames=16,
            tubelet_size=2,
            drop_rate=0.,
            drop_path_rate=0.1,
            attn_drop_rate=0.,
            with_cp=False,
            # with_cp=True,
            use_mean_pooling=True,
            init_scale=0.001,
        )

        # Load pretrained weights
        pretrained_loaded = False
        if videomae_v2_pretrained and os.path.exists(videomae_v2_pretrained):
            print(f"  Loading VideoMAEv2 weights: {videomae_v2_pretrained}")
            checkpoint = torch.load(videomae_v2_pretrained, map_location='cpu')
             
            state_dict = remap_checkpoint_keys(checkpoint)

             
            print("  [Flow] Interpolating pos_embed...")
            state_dict = interpolate_pos_embed_videomae(
                model=flow_videomae,
                checkpoint_model=state_dict
            )

            
            old_conv_weight = state_dict.get('patch_embed.proj.weight')
            old_conv_bias = state_dict.get('patch_embed.proj.bias')

            if old_conv_weight is not None:
                print("  [Flow] Found 'patch_embed.proj' weights in checkpoint for 2-ch init.")

                 
                old_conv_template = flow_videomae.patch_embed.proj   
                new_conv = nn.Conv3d(
                    in_channels=2,
                    out_channels=old_conv_template.out_channels,
                    kernel_size=old_conv_template.kernel_size,
                    stride=old_conv_template.stride,
                    padding=old_conv_template.padding,
                    bias=(old_conv_bias is not None)
                )

                
                with torch.no_grad():
                     
                    new_conv.weight.data.copy_(old_conv_weight.data[:, :2, :, :, :])
                    if old_conv_bias is not None:
                        new_conv.bias.data.copy_(old_conv_bias.data)

                 
                flow_videomae.patch_embed.proj = new_conv
                print("  ✓ Initialized 2-channel Conv3d from pretrained weights")
                pretrained_loaded = True

                 
                state_dict = {k: v for k, v in state_dict.items()
                              if not k.startswith('patch_embed.proj')}
            else:
                print("  ⚠ [Flow] Could not find 'patch_embed.proj' in state_dict. Using random init for 2-ch conv.")
                pretrained_loaded = False
            
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}

             
            missing, unexpected = flow_videomae.load_state_dict(state_dict, strict=False)
            print(f"  ✓ Loaded VideoMAEv2 weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
            print(f"    Missing keys: {missing}")
            print(f"    Unexpected keys: {unexpected}")
        else:
            print(f"  ⚠ No pretrained weights, using random initialization")

        # Ensure patch_embed.proj is always 2-channel (flow has 2 channels, not 3)
        if flow_videomae.patch_embed.proj.in_channels != 2:
            old_proj = flow_videomae.patch_embed.proj
            new_conv = nn.Conv3d(
                in_channels=2,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=(old_proj.bias is not None)
            )
            flow_videomae.patch_embed.proj = new_conv
            print("  Replaced patch_embed.proj with randomly initialized 2-channel Conv3d")

        flow_videomae = flow_videomae.to(device)
        backbones['flow'] = flow_videomae
        feature_dims['flow'] = 768
        print(f"  ✓ Flow VideoMAEv2 initialized: [B, 2, 16, 224, 224] -> 768-dim")

    # ========== Audio: AST (Audio Spectrogram Transformer) or ViT-Base ==========
    if 'audio' in config.dataset.modalities:
        print("\n[Audio] Initializing Audio ViT (AST or ViT-Base)...")

        # Option 1: Use Audio Spectrogram Transformer (AST) if available
        use_ast = config.model.get('use_ast', False)
        ast_pretrained_path = config.model.get('ast_pretrained_path', None)
        if ast_pretrained_path:
            ast_pretrained_path = str(_project_root / ast_pretrained_path)

        if use_ast:
            print("  Using AST (Audio Spectrogram Transformer)...")
            try:
                from transformers import ASTForAudioClassification, ASTConfig
 
                # Create AST model
                if ast_pretrained_path and os.path.exists(ast_pretrained_path):
                    print(f"  Loading AST from: {ast_pretrained_path}")
                    audio_vit = ASTForAudioClassification.from_pretrained(
                        ast_pretrained_path,
                        num_labels=num_classes,
                        ignore_mismatched_sizes=True
                    )
                else:
                    print("  ⚠ No AST pretrained weights, initializing from scratch")
                    ast_config = ASTConfig(num_labels=num_classes)
                    audio_vit = ASTForAudioClassification(ast_config)

                # Wrap AST to match our interface
                class ASTWrapper(nn.Module):
                    def __init__(self, ast_model):
                        super().__init__()
                        self.ast = ast_model

                    def forward_features(self, x):
                        """Extract 768-dim features"""
                        outputs = self.ast.audio_spectrogram_transformer.embeddings(x)  # Get embeddings first
                        outputs = self.ast.audio_spectrogram_transformer.encoder(outputs)  # Then encode
                        features = outputs.last_hidden_state[:, 0]  # [CLS] token
                        return features

                    def forward(self, x):
                        """Forward pass with classification"""
                        outputs = self.ast(x)
                        return outputs.logits

                    @property
                    def head(self):
                        return self.ast.classifier

                audio_vit = ASTWrapper(audio_vit)
                audio_vit = audio_vit.to(device)
                backbones['audio'] = audio_vit
                feature_dims['audio'] = 768
                print(f"  ✓ AST initialized: 768-dim features, {num_classes} classes")

            except ImportError:
                print("  ⚠ AST (transformers library) not available, falling back to ViT-Base")
                use_ast = False

        if not use_ast:
            # Option 2: Use standard ViT-Base with spectrogram as "image"
            print("  Using ViT-Base for audio spectrograms...")

            # Get local pretrained path for ViT
            vit_pretrained_path = config.model.get('vit_pretrained_path', None)
            if vit_pretrained_path:
                vit_pretrained_path = str(_project_root / vit_pretrained_path)

            audio_vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=num_classes,
                img_size=224
            )

            # Load pretrained weights if available
            if vit_pretrained_path and os.path.exists(vit_pretrained_path):
                print(f"  Loading pretrained weights from: {vit_pretrained_path}")
                state_dict = torch.load(vit_pretrained_path, map_location='cpu')
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
                missing, unexpected = audio_vit.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Loaded pretrained weights (missing: {len(missing)}, unexpected: {len(unexpected)})")

            audio_vit = audio_vit.to(device)
            backbones['audio'] = audio_vit
            feature_dims['audio'] = 768
            print(f"  ✓ Audio ViT initialized: 768-dim features, {num_classes} classes")

    print("\n" + "=" * 80)
    print("Model Summary:")
    print(f"  Modalities: {list(backbones.keys())}")
    print(f"  Feature dims: {feature_dims}")
    print(f"  All dims = 768 -> No projector needed!")
    print("=" * 80 + "\n")

    # Create wrapper for VideoMAEv2
    class MultiModalVideoMAEv2Wrapper(nn.Module):
        def __init__(self, backbones):
            super().__init__()
            self.backbones_dict = backbones

            if 'rgb' in backbones:
                self.rgb_backbone = backbones['rgb']
            if 'flow' in backbones:
                self.flow_backbone = backbones['flow']
            if 'audio' in backbones:
                self.audio_backbone = backbones['audio']

        def forward_modality(self, modality_name: str, modality_data: torch.Tensor):
            """
            VideoMAEv2 forward

            Args:
                modality_name: 'rgb', 'flow', or 'audio'
                modality_data:
                    - RGB/Flow: [B, num_clips, C, T, H, W] (6D) or [B, C, T, H, W] (5D) where T=16
                    - Audio: [B, 1, freq, time]

            Returns:
                (logits, features): ([B, num_classes], [B, 768])
            """

            # Handle audio separately (uses 2D ViT/AST)
            if modality_name == 'audio':
                vit = self.audio_backbone

                import torch.nn.functional as F

                # Audio might be 4D [B, 1, freq, time] or incorrectly 5D [B, 1, 1, freq, time]
                if modality_data.dim() == 5:
                    modality_data = modality_data.squeeze(2)

                if modality_data.dim() == 4:
                    # Check if using AST (has 'ast' attribute) or standard ViT
                    is_ast = hasattr(vit, 'ast')

                    if is_ast:
                        # AST expects [B, 1, 128, time]
                        modality_data = modality_data.squeeze(1)  # [B, 1, 128, 1004] -> [B, 128, 1004]

                        # AST has fixed positional embeddings, resize to expected input size
                        # Typical AST expects [B, 128, 1024]
                        modality_data = F.interpolate(
                            modality_data.unsqueeze(1),  # Add channel dim for interpolate: [B, 1, 128, 1004]
                            size=(128, 1024),  # Target size
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)  # Remove channel dim: [B, 128, 1024]
                    else:
                        # Standard ViT expects [B, 3, 224, 224]
                        B = modality_data.size(0)
                        modality_data = F.interpolate(
                            modality_data,
                            size=(224, 224),
                            mode='bilinear',
                            align_corners=False
                        )
                        modality_data = modality_data.repeat(1, 3, 1, 1)

                # Forward through audio model
                if hasattr(vit, 'forward_features'):
                    features = vit.forward_features(modality_data)
                    if features.dim() == 3:
                        features = features[:, 0]
                    logits = vit.head(features) if hasattr(vit, 'head') else None
                else:
                    features = vit.forward_features(modality_data)
                    logits = vit(modality_data)

                return logits, features

            # RGB/Flow uses VideoMAEv2
            if modality_name == 'rgb':
                videomae = self.rgb_backbone
            elif modality_name == 'flow':
                videomae = self.flow_backbone
            else:
                raise ValueError(f"Unknown modality: {modality_name}")

            # VideoMAEv2 expects [B, C, T, H, W]

            modality_data = _prepare_videomae_input(modality_data)
            if modality_data.dim() != 5:
                raise ValueError(f"VideoMAEv2 expects 5D input [B, C, T, H, W], got shape {modality_data.shape}")

            B, C, T, H, W = modality_data.shape

            # Temporal sampling: VideoMAEv2 expects 16 frames
            # If input has more frames (e.g., 32), uniformly sample 16 frames
            target_frames = 16  # VideoMAEv2 is initialized with all_frames=16
            if T != target_frames:
                original_T = T  # Save original frame count for logging
                if T > target_frames:
                    # Uniform sampling: select 16 frames from T frames
                    indices = torch.linspace(0, T - 1, target_frames).long()
                    modality_data = modality_data[:, :, indices, :, :]  # [B, C, 16, H, W]
                    T = target_frames
                    # print(f"  [Temporal Sampling] {modality_name}: {target_frames} frames sampled from {original_T} frames")
                else:
                    # T < target_frames: repeat frames (rare case)
                    repeat_factor = (target_frames + T - 1) // T
                    modality_data = modality_data.repeat(1, 1, repeat_factor, 1, 1)[:, :, :target_frames, :, :]
                    T = target_frames
                    # print(f"  [Temporal Sampling] {modality_name}: {original_T} frames repeated to {target_frames} frames")

            # Resize to 224x224 if needed
            if H != 224 or W != 224:
                import torch.nn.functional as F
                # Reshape to [B*T, C, H, W] for efficient resize
                modality_data = modality_data.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
                modality_data = modality_data.reshape(B * T, C, H, W)
                modality_data = F.interpolate(
                    modality_data,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
                # Reshape back to [B, C, T, 224, 224]
                modality_data = modality_data.reshape(B, T, C, 224, 224)
                modality_data = modality_data.permute(0, 2, 1, 3, 4)

            # VideoMAEv2 forward
            features = videomae.forward_features(modality_data)  # [B, num_patches, 768]

            # Extract class token
            if features.dim() == 3:
                features = features[:, 0]  # [B, 768]

            # Get logits
            if hasattr(videomae, 'head') and videomae.head is not None:
                logits = videomae.head(features)
            else:
                logits = None

            return logits, features

    wrapper_model = MultiModalVideoMAEv2Wrapper(backbones)
    return wrapper_model, feature_dims


def train_epoch(algorithm, dataloader0, dataloader1, epoch, config):
    """Train for one epoch with two domain dataloaders"""
    algorithm.model.train()
    total_losses = {}
    correct = 0
    total = 0

    # GBL loss accumulators
    gbl_losses = None

    if algorithm.use_gblend:
        gbl_losses = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses['fusion'] = 0.0

    # Create iterators
    loader0_iter = iter(dataloader0)
    loader1_iter = iter(dataloader1)
    min_len = min(len(dataloader0), len(dataloader1))

    pbar = tqdm.tqdm(range(min_len), desc=f"Epoch {epoch + 1}")

    for batch_idx in pbar:
        try:
            modality_inputs0, labels0 = next(loader0_iter)
            modality_inputs1, labels1 = next(loader1_iter)
        except StopIteration:
            break

        # Process domain 0
        modality_inputs0 = {
            k: v.to(algorithm.device) if isinstance(v, torch.Tensor) and k != 'audio' else v
            for k, v in modality_inputs0.items()
        }
        if 'audio' in modality_inputs0 and isinstance(modality_inputs0['audio'], torch.Tensor):
            modality_inputs0['audio'] = modality_inputs0['audio'].unsqueeze(1).to(algorithm.device)
        if 'rgb' in modality_inputs0 and isinstance(modality_inputs0['rgb'], torch.Tensor):
            modality_inputs0['rgb'] = modality_inputs0['rgb'].squeeze(1)
        if 'flow' in modality_inputs0 and isinstance(modality_inputs0['flow'], torch.Tensor):
            modality_inputs0['flow'] = modality_inputs0['flow'].squeeze(1)
        labels0 = labels0.to(algorithm.device)

        # Process domain 1
        modality_inputs1 = {
            k: v.to(algorithm.device) if isinstance(v, torch.Tensor) and k != 'audio' else v
            for k, v in modality_inputs1.items()
        }
        if 'audio' in modality_inputs1 and isinstance(modality_inputs1['audio'], torch.Tensor):
            modality_inputs1['audio'] = modality_inputs1['audio'].unsqueeze(1).to(algorithm.device)
        if 'rgb' in modality_inputs1 and isinstance(modality_inputs1['rgb'], torch.Tensor):
            modality_inputs1['rgb'] = modality_inputs1['rgb'].squeeze(1)
        if 'flow' in modality_inputs1 and isinstance(modality_inputs1['flow'], torch.Tensor):
            modality_inputs1['flow'] = modality_inputs1['flow'].squeeze(1)
        labels1 = labels1.to(algorithm.device)

        # Training step - pass both domains
        loss_dict, predictions = algorithm.train_step_two_domains(
            modality_inputs0, labels0, modality_inputs1, labels1
        )

        # Accumulate GBL per-modality losses
        if algorithm.use_gblend and 'modality_losses' in loss_dict:
            for modality, loss_value in loss_dict['modality_losses'].items():
                gbl_losses[modality] += loss_value

        # Accumulate losses
        for key, value in loss_dict.items():
            if isinstance(value, dict):  # Skip nested dicts
                continue
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value

        # Compute accuracy
        with torch.no_grad():
            labels_combined = torch.cat([labels0, labels1], dim=0)
            _, predicted = predictions.max(1)
            total += labels_combined.size(0)
            correct += predicted.eq(labels_combined).sum().item()

        # Update progress bar
        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
                'loss': loss_dict['total_loss'],
                'acc': 100. * correct / total
            })

        # Average losses
    avg_losses = {k: v / min_len for k, v in total_losses.items()}
    accuracy = 100. * correct / total

    # Average GBL losses
    if algorithm.use_gblend:
        for key in gbl_losses.keys():
            gbl_losses[key] /= min_len
        return avg_losses, accuracy, gbl_losses
    else:
        return avg_losses, accuracy


def validate(algorithm, dataloader):
    """Validate model on single domain"""
    algorithm.model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for modality_inputs, labels in tqdm.tqdm(dataloader, desc="Validating"):
            # Move to device
            modality_inputs = {
                k: v.to(algorithm.device) if isinstance(v, torch.Tensor) and k != 'audio' else v
                for k, v in modality_inputs.items()
            }

            if 'audio' in modality_inputs and isinstance(modality_inputs['audio'], torch.Tensor):
                modality_inputs['audio'] = modality_inputs['audio'].unsqueeze(1).to(algorithm.device)

            if 'rgb' in modality_inputs and isinstance(modality_inputs['rgb'], torch.Tensor):
                modality_inputs['rgb'] = modality_inputs['rgb'].squeeze(1)
            if 'flow' in modality_inputs and isinstance(modality_inputs['flow'], torch.Tensor):
                modality_inputs['flow'] = modality_inputs['flow'].squeeze(1)

            labels = labels.to(algorithm.device)

            # Predict
            predictions = algorithm.predict(modality_inputs)
            loss = nn.functional.cross_entropy(predictions, labels)

            total_loss += loss.item()
            _, predicted = predictions.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy



def validate_two_domains(algorithm, dataloader0, dataloader1):
    """
    Validate model on two domains simultaneously

    Returns:
        total_acc, total_count, gbl_losses
    """
    algorithm.model.eval()

    total_acc = 0
    total_count = 0

    # GBL loss accumulators
    gbl_losses = None
    if algorithm.use_gblend:
        gbl_losses = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses['fusion'] = 0.0

    loader0_iter = iter(dataloader0)
    loader1_iter = iter(dataloader1)
    min_len = min(len(dataloader0), len(dataloader1))

    with torch.no_grad():
        for _ in tqdm.tqdm(range(min_len), desc="Validating two domains"):
            try:
                modality_inputs0, labels0 = next(loader0_iter)
                modality_inputs1, labels1 = next(loader1_iter)
            except StopIteration:
                break

            # Process domain 0
            modality_inputs0 = {
                k: v.to(algorithm.device) if isinstance(v, torch.Tensor) and k != 'audio' else v
                for k, v in modality_inputs0.items()
            }
            if 'audio' in modality_inputs0 and isinstance(modality_inputs0['audio'], torch.Tensor):
                modality_inputs0['audio'] = modality_inputs0['audio'].unsqueeze(1).to(algorithm.device)
            if 'rgb' in modality_inputs0 and isinstance(modality_inputs0['rgb'], torch.Tensor):
                modality_inputs0['rgb'] = modality_inputs0['rgb'].squeeze(1)
            if 'flow' in modality_inputs0 and isinstance(modality_inputs0['flow'], torch.Tensor):
                modality_inputs0['flow'] = modality_inputs0['flow'].squeeze(1)
            labels0 = labels0.to(algorithm.device)

            # Process domain 1
            modality_inputs1 = {
                k: v.to(algorithm.device) if isinstance(v, torch.Tensor) and k != 'audio' else v
                for k, v in modality_inputs1.items()
            }
            if 'audio' in modality_inputs1 and isinstance(modality_inputs1['audio'], torch.Tensor):
                modality_inputs1['audio'] = modality_inputs1['audio'].unsqueeze(1).to(algorithm.device)
            if 'rgb' in modality_inputs1 and isinstance(modality_inputs1['rgb'], torch.Tensor):
                modality_inputs1['rgb'] = modality_inputs1['rgb'].squeeze(1)
            if 'flow' in modality_inputs1 and isinstance(modality_inputs1['flow'], torch.Tensor):
                modality_inputs1['flow'] = modality_inputs1['flow'].squeeze(1)
            labels1 = labels1.to(algorithm.device)

            # Compute all losses using _compute_all_losses (reuse code)
            _, loss_dict, fusion_predictions = algorithm._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1, current_epoch=-1
            )

            # Accumulate GBL losses if enabled
            if algorithm.use_gblend and 'modality_losses' in loss_dict:
                for modality, loss_value in loss_dict['modality_losses'].items():
                    gbl_losses[modality] += loss_value

            # Compute accuracy
            labels_combined = torch.cat([labels0, labels1], dim=0)
            _, pred = torch.max(fusion_predictions.detach().cpu(), dim=1)
            acc = (pred == labels_combined.cpu()).sum().item()

            total_acc += int(acc)
            total_count += pred.size(0)

    # Average GBL losses
    if algorithm.use_gblend:
        for key in gbl_losses.keys():
            gbl_losses[key] /= min_len

    return total_acc, total_count, gbl_losses

def main():
    # Get configuration
    config = get_config()
    print_config(config)

    # Set random seed
    set_random_seed(config.seed)

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create output directories
    log_dir = Path(config.logging.log_dir)
    checkpoint_dir = Path(config.checkpointing.checkpoint_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_config(config, log_dir / 'config.yaml')

    # Create datasets - one per source domain
    print("\nCreating datasets...")
    source_domains = config.training.source_domains
    target_domain = config.training.target_domain

    # Create log file name based on modalities and target domain
    # Modality naming: rgb->v, flow->f, audio->a
    modality_map = {'rgb': 'v', 'flow': 'f', 'audio': 'a'}
    modality_str = ''.join([modality_map.get(m, m) for m in sorted(config.dataset.modalities)])
    # Target domain naming: D1->t1, D2->t2, D3->t3
    target_str = target_domain.lower().replace('d', 't')
    log_filename = f"{modality_str}_vit_{target_str}_log.txt"
    results_filename = f"{modality_str}_vit_{target_str}_results.txt"

    log_file_path = log_dir / log_filename
    results_file_path = log_dir / results_filename

    # Open log file for writing training progress
    log_file = open(log_file_path, 'w')
    log_file.write(f"ViT Option A: Three Separate ViTs\n")
    log_file.write(f"Log file: {log_file_path}\n")
    log_file.write(f"Results file: {results_file_path}\n")
    log_file.write(f"Modalities: {config.dataset.modalities}\n")
    log_file.write(f"Source domains: {source_domains}\n")
    log_file.write(f"Target domain: {target_domain}\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()

    print(f"Log file: {log_file_path}")
    print(f"Results file: {results_file_path}")
    print(f"Modalities: {config.dataset.modalities}")
    print(f"Source domains: {source_domains}")
    print(f"Target domain: {target_domain}")

    # Create separate datasets for each source domain (for training and validation)
    train_dataset0 = get_dataset(config, split='train', domains=[source_domains[0]])
    train_dataset1 = get_dataset(config, split='train', domains=[source_domains[1]])
    val_dataset0 = get_dataset(config, split='test', domains=[source_domains[0]])
    val_dataset1 = get_dataset(config, split='test', domains=[source_domains[1]])
    config.dataset['load_train_split'] = True
    test_dataset = get_dataset(config, split='test', domains=[target_domain])
    config.dataset['load_train_split'] = False
    print(f"Train dataset 0: {len(train_dataset0)} samples from {source_domains[0]}")
    print(f"Train dataset 1: {len(train_dataset1)} samples from {source_domains[1]}")
    print(f"Val dataset 0: {len(val_dataset0)} samples from {source_domains[0]}")
    print(f"Val dataset 1: {len(val_dataset1)} samples from {source_domains[1]}")
    print(f"Test dataset: {len(test_dataset)} samples from {target_domain}")

    # Create dataloaders - one per source domain
    collator = MultiModalCollator(config.dataset.modalities)

    train_loader0 = DataLoader(
        train_dataset0,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collator,
        pin_memory=False,
        drop_last=True
    )

    train_loader1 = DataLoader(
        train_dataset1,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collator,
        pin_memory=False,
        drop_last=True
    )

    val_loader0 = DataLoader(
        val_dataset0,
        batch_size=config.evaluation.get('batch_size', 32),
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
        pin_memory=False
    )

    val_loader1 = DataLoader(
        val_dataset1,
        batch_size=config.evaluation.get('batch_size', 32),
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.get('batch_size', 32),
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
        pin_memory=False
    )

    # Create model and algorithm

    use_videomae_v2 = config.model.get('use_videomae', False)

    if use_videomae_v2:
        print("\nCreating VideoMAEv2-based model and algorithm (方案 B: 原生 VideoMAEv2)...")
        model, feature_dims = create_base_model_videomae_v2(config)

    # Create LateFusionDG algorithm

    algorithm_config = {
        # Basic training parameters
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.get('weight_decay', 1e-5),
        'num_epochs': config.training.num_epochs,
        'min_lr': config.training.get('min_lr', 1e-6),

        # Optimizer configuration
        'optimizer_type': config.training.get('optimizer_type', 'sgd'),
        'sam_rho': config.training.get('sam_rho', 0.05),
        'sam_adaptive': config.training.get('sam_adaptive', False),
        'use_scheduler': config.training.get('use_scheduler', False),

        # ViT configuration (CRITICAL for parameter collection!)
        'vit_pretrained_path': config.model.get('vit_pretrained_path', None),
        'videomae_v2_pretrained': config.model.get('videomae_pretrained_path', None),
        'use_ast': config.model.get('use_ast', None),
        # Freeze params should come from training section (not model section)
        'freeze_vit_backbone': config.model.get('freeze_vit_backbone', False),
        'freeze_vit_layers': config.model.get('freeze_vit_layers', None),

        # Projection parameters
        'hidden_dim': config.algorithm.get('hidden_dim', 2048),
        'proj_dim': config.algorithm.get('proj_dim', 128),

        # Loss weights
        'alpha_contrast': config.algorithm.get('alpha_contrast', 0.15),
        'alpha_trans': config.algorithm.get('alpha_trans', 0.05),
        'temperature': config.algorithm.get('temperature', 0.1),

        # Modal gap methods
        'use_contras': config.algorithm.get('use_contras', True),
        'use_modtrans': config.algorithm.get('use_modtrans', True),

        # Fusion configuration
        'fusion_type': config.algorithm.get('fusion_type', 'bottleneck'),
        'fusion_hidden_dim': config.algorithm.get('fusion_hidden_dim', 512),
        'num_bottlenecks': config.algorithm.get('num_bottlenecks', 4),
        'dg_hidden_dim': config.algorithm.get('dg_hidden_dim', 256),

        # GBL configuration
        'use_gblend': config.algorithm.get('use_gblend', False),
        'gbl_momentum': config.algorithm.get('gbl_momentum', 0.5),

        # Domain gap mitigation methods
        'use_domain_contrastive': config.algorithm.get('use_domain_contrastive', False),
        'dg_contrastive_weight': config.algorithm.get('dg_contrastive_weight', 0.05),
        'dg_contrastive_temp': config.algorithm.get('dg_contrastive_temp', 0.1),
        'use_mmd': config.algorithm.get('use_mmd', False),
        'mmd_weight': config.algorithm.get('mmd_weight', 0.05),
        'mmd_kernel_num': config.algorithm.get('mmd_kernel_num', 5),
        'mmd_kernel_mul': config.algorithm.get('mmd_kernel_mul', 2.0),

        # Mixup configuration
        'use_mixup': config.algorithm.get('use_mixup', False),
        'mixup_alpha': config.algorithm.get('mixup_alpha', 1.0),
        'mixup_weight': config.algorithm.get('mixup_weight', 0.5),

        # MIRO configuration
        'use_miro': config.algorithm.get('use_miro', False),
        'miro_weight': config.algorithm.get('miro_weight', 1.0),
        'miro_var_init': config.algorithm.get('miro_var_init', 0.1),

        # LD weight (for modal gap)
        'ld': config.algorithm.get('ld', 0.05),

        # VideoMAEv2 optimizer configuration
        'use_videomae_optim': config.training.get('use_videomae_optim', True),
        'layer_decay': config.training.get('layer_decay', 0.75),
        'warmup_epochs': config.training.get('warmup_epochs', 5),
        'warmup_lr': config.training.get('warmup_lr', 1e-8)
    }

    algorithm = EarlyFusionDG(
        model=model,
        config=algorithm_config,
        modalities=config.dataset.modalities,
        feature_dims=feature_dims,
        num_classes=config.task.num_classes
    )

    print(f"Algorithm created with early fusion DG (ViT Option A)")
    print(f"Modalities: {config.dataset.modalities}")
    print(f"Feature dims: {feature_dims}")

    # Initialize step-level scheduler (VideoMAEv2 style)
    if algorithm.use_videomae_optim and algorithm.use_scheduler:
        from algorithms.early_fusion_dg_mae import cosine_scheduler
        import numpy as np

        # Calculate total batch size and learning rate scaling
        batch_size = config.training.batch_size
        total_batch_size = batch_size  # Single GPU training
        base_lr = config.training.learning_rate
        base_wd = config.training.get('weight_decay', 0.05)

        # Scale learning rate by batch size (VideoMAEv2: lr = lr * total_batch_size / 256)
        scaled_lr = base_lr * total_batch_size / 256
        scaled_min_lr = algorithm.min_lr * total_batch_size / 256
        scaled_warmup_lr = algorithm.warmup_lr * total_batch_size / 256

        print("\n" + "=" * 80)
        print("[LR Scaling] Applying batch size scaling (VideoMAEv2 style)")
        print(f"  Total batch size: {total_batch_size}")
        print(f"  Base LR: {base_lr:.6f} -> Scaled LR: {scaled_lr:.6f} (×{total_batch_size / 256:.3f})")
        print(f"  Min LR: {algorithm.min_lr:.6f} -> Scaled Min LR: {scaled_min_lr:.6f}")
        print(f"  Warmup LR: {algorithm.warmup_lr:.6f} -> Scaled Warmup LR: {scaled_warmup_lr:.6f}")
        print("=" * 80 + "\n")

        # Calculate number of training steps per epoch
        num_training_steps_per_epoch = min(len(train_loader0), len(train_loader1))
        total_epochs = config.training.num_epochs

        # Create LR schedule with warmup
        lr_schedule_values = cosine_scheduler(
            scaled_lr,
            scaled_min_lr,
            total_epochs,
            num_training_steps_per_epoch,
            warmup_epochs=algorithm.warmup_epochs,
            start_warmup_value=scaled_warmup_lr
        )

        # Create WD schedule (constant or cosine)
        wd_schedule_values = cosine_scheduler(
            base_wd,
            base_wd,  # Constant weight decay
            total_epochs,
            num_training_steps_per_epoch
        )

        # Store in algorithm
        algorithm.lr_schedule_values = lr_schedule_values
        algorithm.wd_schedule_values = wd_schedule_values

        print(f"[Scheduler] Step-level scheduler initialized:")
        print(f"  Total steps: {len(lr_schedule_values)}")
        print(f"  Steps per epoch: {num_training_steps_per_epoch}")
        print(f"  Warmup steps: {algorithm.warmup_epochs * num_training_steps_per_epoch}")
        print(f"  Peak LR: {scaled_lr:.6f}, Min LR: {scaled_min_lr:.6f}")
        print(f"  Weight decay: {base_wd:.6f}\n")

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("Starting training...\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0

    for epoch in range(config.training.num_epochs):
        epoch_msg = f"\nEpoch [{epoch + 1}/{config.training.num_epochs}]"
        print(epoch_msg)
        log_file.write(epoch_msg + "\n")

        # Train - pass both domain loaders
        train_result = train_epoch(algorithm, train_loader0, train_loader1, epoch, config)
        if algorithm.use_gblend:
            train_losses, train_acc, gbl_train_losses = train_result
        else:
            train_losses, train_acc = train_result
            gbl_train_losses = None

        train_msg = f"Train - Loss: {train_losses['total_loss']:.4f}, Acc: {train_acc:.2f}%"
        modal_losses_msg = f"  Modal losses: {', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items() if k != 'total_loss'])}"
        print(train_msg)
        print(modal_losses_msg)
        log_file.write(train_msg + "\n")
        log_file.write(modal_losses_msg + "\n")
        log_file.flush()

        # Validate
        if (epoch + 1) % config.training.val_frequency == 0:
            # Validate on two source domains
            val_result = validate_two_domains(algorithm, val_loader0, val_loader1)
            if algorithm.use_gblend:
                total_acc, total_count, gbl_val_losses = val_result
            else:
                total_acc, total_count = val_result[:2]
                gbl_val_losses = None

            current_val_acc = 100. * total_acc / total_count

            # Validate on test domain
            test_loss, test_acc = validate(algorithm, test_loader)

            val_msg = f"Val Overall - Acc: {current_val_acc:.2f}%"
            test_msg = f"Test (Target) - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%"

            print(val_msg)
            print(test_msg)

            log_file.write(val_msg + "\n")
            log_file.write(test_msg + "\n")
            log_file.flush()

            # Update GBL weights
            if algorithm.use_gblend:
                algorithm.update_loss_history(epoch, 'train', gbl_train_losses)
                algorithm.update_loss_history(epoch, 'val', gbl_val_losses)

                updated_weights = algorithm.update_gbl_weights(epoch)

                if updated_weights:
                    gbl_header = f"\n[EPOCH {epoch + 1}] GBL Weights Update (momentum={algorithm.gbl_momentum:.2f}):"
                    print(gbl_header)
                    log_file.write(gbl_header + "\n")

                    modality_names = sorted([m for m in updated_weights.keys() if m != 'fusion'])
                    weight_str = ', '.join([f'{m}={updated_weights[m]:.4f}' for m in modality_names])
                    weight_str += f', fusion={updated_weights["fusion"]:.4f}'
                    gbl_msg = f"  Weights: {weight_str}"
                    print(gbl_msg)
                    log_file.write(gbl_msg + "\n")
                    log_file.flush()

            # Save best model
            if current_val_acc >= best_val_acc:
                best_val_acc = current_val_acc
                best_test_acc = test_acc
                best_epoch = epoch + 1

                if config.checkpointing.save_best:
                    checkpoint_path = checkpoint_dir / 'best_model.pth'
                    algorithm.save_checkpoint(str(checkpoint_path))
                    best_msg = f"*** New best model! Val acc: {current_val_acc:.2f}%, Test acc: {test_acc:.2f}% ***"
                    save_msg = f"Saved best model to {checkpoint_path}"
                    print(best_msg)
                    print(save_msg)
                    log_file.write(best_msg + "\n")
                    log_file.write(save_msg + "\n")
                    log_file.flush()

        # Save checkpoint
        if (epoch + 1) % config.training.save_frequency == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            algorithm.save_checkpoint(str(checkpoint_path))

        # Save final model
    if config.checkpointing.save_last:
        checkpoint_path = checkpoint_dir / 'last_model.pth'
        algorithm.save_checkpoint(str(checkpoint_path))
        final_save_msg = f"\nSaved final model to {checkpoint_path}"
        print(final_save_msg)
        log_file.write(final_save_msg + "\n")

        # Print final results
    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Corresponding test accuracy: {best_test_acc:.2f}%")
    print("=" * 80)

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("Training completed!\n")
    log_file.write(f"Best epoch: {best_epoch}\n")
    log_file.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
    log_file.write(f"Corresponding test accuracy: {best_test_acc:.2f}%\n")
    log_file.write("=" * 80 + "\n")
    log_file.close()

    # Write results file
    with open(results_file_path, 'w') as results_file:
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Experiment: {modality_str} modalities, Target: {target_str}\n")
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Modalities: {config.dataset.modalities}\n")
        results_file.write(f"Source domains: {source_domains}\n")
        results_file.write(f"Target domain: {target_domain}\n")
        results_file.write("\n")
        results_file.write("Training Configuration:\n")
        results_file.write(f"  Epochs: {config.training.num_epochs}\n")
        results_file.write(f"  Batch size: {config.training.batch_size}\n")
        results_file.write(f"  Learning rate: {config.training.learning_rate}\n")
        results_file.write(f"  Optimizer: {algorithm.optimizer_type}\n")
        results_file.write(f"  Use GBL: {algorithm.use_gblend}\n")
        results_file.write(f"  Fusion type: {config.algorithm.fusion_type}\n")
        results_file.write(f"  Alpha contrast: {config.algorithm.alpha_contrast}\n")
        results_file.write(f"  Alpha trans: {config.algorithm.alpha_trans}\n")
        results_file.write(f"  ld (modal gap weight): {config.algorithm.ld}\n")
        results_file.write("\n")
        results_file.write("=" * 80 + "\n")
        results_file.write("FINAL RESULTS\n")
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Best epoch: {best_epoch}\n")
        results_file.write(f"Best validation accuracy: {best_val_acc:.2f}%\n")
        results_file.write(f"Corresponding test accuracy: {best_test_acc:.2f}%\n")
        results_file.write("=" * 80 + "\n")

    print(f"\nResults saved to: {results_file_path}")
    print(f"Training log saved to: {log_file_path}")


if __name__ == '__main__':
    main()
