 

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
from datasets.action_recognition import  HACDataset, EPICKitchensDataset
from common_utils.config import get_config, print_config, save_config
from datasets.base import MultiModalCollator
from algorithms.late_fusion_dg import LateFusionDG
# from algorithms.late_fusion_dg_freeze import LateFusionDG

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def get_dataset(config, split, domains):
    """
    Create dataset based on config.dataset.name

    Supports:
    - 'epic_kitchens': EPIC-Kitchens dataset
    - 'hac': HAC (Human-Animal-Computer) dataset
    """


 
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

def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def unfreeze_audio_layers(audio_backbone, audio_head, n_layers, modality_name="Audio"):
    """
    n_layers:
      0 : freeze all
      1 : unfreeze AudioAttGenModule.layer4 + fc
      2 : + audio_backbone.layer4 (conv5_x)
      3 : + audio_backbone.layer3 (conv4_x)
      4 : + audio_backbone.layer2 (conv3_x)
      5 : + audio_backbone.layer1 (conv2_x)
     -1 : unfreeze all
    """
    audio_backbone = _unwrap(audio_backbone) if audio_backbone is not None else None
    audio_head     = _unwrap(audio_head)     if audio_head is not None     else None

 
    if audio_backbone is not None:
        for p in audio_backbone.parameters(): p.requires_grad = False
    if audio_head is not None:
        for p in audio_head.parameters(): p.requires_grad = False

    if n_layers == 0:
        pass
    elif n_layers == -1:
        if audio_backbone is not None:
            for p in audio_backbone.parameters(): p.requires_grad = True
        if audio_head is not None:
            for p in audio_head.parameters(): p.requires_grad = True
    else:
         
        if audio_head is not None:
            # head.layer4
            if hasattr(audio_head, "layer4"):
                for p in audio_head.layer4.parameters(): p.requires_grad = True
            # head.fc / fc_
            if hasattr(audio_head, "fc"):
                for p in audio_head.fc.parameters(): p.requires_grad = True
            if hasattr(audio_head, "fc_"):
                for p in audio_head.fc_.parameters(): p.requires_grad = True
            
            if hasattr(audio_head, "avgpool") and hasattr(audio_head.avgpool, "parameters"):
                
                pass

         
        if audio_backbone is not None and n_layers > 1:
            order = ["layer4", "layer3", "layer2", "layer1"]
            for i in range(min(n_layers-1, len(order))):
                name = order[i]
                if hasattr(audio_backbone, name):
                    layer = getattr(audio_backbone, name)
                    for p in layer.parameters(): p.requires_grad = True

     
    trainable, total = 0, 0
    if audio_backbone is not None:
        trainable += sum(p.numel() for p in audio_backbone.parameters() if p.requires_grad)
        total     += sum(p.numel() for p in audio_backbone.parameters())
    if audio_head is not None:
        trainable += sum(p.numel() for p in audio_head.parameters() if p.requires_grad)
        total     += sum(p.numel() for p in audio_head.parameters())

    print(f"{modality_name}: unfroze setting = {n_layers}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({(100*trainable/total if total else 0):.2f}%)")



def unfreeze_last_n_layers(model, n_layers, modality_name=''):
    """
    Unfreeze the last n layers of a model

    Args:
        model: Model to unfreeze
        n_layers: Number of layers to unfreeze
            -1: Unfreeze all layers
            0: Freeze all layers
            >0: Unfreeze last n layers
        modality_name: Modality name for logging
    """
    # First freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    if n_layers == -1:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{modality_name} backbone: unfroze ALL layers")
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    elif n_layers > 0:
        # Unfreeze last n layers
        # Get backbone (handle DataParallel wrapper)
        if hasattr(model, 'module'):
            backbone = model.module.backbone
            cls_head = model.module.cls_head
        else:
            backbone = model.backbone
            cls_head = model.cls_head

        # Always unfreeze the classification head
        for param in cls_head.parameters():
            param.requires_grad = True

        # For SlowFast network (RGB)
        if hasattr(backbone, 'slow_path') and hasattr(backbone, 'fast_path'):
            # SlowFast has res2, res3, res4, res5 in each path
            # Unfreeze from the end: res5, res4, res3, res2
            layer_names = ['layer4', 'layer3', 'layer2', 'layer1']

            for i in range(min(n_layers, len(layer_names))):
                layer_name = layer_names[i]
                # Unfreeze in slow path
                if hasattr(backbone.slow_path, layer_name):
                    layer = getattr(backbone.slow_path, layer_name)
                    for param in layer.parameters():
                        param.requires_grad = True
                # Unfreeze in fast path
                if hasattr(backbone.fast_path, layer_name):
                    layer = getattr(backbone.fast_path, layer_name)
                    for param in layer.parameters():
                        param.requires_grad = True

        # For ResNet network (Flow)
        elif hasattr(backbone, 'layer4'):
            # ResNet has layer1, layer2, layer3, layer4
            layer_names = ['layer4', 'layer3', 'layer2', 'layer1']
            for i in range(min(n_layers, len(layer_names))):
                layer_name = layer_names[i]
                layer = getattr(backbone, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True

        # For other architectures, unfreeze from the end
        else:
            all_params = list(model.parameters())
            # Estimate parameters per layer (rough approximation)
            n_params_to_unfreeze = len(all_params) * n_layers // 10
            for param in all_params[-n_params_to_unfreeze:]:
                param.requires_grad = True

        # Statistics
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{modality_name} backbone: unfroze last {n_layers} layer(s)")
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    else:
        # n_layers == 0, keep all frozen
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{modality_name} backbone: all layers frozen")
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")


def create_base_model(config):
    """
    Create base model with backbones only (no fusion head)
    LateFusionDG algorithm will add its own components
    """
    from mmaction.apis import init_recognizer
    from VGGSound.model import AVENet
    from VGGSound.models.resnet import AudioAttGenModule
    from VGGSound.test import get_arguments

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get unfreeze configuration
    unfreeze_config = config.algorithm.get('unfreeze_layers', {})
    unfreeze_backbone = config.algorithm.get('unfreeze_backbone', False)

    print("\n" + "=" * 80)
    print("Backbone Freezing Configuration:")
    print(f"  Unfreeze backbone: {unfreeze_backbone}")
    if unfreeze_backbone:
        print(f"  Unfreeze layers: {unfreeze_config}")
    print("=" * 80 + "\n")

    # Store backbone references
    backbones = {}
    feature_dims = {}

    # Video backbone
    if 'rgb' in config.dataset.modalities:
        # Get config path from config file (relative to project root)
        config_file = _project_root / config.dataset.cfg_video_path
        # Checkpoint path (resolve relative to project root if provided)
        checkpoint_file = str(_project_root / config.dataset.checkpoint_video) if config.dataset.get(
            'checkpoint_video') else None

        model_video = init_recognizer(str(config_file), checkpoint_file, device=device, use_frames=True)
        model_video.cls_head.fc_cls = nn.Linear(2304, config.task.num_classes).cuda()
        model_video = torch.nn.DataParallel(model_video)

        # Apply freezing/unfreezing strategy
        if unfreeze_backbone and 'rgb' in unfreeze_config:
            unfreeze_last_n_layers(model_video, unfreeze_config['rgb'], 'RGB')
        else:
            # Default: freeze all layers
            for param in model_video.parameters():
                param.requires_grad = False
            print("RGB backbone: all layers frozen (default)")

        backbones['rgb'] = model_video
        feature_dims['rgb'] = 2304

    # Flow backbone
    if 'flow' in config.dataset.modalities:
        # Get config path from config file (relative to project root)
        config_file_flow = _project_root / config.dataset.cfg_flow_path
        # Checkpoint path (resolve relative to project root if provided)
        checkpoint_file_flow = str(_project_root / config.dataset.checkpoint_flow) if config.dataset.get(
            'checkpoint_flow') else None

        model_flow = init_recognizer(str(config_file_flow), checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, config.task.num_classes).cuda()
        model_flow = torch.nn.DataParallel(model_flow)

        # Apply freezing/unfreezing strategy
        if unfreeze_backbone and 'flow' in unfreeze_config:
            unfreeze_last_n_layers(model_flow, unfreeze_config['flow'], 'Flow')
        else:
            # Default: freeze all layers
            for param in model_flow.parameters():
                param.requires_grad = False
            print("Flow backbone: all layers frozen (default)")

        backbones['flow'] = model_flow
        feature_dims['flow'] = 2048

    # Audio backbone
    if 'audio' in config.dataset.modalities:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)

        audio_checkpoint_path = str(_project_root / config.dataset.checkpoint_audio) if config.dataset.get(
                'checkpoint_audio') else None
        if audio_checkpoint_path:
                checkpoint = torch.load(audio_checkpoint_path, map_location=torch.device("cpu"))
                audio_model.load_state_dict(checkpoint['model_state_dict'])

        audio_model = audio_model.cuda()

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Use num_classes from config instead of hardcoding to 8
        num_classes = config.task.num_classes
        audio_cls_model.fc = nn.Linear(512, num_classes)
        audio_cls_model = audio_cls_model.cuda()

        # Apply freezing/unfreezing strategy
        if unfreeze_backbone and 'audio' in unfreeze_config:

            unfreeze_audio_layers(audio_model, audio_cls_model, unfreeze_config['audio'], modality_name="Audio")

        else:
            # Default: freeze all layers
            for param in audio_model.parameters():
                    param.requires_grad = False
            for param in audio_cls_model.parameters():
                    param.requires_grad = False
            print("Audio backbone: all layers frozen (default)")

        # Always set audio_model to eval mode (feature extractor)
        audio_model.eval()

        backbones['audio'] = (audio_model, audio_cls_model)
        feature_dims['audio'] = 512

    # Create a wrapper model that holds all backbones
    class MultiModalBackboneWrapper(nn.Module):
        def __init__(self, backbones):
            super().__init__()
            self.backbones_dict = backbones

            # Register modules properly
            if 'rgb' in backbones:
                self.rgb_backbone = backbones['rgb']
            if 'flow' in backbones:
                self.flow_backbone = backbones['flow']
            if 'audio' in backbones:
                self.audio_backbone = backbones['audio'][0]
                self.audio_cls = backbones['audio'][1]

        def forward_modality(self, modality_name, modality_data):
            """Extract features from a specific modality

            Note: torch.no_grad() and .detach() are removed to allow gradient
            backpropagation to unfrozen layers. Frozen parameters will still not
            receive gradients due to requires_grad=False.
            """
            if modality_name == 'rgb':
                # Allow gradients to flow through unfrozen layers
                x_slow, x_fast = self.rgb_backbone.module.backbone.get_feature(modality_data)
                v_feat = (x_slow, x_fast)
                v_feat = self.rgb_backbone.module.backbone.get_predict(v_feat)
                predict, features = self.rgb_backbone.module.cls_head(v_feat)
                return predict, features

            elif modality_name == 'flow':
                # Allow gradients to flow through unfrozen layers
                f_feat = self.flow_backbone.module.backbone.get_feature(modality_data)
                f_feat = self.flow_backbone.module.backbone.get_predict(f_feat)
                predict, features = self.flow_backbone.module.cls_head(f_feat)
                return predict, features

            elif modality_name == 'audio':


                _, audio_feat, _ = self.audio_backbone(modality_data)
                # Allow gradients through audio_cls if unfrozen
                predict, features = self.audio_cls(audio_feat)
                return predict, features

            else:
                raise ValueError(f"Unknown modality: {modality_name}")

    wrapper_model = MultiModalBackboneWrapper(backbones)

    return wrapper_model, feature_dims


def train_epoch(algorithm, dataloader0, dataloader1, epoch, config):
    """Train for one epoch with two domain dataloaders"""
    algorithm.model.train()
    total_losses = {}
    correct = 0
    total = 0

    # Initialize GBL loss accumulators if GBL is enabled
    gbl_losses_d0 = None
    gbl_losses_d1 = None
    if algorithm.use_gblend:
        # Initialize accumulators for each modality + fusion
        gbl_losses_d0 = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses_d0['fusion'] = 0.0
        gbl_losses_d1 = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses_d1['fusion'] = 0.0

    # Create iterators for both domains
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
        loss_dict, fusion_pred0, fusion_pred1, dg_predictions = algorithm.train_step_two_domains(
            modality_inputs0, labels0, modality_inputs1, labels1
        )

        # Accumulate GBL per-modality losses if enabled
        if algorithm.use_gblend and 'domain0_modality_losses' in loss_dict:
            for modality_key, loss_value in loss_dict['domain0_modality_losses'].items():
                # Remove '_loss' suffix to match gbl_losses_d0 keys
                modality = modality_key.replace('_loss', '')
                gbl_losses_d0[modality] += loss_value if not isinstance(loss_value, torch.Tensor) else loss_value.item()
            for modality_key, loss_value in loss_dict['domain1_modality_losses'].items():
                # Remove '_loss' suffix to match gbl_losses_d1 keys
                modality = modality_key.replace('_loss', '')
                gbl_losses_d1[modality] += loss_value if not isinstance(loss_value, torch.Tensor) else loss_value.item()

        # Accumulate losses (skip nested dicts for GBL)

        for key, value in loss_dict.items():
            # Skip nested dictionaries (GBL per-modality losses)

            if isinstance(value, dict):
                continue
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value

        # Compute accuracy (using fusion predictions from both domains)
        with torch.no_grad():
            # predictions0 = algorithm.predict(modality_inputs0, labels0)
            # predictions1 = algorithm.predict(modality_inputs1, labels1)

            # _, predicted0 = predictions0.max(1)
            # _, predicted1 = predictions1.max(1)

            total += labels0.size(0) + labels1.size(0)
            # correct += predicted0.eq(labels0).sum().item() + predicted1.eq(labels1).sum().item()

            labels_combined = torch.cat([labels0, labels1], dim=0)
            _, dg_predictions = dg_predictions.max(1)
            correct += dg_predictions.eq(labels_combined).sum().item()

        # Update progress bar
        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
                'loss': loss_dict['total_loss'],
                'acc': 100. * correct / total
            })

    # Average losses
    avg_losses = {k: v / min_len for k, v in total_losses.items()}
    accuracy = 100. * correct / total

    # Average GBL losses if enabled
    if algorithm.use_gblend:
        for key in gbl_losses_d0.keys():
            gbl_losses_d0[key] /= min_len
            gbl_losses_d1[key] /= min_len
        return avg_losses, accuracy, gbl_losses_d0, gbl_losses_d1
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

            # Squeeze rgb/flow if needed (collator already extracted 'imgs' from MMAction2 dict)
            if 'rgb' in modality_inputs and isinstance(modality_inputs['rgb'], torch.Tensor):
                modality_inputs['rgb'] = modality_inputs['rgb'].squeeze(1)
            if 'flow' in modality_inputs and isinstance(modality_inputs['flow'], torch.Tensor):
                modality_inputs['flow'] = modality_inputs['flow'].squeeze(1)

            labels = labels.to(algorithm.device)

            # Predict
            predictions = algorithm.predict(modality_inputs, labels)
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
    Validate model on two domains simultaneously, matching train_late_EPIC_CTG_mlp.py logic

    Computes:
    - acc0: Domain 0 fusion predictions
    - acc1: Domain 1 fusion predictions
    - acc: Overall DG classifier predictions (used for best model selection)

    Returns:
        total_acc0, total_acc1, total_acc, total_d_count, total_count, gbl_losses_d0, gbl_losses_d1
    """
    algorithm.model.eval()

    total_acc0 = 0
    total_acc1 = 0
    total_acc = 0
    total_d_count = 0
    total_count = 0

    # Initialize GBL loss accumulators if GBL is enabled
    gbl_losses_d0 = None
    gbl_losses_d1 = None
    if algorithm.use_gblend:
        gbl_losses_d0 = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses_d0['fusion'] = 0.0
        gbl_losses_d1 = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses_d1['fusion'] = 0.0

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

             

            # Extract features and get DG prediction on combined domains
            modality_features0, _, val_losses_batch_d0, _ = algorithm.extract_features(modality_inputs0, labels0)

            modality_features1, _, val_losses_batch_d1, _ = algorithm.extract_features(modality_inputs1, labels1)

            # Get fused features and compute fusion losses for GBL
            feature_list0 = [modality_features0[mod] for mod in sorted(modality_features0.keys())]
            feature_list1 = [modality_features1[mod] for mod in sorted(modality_features1.keys())]
            fusion_predictions0, fused_feat0 = algorithm.fusion_module(feature_list0)
            fusion_predictions1, fused_feat1 = algorithm.fusion_module(feature_list1)

            # Compute fusion losses for GBL tracking
            if algorithm.use_gblend:
                fusion_loss0 = algorithm.criterion(fusion_predictions0, labels0)
                fusion_loss1 = algorithm.criterion(fusion_predictions1, labels1)
                val_losses_batch_d0['fusion'] = fusion_loss0
                val_losses_batch_d1['fusion'] = fusion_loss1

            if algorithm.use_gblend:

                for modality_key in val_losses_batch_d0.keys():
                    modality = modality_key.replace('_loss', '')
                    gbl_losses_d0[modality] += val_losses_batch_d0[modality_key]
                    gbl_losses_d1[modality] += val_losses_batch_d1[modality_key]

            # Get DG classifier predictions for each domain separately
            logits0 = algorithm.dg_classifier(fused_feat0)
            logits1 = algorithm.dg_classifier(fused_feat1)
            # Concatenate for combined DG classifier prediction
            fused_feat_combined = torch.cat([fused_feat0, fused_feat1], dim=0)
            logits_combined = algorithm.dg_classifier(fused_feat_combined)
            labels_combined = torch.cat([labels0, labels1], dim=0)

            # Compute predictions
            pred = torch.softmax(logits_combined, dim=1)
            _, pred0 = torch.max(logits0.detach().cpu(), dim=1)
            _, pred1 = torch.max(logits1.detach().cpu(), dim=1)
            _, pred = torch.max(pred.detach().cpu(), dim=1)

            # Compute accuracies
            acc0 = (pred0 == labels0.cpu()).sum().item()
            acc1 = (pred1 == labels1.cpu()).sum().item()
            acc = (pred == labels_combined.cpu()).sum().item()

            total_acc0 += int(acc0)
            total_acc1 += int(acc1)
            total_acc += int(acc)
            total_d_count += pred0.size(0)
            total_count += pred.size(0)

    # Average GBL losses if enabled

    if algorithm.use_gblend:
        for key in gbl_losses_d0.keys():
            gbl_losses_d0[key] /= min_len
            gbl_losses_d1[key] /= min_len

    return total_acc0, total_acc1, total_acc, total_d_count, total_count, gbl_losses_d0, gbl_losses_d1


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
    log_filename = f"{modality_str}_{target_str}_log.txt"
    results_filename = f"{modality_str}_{target_str}_results.txt"

    log_file_path = log_dir / log_filename
    results_file_path = log_dir / results_filename

    # Open log file for writing training progress
    log_file = open(log_file_path, 'w')
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
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )

    train_loader1 = DataLoader(
        train_dataset1,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True
    )

    val_loader0 = DataLoader(
        val_dataset0,
        batch_size=config.evaluation.get('batch_size', 32),
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True
    )

    val_loader1 = DataLoader(
        val_dataset1,
        batch_size=config.evaluation.get('batch_size', 32),
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.get('batch_size', 32),
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True
    )

    # Create model and algorithm
    print("\nCreating model and algorithm...")
    model, feature_dims = create_base_model(config)

    # Create LateFusionDG algorithm
    algorithm_config = {
        # Basic training parameters
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.get('weight_decay', 1e-5),

        # Optimizer configuration
        'optimizer_type': config.training.get('optimizer_type', 'sgd'),
        'sam_rho': config.training.get('sam_rho', 0.05),
        'sam_adaptive': config.training.get('sam_adaptive', False),
        'use_scheduler': config.training.get('use_scheduler', False),
        "backbone_lr_scale": config.algorithm.get('backbone_lr_scale', 0.1),

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
        'ld': config.algorithm.get('ld', 0.05)
    }

    algorithm = LateFusionDG(
        model=model,
        config=algorithm_config,
        modalities=config.dataset.modalities,
        feature_dims=feature_dims,
        num_classes=config.task.num_classes
    )

    print(f"Algorithm created with late fusion DG")
    print(f"Modalities: {config.dataset.modalities}")
    print(f"Feature dims: {feature_dims}")

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
            train_losses, train_acc, gbl_train_losses_d0, gbl_train_losses_d1 = train_result
        else:
            train_losses, train_acc = train_result
            gbl_train_losses_d0, gbl_train_losses_d1 = None, None

        train_msg = f"Train - Loss: {train_losses['total_loss']:.4f}, Acc: {train_acc:.2f}%"
        modal_losses_msg = f"  Modal losses: {', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items() if k != 'total_loss'])}"
        print(train_msg)
        print(modal_losses_msg)
        log_file.write(train_msg + "\n")
        log_file.write(modal_losses_msg + "\n")
        log_file.flush()

        # Validate on both source domains and test domain
        if (epoch + 1) % config.training.val_frequency == 0:
            # Validate on two source domains (matching original implementation)
            val_result = validate_two_domains(algorithm, val_loader0, val_loader1)
            if algorithm.use_gblend:
                total_acc0, total_acc1, total_acc, total_d_count, total_count, gbl_val_losses_d0, gbl_val_losses_d1 = val_result
            else:
                total_acc0, total_acc1, total_acc, total_d_count, total_count = val_result[:5]
                gbl_val_losses_d0, gbl_val_losses_d1 = None, None

            # Calculate accuracies
            val_acc0 = 100. * total_acc0 / total_d_count
            val_acc1 = 100. * total_acc1 / total_d_count
            current_val_acc = 100. * total_acc / total_count  # DG classifier accuracy

            # Validate on test domain (single domain)
            test_loss, test_acc = validate(algorithm, test_loader)

            val_msg0 = f"Val Domain 0 - Acc: {val_acc0:.2f}%"
            val_msg1 = f"Val Domain 1 - Acc: {val_acc1:.2f}%"
            val_msg_overall = f"Val Overall (DG Classifier) - Acc: {current_val_acc:.2f}%"
            test_msg = f"Test (Target) - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%"

            print(val_msg0)
            print(val_msg1)
            print(val_msg_overall)
            print(test_msg)

            log_file.write(val_msg0 + "\n")
            log_file.write(val_msg1 + "\n")
            log_file.write(val_msg_overall + "\n")
            log_file.write(test_msg + "\n")
            log_file.flush()

            # Update GBL weights if enabled and after first epoch
            # if algorithm.use_gblend and epoch > 0:
            if algorithm.use_gblend:
                # Update loss history

                algorithm.update_loss_history(epoch, 'train', 'D0', gbl_train_losses_d0)
                algorithm.update_loss_history(epoch, 'train', 'D1', gbl_train_losses_d1)
                algorithm.update_loss_history(epoch, 'val', 'D0', gbl_val_losses_d0)
                algorithm.update_loss_history(epoch, 'val', 'D1', gbl_val_losses_d1)

                # Update GBL weights
                updated_weights = algorithm.update_gbl_weights(epoch)

                if updated_weights:
                    gbl_header = f"\n[EPOCH {epoch + 1}] GBL Weights Update (momentum={algorithm.gbl_momentum:.2f}):"
                    print(gbl_header)
                    log_file.write(gbl_header + "\n")

                    for domain_name, domain_key in [('D0', 'D0'), ('D1', 'D1')]:
                        weights = updated_weights[domain_key]
                        modality_names = sorted([m for m in weights.keys() if m != 'fusion'])
                        weight_str = ', '.join([f'{m}={weights[m]:.4f}' for m in modality_names])
                        weight_str += f', fusion={weights["fusion"]:.4f}'
                        gbl_msg = f"  Domain {domain_name}: {weight_str}"
                        print(gbl_msg)
                        log_file.write(gbl_msg + "\n")
                    log_file.flush()

            # Save best model based on validation DG classifier accuracy
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

        # Update GBL loss history for first epoch (before validation)
        # elif algorithm.use_gblend and epoch == 0:
        #     algorithm.update_loss_history(epoch, 'train', 'D0', gbl_train_losses_d0)
        #     algorithm.update_loss_history(epoch, 'train', 'D1', gbl_train_losses_d1)

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

    # Print and log final results
    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy (DG Classifier): {best_val_acc:.2f}%")
    print(f"Corresponding test accuracy: {best_test_acc:.2f}%")
    print("=" * 80)

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("Training completed!\n")
    log_file.write(f"Best epoch: {best_epoch}\n")
    log_file.write(f"Best validation accuracy (DG Classifier): {best_val_acc:.2f}%\n")
    log_file.write(f"Corresponding test accuracy: {best_test_acc:.2f}%\n")
    log_file.write("=" * 80 + "\n")
    log_file.close()

    # Write results to separate results file
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
        results_file.write("\n")
        results_file.write("=" * 80 + "\n")
        results_file.write("FINAL RESULTS\n")
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Best epoch: {best_epoch}\n")
        results_file.write(f"Best validation accuracy (DG Classifier): {best_val_acc:.2f}%\n")
        results_file.write(f"Corresponding test accuracy: {best_test_acc:.2f}%\n")
        results_file.write("=" * 80 + "\n")

    print(f"\nResults saved to: {results_file_path}")
    print(f"Training log saved to: {log_file_path}")


if __name__ == '__main__':
    main()
