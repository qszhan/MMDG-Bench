import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
from datetime import datetime

# Add parent directory to path
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

from datasets.face_antispoofing import get_fas_dataset_raw
from common_utils.config import get_config, print_config, save_config
from datasets.base import MultiModalCollator
from algorithms.late_fusion_dg import LateFusionDG
from models.fas import FaceAntiSpoofingModel
from evaluation.fas_metrics import evaluate_fas, print_fas_metrics


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def create_fas_model(config):
    """
    Create FAS model with ViT backbones (backbones only for late fusion)

    Returns:
        model: FaceAntiSpoofingModel instance (backbones only)
        feature_dims: Dict of feature dimensions per modality
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_dims = {mod: config.dataset.vit_feature_dim for mod in config.dataset.modalities}
    model = FaceAntiSpoofingModel(
        modalities=config.dataset.modalities,
        num_classes=config.task.num_classes,
        model_config=config.model,
        task_name=config.task.name,
        use_pretrained_features=config.dataset.use_extracted_features,
        backbones_only = True
    )

    model = model.to(device)

    print(f"\nFAS Model created:")
    print(f"  Modalities: {config.dataset.modalities}")
    print(f"  Feature dims: {feature_dims}")
    print(f"  Use pre-extracted features: {config.dataset.use_extracted_features}")
    print(f"  Device: {device}\n")

    return model, feature_dims


def train_epoch(algorithm, dataloader0, dataloader1, epoch, config):
    """
    Train for one epoch with two domain dataloaders

    Late Fusion: Each domain fuses modalities first, then DG is applied
    """
    algorithm.model.train()
    total_losses = {}
 
    gbl_losses_d0 = None
    gbl_losses_d1 = None
    if algorithm.use_gblend:
         
        gbl_losses_d0 = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses_d0['fusion'] = 0.0
        gbl_losses_d1 = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses_d1['fusion'] = 0.0

     
    all_preds = []
    all_labels = []

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

        
        device = algorithm.device
        modality_inputs0 = {k: v.to(device) for k, v in modality_inputs0.items()}
        modality_inputs1 = {k: v.to(device) for k, v in modality_inputs1.items()}
        labels0 = labels0.to(device)
        labels1 = labels1.to(device)

         
        if labels0.dim() > 1:
            labels0 = labels0.squeeze()
        if labels1.dim() > 1:
            labels1 = labels1.squeeze()

        
        loss_dict, fusion_pred0, fusion_pred1, dg_predictions = algorithm.train_step_two_domains(
            modality_inputs0, labels0, modality_inputs1, labels1
        )

        
        if algorithm.use_gblend and 'domain0_modality_losses' in loss_dict:
            for modality_key, loss_value in loss_dict['domain0_modality_losses'].items():
                
                modality = modality_key.replace('_loss', '')
                gbl_losses_d0[modality] += loss_value if not isinstance(loss_value, torch.Tensor) else loss_value.item()
            for modality_key, loss_value in loss_dict['domain1_modality_losses'].items():
               
                modality = modality_key.replace('_loss', '')
                gbl_losses_d1[modality] += loss_value if not isinstance(loss_value, torch.Tensor) else loss_value.item()

        
        for key, value in loss_dict.items():
             
            if isinstance(value, dict):
                continue
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value

        
        with torch.no_grad():
            labels_combined = torch.cat([labels0, labels1], dim=0)
            probs = torch.softmax(dg_predictions, dim=1)[:, 1]  # Spoof probability
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels_combined.cpu().numpy().tolist())

         
        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
                'loss': loss_dict['total_loss'],
            })

  
    avg_losses = {k: v / min_len for k, v in total_losses.items()}

     
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    from evaluation.fas_metrics import evaluate_fas
    all_metrics = evaluate_fas(all_labels, all_preds)
     
    metrics = {
        'hter': all_metrics['hter'],
        'auc': all_metrics['auc']
    }

     
    if algorithm.use_gblend:
        for key in gbl_losses_d0.keys():
            gbl_losses_d0[key] /= min_len
            gbl_losses_d1[key] /= min_len
        return avg_losses, metrics, gbl_losses_d0, gbl_losses_d1
    else:
        return avg_losses, metrics


def validate_two_domains(algorithm, dataloader0, dataloader1):
    """
    Validate model on two source domains simultaneously
    Used for GBL weight updates during training

    Returns:
        metrics, gbl_losses_d0, gbl_losses_d1
    """
    algorithm.model.eval()

    gbl_losses_d0 = None
    gbl_losses_d1 = None
    if algorithm.use_gblend:
        gbl_losses_d0 = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses_d0['fusion'] = 0.0
        gbl_losses_d1 = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses_d1['fusion'] = 0.0

    all_preds = []
    all_labels = []

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
 
            device = algorithm.device
            modality_inputs0 = {k: v.to(device) for k, v in modality_inputs0.items()}
            modality_inputs1 = {k: v.to(device) for k, v in modality_inputs1.items()}
            labels0 = labels0.to(device)
            labels1 = labels1.to(device)
 
            if labels0.dim() > 1:
                labels0 = labels0.squeeze()
            if labels1.dim() > 1:
                labels1 = labels1.squeeze()

            
            _, loss_dict, fusion_pred0, fusion_pred1, dg_predictions = algorithm._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1
            )

             
            if algorithm.use_gblend and 'domain0_modality_losses' in loss_dict:
                for modality_key, loss_value in loss_dict['domain0_modality_losses'].items():
                    modality = modality_key.replace('_loss', '')
                    gbl_losses_d0[modality] += loss_value if not isinstance(loss_value, torch.Tensor) else loss_value.item()
                for modality_key, loss_value in loss_dict['domain1_modality_losses'].items():
                    modality = modality_key.replace('_loss', '')
                    gbl_losses_d1[modality] += loss_value if not isinstance(loss_value, torch.Tensor) else loss_value.item()

            labels_combined = torch.cat([labels0, labels1], dim=0)
            probs = torch.softmax(dg_predictions, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels_combined.cpu().numpy().tolist())

    if algorithm.use_gblend:
        for key in gbl_losses_d0.keys():
            gbl_losses_d0[key] /= min_len
            gbl_losses_d1[key] /= min_len

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = evaluate_fas(all_labels, all_preds)
   
    metrics = {
        'hter': metrics['hter'],
        'auc': metrics['auc']
    }

    if algorithm.use_gblend:
        return metrics, gbl_losses_d0, gbl_losses_d1
    else:
        return metrics


def validate(algorithm, dataloader):
    """
    Validate on target domain

    Returns:
        metrics: Dict with HTER, AUC, etc.
    """
    algorithm.model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for modality_inputs, labels in tqdm.tqdm(dataloader, desc="Validating target domain"):
            device = algorithm.device
            modality_inputs = {k: v.to(device) for k, v in modality_inputs.items()}
            labels = labels.to(device)

            if labels.dim() > 1:
                labels = labels.squeeze()

            predictions = algorithm.predict(modality_inputs, labels)
            probs = torch.softmax(predictions, dim=1)[:, 1]

            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = evaluate_fas(all_labels, all_preds)
    metrics = {
        'hter': metrics['hter'],
        'auc': metrics['auc']
    }

    return metrics


def main():
    
    config = get_config()

     
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.checkpointing.checkpoint_dir, exist_ok=True)

     
    modality_map = {'rgb': 'r', 'depth': 'd', 'ir': 'i'}
    modality_str = ''.join([modality_map.get(m, m[0]) for m in sorted(config.dataset.modalities)])
    source_str = '+'.join(config.training.source_domains)
    target_str = config.training.target_domain

    log_filename = f"{modality_str}_{source_str}_to_{target_str}_log.txt"
    results_filename = f"{modality_str}_{source_str}_to_{target_str}_results.txt"

    log_file_path = os.path.join(config.logging.log_dir, log_filename)
    results_file_path = os.path.join(config.logging.log_dir, results_filename)

  
    log_file = open(log_file_path, 'w')
    log_file.write(f"Log file: {log_file_path}\n")
    log_file.write(f"Results file: {results_file_path}\n")
    log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Modalities: {config.dataset.modalities}\n")
    log_file.write(f"Source domains: {config.training.source_domains}\n")
    log_file.write(f"Target domain: {config.training.target_domain}\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()

    print(f"Log file: {log_file_path}")
    print(f"Results file: {results_file_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modalities: {config.dataset.modalities}")
    print(f"Source domains: {config.training.source_domains}")
    print(f"Target domain: {config.training.target_domain}")
    print_config(config)
    set_random_seed(config.seed)
    save_config(config, os.path.join(config.logging.log_dir, 'config.yaml'))
    print("\n" + "=" * 60)
    print("Creating FAS Model")
    print("=" * 60)
    model, feature_dims = create_fas_model(config)
    print("\n" + "=" * 60)
    print("Creating Late Fusion DG Algorithm")
    print("=" * 60)
    algorithm_config = {
        'learning_rate': config.training.learning_rate,
        'weight_decay': config.training.get('weight_decay', 1e-4),
        'num_epochs': config.training.num_epochs,
        'min_lr': config.training.get('min_lr', 1e-6),
        'optimizer_type': config.training.get('optimizer_type', 'sam'),
        'sam_rho': config.training.get('sam_rho', 0.05),
        'sam_adaptive': config.training.get('sam_adaptive', False),
        'use_scheduler': config.training.get('use_scheduler', True),
        'vit_pretrained_path': config.model.get('load_pretrained_path', None),
        'freeze_vit_backbone': config.model.get('freeze_vit_backbone', False),
        'freeze_vit_layers': config.model.get('freeze_vit_layers', None),
        'hidden_dim': config.algorithm.get('hidden_dim', 2048),
        'proj_dim': config.algorithm.get('proj_dim', 128),
        'alpha_contrast': config.algorithm.get('alpha_contrast', 3.0),
        'alpha_trans': config.algorithm.get('alpha_trans', 1.0),
        'ld': config.algorithm.get('ld', 0.05),
        'temperature': config.algorithm.get('temperature', 0.1),
        'use_contras': config.algorithm.get('use_contras', True),
        'use_modtrans': config.algorithm.get('use_modtrans', True),
        'use_domain_contrastive': config.algorithm.get('use_domain_contrastive', False),
        'dg_contrastive_weight': config.algorithm.get('dg_contrastive_weight', 0.05),
        'dg_contrastive_temp': config.algorithm.get('dg_contrastive_temp', 0.1),
        'use_mmd': config.algorithm.get('use_mmd', False),
        'mmd_weight': config.algorithm.get('mmd_weight', 0.05),
        'mmd_kernel_num': config.algorithm.get('mmd_kernel_num', 5),
        'mmd_kernel_mul': config.algorithm.get('mmd_kernel_mul', 2.0),
        'use_mixup': config.algorithm.get('use_mixup', False),
        'mixup_alpha': config.algorithm.get('mixup_alpha', 1.0),
        'mixup_weight': config.algorithm.get('mixup_weight', 0.5),
        'use_miro': config.algorithm.get('use_miro', False),
        'miro_weight': config.algorithm.get('miro_weight', 1.0),
        'miro_var_init': config.algorithm.get('miro_var_init', 0.1),
        'fusion_type': config.algorithm.get('fusion_type', 'bottleneck'),
        'fusion_hidden_dim': config.algorithm.get('fusion_hidden_dim', 512),
        'num_bottlenecks': config.algorithm.get('num_bottlenecks', 4),
        'dg_hidden_dim': config.algorithm.get('dg_hidden_dim', 256),
        'use_gblend': config.algorithm.get('use_gblend', True),
        'gbl_momentum': config.algorithm.get('gbl_momentum', 0.5),
    }


    algorithm = LateFusionDG(
        model=model,   
        config=algorithm_config,
        modalities=config.dataset.modalities,
        feature_dims=feature_dims,
        num_classes=config.task.num_classes
    )

   
    print("\n" + "=" * 60)
    print("Loading Datasets")
    print("=" * 60)
    print(f"Source domains: {config.training.source_domains}")
    print(f"Target domain: {config.training.target_domain}")
    print(f"Data root: {config.dataset.data_root}")
    print(f"Use extracted features: {config.dataset.use_extracted_features}")

  
    train_datasets = []
    for domain in config.training.source_domains:
        dataset = get_fas_dataset_raw(
            split='train',
            domains=[domain],
            modalities=config.dataset.modalities,
            data_root=config.dataset.data_root,
            use_extracted_features=config.dataset.use_extracted_features,
            source=True
        )
        train_datasets.append(dataset)

    
    val_datasets_source = []
    for domain in config.training.source_domains:
        dataset = get_fas_dataset_raw(
            split='val',
            domains=[domain],
            modalities=config.dataset.modalities,
            data_root=config.dataset.data_root,
            use_extracted_features=config.dataset.use_extracted_features,
            source=True  # Source domain validation
        )
        val_datasets_source.append(dataset)

     
    val_dataset_target = get_fas_dataset_raw(
        split='test',
        domains=[config.training.target_domain],
        modalities=config.dataset.modalities,
        data_root=config.dataset.data_root,
        use_extracted_features=config.dataset.use_extracted_features,
        source=False   
    )

    
    dataset_info = []
    dataset_info.append(f"Train datasets:")
    for i, (domain, ds) in enumerate(zip(config.training.source_domains, train_datasets)):
        info = f"  Domain {domain}: {len(ds)} samples"
        dataset_info.append(info)
        print(info)
    dataset_info.append(f"Val datasets (source):")
    for i, (domain, ds) in enumerate(zip(config.training.source_domains, val_datasets_source)):
        info = f"  Domain {domain}: {len(ds)} samples"
        dataset_info.append(info)
        print(info)
    dataset_info.append(f"Test dataset (target {config.training.target_domain}): {len(val_dataset_target)} samples")
    print(dataset_info[-1])

     
    for line in dataset_info:
        log_file.write(line + "\n")
    log_file.flush()

     
    if config.dataset.use_extracted_features:
        collator = MultiModalCollator(config.dataset.modalities)
        collate_fn = collator
    else:
        collate_fn = None   

    train_loaders = [
        DataLoader(
            ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            drop_last=True
        )
        for ds in train_datasets
    ]

    val_loaders_source = [
        DataLoader(
            ds,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn
        )
        for ds in val_datasets_source
    ]

    val_loader_target = DataLoader(
        val_dataset_target,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    
    print("\n" + "=" * 80)
    print("Starting Training (Late Fusion DG)")
    print("=" * 80)
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("Starting Training (Late Fusion DG)\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()

    best_val_hter = float('inf')
    best_test_hter = float('inf')
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_epoch = 0

    for epoch in range(config.training.num_epochs):
        epoch_msg = f"\nEpoch [{epoch + 1}/{config.training.num_epochs}]"
        print(epoch_msg)
        log_file.write(epoch_msg + "\n")

         
        train_result = train_epoch(algorithm, train_loaders[0], train_loaders[1], epoch, config)
        if algorithm.use_gblend:
            train_losses, train_metrics, gbl_train_losses_d0, gbl_train_losses_d1 = train_result
        else:
            train_losses, train_metrics = train_result
            gbl_train_losses_d0 = None
            gbl_train_losses_d1 = None

        train_msg = f"Train - Loss: {train_losses['total_loss']:.4f}, HTER: {train_metrics['hter']:.4f}, AUC: {train_metrics['auc']:.4f}"
        modal_losses_msg = f"  Modal losses: {', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items() if k != 'total_loss'])}"
        print(train_msg)
        print(modal_losses_msg)
        log_file.write(train_msg + "\n")
        log_file.write(modal_losses_msg + "\n")
        log_file.flush()

 
        if (epoch + 1) % config.training.val_frequency == 0:
            
            val_result = validate_two_domains(algorithm, val_loaders_source[0], val_loaders_source[1])
            if algorithm.use_gblend:
                val_metrics, gbl_val_losses_d0, gbl_val_losses_d1 = val_result
            else:
                val_metrics = val_result

            
            test_metrics = validate(algorithm, val_loader_target)

            val_msg = f"Val (Source) - HTER: {val_metrics['hter']:.4f}, AUC: {val_metrics['auc']:.4f}"
            test_msg = f"Test (Target) - HTER: {test_metrics['hter']:.4f}, AUC: {test_metrics['auc']:.4f}"

            print(val_msg)
            print(test_msg)

            log_file.write(val_msg + "\n")
            log_file.write(test_msg + "\n")
            log_file.flush()

           
            if algorithm.use_gblend:
                algorithm.update_loss_history(epoch, 'train', 'D0', gbl_train_losses_d0)
                algorithm.update_loss_history(epoch, 'train', 'D1', gbl_train_losses_d1)
                algorithm.update_loss_history(epoch, 'val', 'D0', gbl_val_losses_d0)
                algorithm.update_loss_history(epoch, 'val', 'D1', gbl_val_losses_d1)

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

             
            if val_metrics['hter'] <= best_val_hter:
                best_val_hter = val_metrics['hter']
                best_val_auc = val_metrics['auc']
                best_test_hter = test_metrics['hter']
                best_test_auc = test_metrics['auc']
                best_epoch = epoch + 1

                if config.checkpointing.save_best:
                    checkpoint_path = os.path.join(config.checkpointing.checkpoint_dir, 'best_model.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': algorithm.model.state_dict(),
                        'optimizer_state_dict': algorithm.optimizer.state_dict(),
                        'val_hter': val_metrics['hter'],
                        'val_auc': val_metrics['auc'],
                        'test_hter': test_metrics['hter'],
                        'test_auc': test_metrics['auc'],
                    }, checkpoint_path)
                    best_msg = f"*** New best model! Val HTER: {val_metrics['hter']:.4f}, Test HTER: {test_metrics['hter']:.4f} ***"
                    save_msg = f"Saved best model to {checkpoint_path}"
                    print(best_msg)
                    print(save_msg)
                    log_file.write(best_msg + "\n")
                    log_file.write(save_msg + "\n")
                    log_file.flush()

    
        if (epoch + 1) % config.training.save_frequency == 0:
            checkpoint_path = os.path.join(config.checkpointing.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': algorithm.model.state_dict(),
                'optimizer_state_dict': algorithm.optimizer.state_dict(),
            }, checkpoint_path)

  
    if config.checkpointing.save_last:
        final_path = os.path.join(config.checkpointing.checkpoint_dir, 'last_model.pth')
        torch.save({
            'epoch': config.training.num_epochs,
            'model_state_dict': algorithm.model.state_dict(),
            'optimizer_state_dict': algorithm.optimizer.state_dict(),
        }, final_path)
        final_save_msg = f"\nSaved final model to {final_path}"
        print(final_save_msg)
        log_file.write(final_save_msg + "\n")

 
    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation HTER: {best_val_hter:.4f}, AUC: {best_val_auc:.4f}")
    print(f"Corresponding test HTER: {best_test_hter:.4f}, AUC: {best_test_auc:.4f}")
    print("=" * 80)

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("Training completed!\n")
    log_file.write(f"Best epoch: {best_epoch}\n")
    log_file.write(f"Best validation HTER: {best_val_hter:.4f}, AUC: {best_val_auc:.4f}\n")
    log_file.write(f"Corresponding test HTER: {best_test_hter:.4f}, AUC: {best_test_auc:.4f}\n")
    log_file.write("=" * 80 + "\n")
    log_file.close()

 
    with open(results_file_path, 'w') as results_file:
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Experiment: {modality_str} modalities, Target: {target_str} (Late Fusion)\n")
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Modalities: {config.dataset.modalities}\n")
        results_file.write(f"Source domains: {config.training.source_domains}\n")
        results_file.write(f"Target domain: {config.training.target_domain}\n")
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
        results_file.write(f"Best validation HTER: {best_val_hter:.4f}\n")
        results_file.write(f"Best validation AUC: {best_val_auc:.4f}\n")
        results_file.write(f"Corresponding test HTER: {best_test_hter:.4f}\n")
        results_file.write(f"Corresponding test AUC: {best_test_auc:.4f}\n")
        results_file.write("=" * 80 + "\n")

    print(f"\nResults saved to: {results_file_path}")
    print(f"Training log saved to: {log_file_path}")


if __name__ == '__main__':
    main()