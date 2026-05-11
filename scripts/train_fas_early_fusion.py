import sys
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm
import argparse
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

from datasets.face_antispoofing import get_fas_dataset_raw
from datetime import datetime

from common_utils.config import get_config, print_config, save_config
from datasets.base import MultiModalCollator
from algorithms.early_fusion_dg import EarlyFusionDG
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
    Create FAS model with shared ViT backbone (backbones only for early fusion)

    Returns:
        model: FaceAntiSpoofingModel instance (backbones only, shared backbone)
        feature_dims: Dict of feature dimensions per modality
    """
    from models.fas import FaceAntiSpoofingModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dims = {mod: 768 for mod in config.dataset.modalities}
    if not hasattr(config.model, 'shared_backbone'):
        config.model['shared_backbone'] = True
    model = FaceAntiSpoofingModel(
        modalities=config.dataset.modalities,
        num_classes=config.task.num_classes,
        model_config=config.model,
        task_name=config.task.name,
        use_pretrained_features=config.dataset.use_extracted_features,
        backbones_only=True   
    )
    model = model.to(device)
    print(f"\nFAS Model created (shared ViT, backbones only):")
    print(f"  Modalities: {config.dataset.modalities}")
    print(f"  Feature dims: {feature_dims}")
    print(f"  Shared ViT: True (1 ViT for all modalities)")
    print(f"  Use pre-extracted features: {config.dataset.use_extracted_features}")
    print(f"  Device: {device}\n")
    return model, feature_dims



def train_epoch(algorithm, dataloader0, dataloader1, epoch, config):
    """Train for one epoch with two domain dataloaders (supports both data formats)"""
    algorithm.model.train()
    total_losses = {}
    all_preds = []
    all_labels = []
    gbl_losses = None
    if algorithm.use_gblend:
        gbl_losses = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses['fusion'] = 0.0

    # Create iterators
    loader0_iter = iter(dataloader0)
    loader1_iter = iter(dataloader1)
    min_len = min(len(dataloader0), len(dataloader1))

    pbar = tqdm.tqdm(range(min_len), desc=f"Epoch {epoch + 1}")

     
    use_extracted_features = config.dataset.use_extracted_features
    for batch_idx in pbar:
        try:
            batch0 = next(loader0_iter)
            batch1 = next(loader1_iter)
        except StopIteration:
            break
        device = algorithm.device
        if use_extracted_features:
            modality_inputs0, labels0 = batch0
            modality_inputs1, labels1 = batch1
            modality_inputs0 = {k: v.to(device) for k, v in modality_inputs0.items()}
            modality_inputs1 = {k: v.to(device) for k, v in modality_inputs1.items()}
            labels0 = labels0.to(device).long()
            labels1 = labels1.to(device).long()
        else:
            modality_inputs0 = {
                mod: batch0[0][mod].to(device) for mod in config.dataset.modalities
            }
            labels0 = batch0[1].to(device).squeeze(-1).long()

            modality_inputs1 = {
                mod: batch1[0][mod].to(device) for mod in config.dataset.modalities
            }
            labels1 = batch1[1].to(device).squeeze(-1).long()

        loss_dict, predictions = algorithm.train_step_two_domains(
            modality_inputs0, labels0, modality_inputs1, labels1
        )
        if algorithm.use_gblend and 'modality_losses' in loss_dict:
            for modality, loss_value in loss_dict['modality_losses'].items():
                gbl_losses[modality] += loss_value

        for key, value in loss_dict.items():
            if isinstance(value, dict):  
                continue
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value

         
        with torch.no_grad():
            labels_combined = torch.cat([labels0, labels1], dim=0)
            probs = torch.softmax(predictions, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels_combined.cpu().numpy().tolist())

 
        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
                'loss': loss_dict['total_loss']
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
        for key in gbl_losses.keys():
            gbl_losses[key] /= min_len
        return avg_losses, metrics, gbl_losses
    else:
        return avg_losses, metrics


def validate(algorithm, dataloader, config):
    """Validate on a single domain (supports both data formats)"""
    algorithm.model.eval()

    all_preds = []
    all_labels = []

     
    use_extracted_features = config.dataset.use_extracted_features

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Validating"):
            device = algorithm.device

             
            if use_extracted_features:
                 
                modality_inputs, labels = batch
                modality_inputs = {k: v.to(device) for k, v in modality_inputs.items()}
                labels = labels.to(device).long()
            else:
                modality_inputs = {
                    mod: batch[0][mod].to(device) for mod in config.dataset.modalities
                }
                labels = batch[1].to(device).squeeze(-1).long()

            modality_features = {}
            for modality in algorithm.modalities:
                _, features = algorithm.model.forward_modality(modality, modality_inputs[modality])
                modality_features[modality] = features

            feature_list = [modality_features[mod] for mod in sorted(modality_features.keys())]
            logits, _ = algorithm.fusion_module(feature_list)
            probs = torch.softmax(logits, dim=1)[:, 1]

     
            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

 
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    from evaluation.fas_metrics import evaluate_fas
    all_metrics = evaluate_fas(all_labels, all_preds)
   
    metrics = {
        'hter': all_metrics['hter'],
        'auc': all_metrics['auc']
    }

    return metrics


def validate_two_domains(algorithm, dataloader0, dataloader1, config):
    """
    Validate model on two domains simultaneously (supports both data formats)

    Returns:
        metrics: FAS metrics dict
        gbl_losses: GBL losses (if enabled)
    """
    algorithm.model.eval()
 
    all_preds = []
    all_labels = []
    gbl_losses = None
    if algorithm.use_gblend:
        gbl_losses = {mod: 0.0 for mod in algorithm.modalities}
        gbl_losses['fusion'] = 0.0

    loader0_iter = iter(dataloader0)
    loader1_iter = iter(dataloader1)
    min_len = min(len(dataloader0), len(dataloader1))

 
    use_extracted_features = config.dataset.use_extracted_features

    with torch.no_grad():
        for _ in tqdm.tqdm(range(min_len), desc="Validating two domains"):
            try:
                batch0 = next(loader0_iter)
                batch1 = next(loader1_iter)
            except StopIteration:
                break
            if use_extracted_features:
                modality_inputs0, labels0 = batch0
                modality_inputs1, labels1 = batch1
                modality_inputs0 = {k: v.to(algorithm.device) for k, v in modality_inputs0.items()}
                modality_inputs1 = {k: v.to(algorithm.device) for k, v in modality_inputs1.items()}
                labels0 = labels0.to(algorithm.device).long()
                labels1 = labels1.to(algorithm.device).long()
            else:
                modality_inputs0 = {
                    mod: batch0[0][mod].to(algorithm.device) for mod in config.dataset.modalities
                }
                labels0 = batch0[1].to(algorithm.device).squeeze(-1).long()

                modality_inputs1 = {
                    mod: batch1[0][mod].to(algorithm.device) for mod in config.dataset.modalities
                }
                labels1 = batch1[1].to(algorithm.device).squeeze(-1).long()
            _, loss_dict, fusion_predictions = algorithm._compute_all_losses(
                modality_inputs0, labels0, modality_inputs1, labels1, current_epoch=-1
            )
            if algorithm.use_gblend and 'modality_losses' in loss_dict:
                for modality, loss_value in loss_dict['modality_losses'].items():
                    gbl_losses[modality] += loss_value
            labels_combined = torch.cat([labels0, labels1], dim=0)
            probs = torch.softmax(fusion_predictions, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels_combined.cpu().numpy().tolist())
    if algorithm.use_gblend:
        for key in gbl_losses.keys():
            gbl_losses[key] /= min_len

 
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    from evaluation.fas_metrics import evaluate_fas
    all_metrics = evaluate_fas(all_labels, all_preds)
    metrics = {
        'hter': all_metrics['hter'],
        'auc': all_metrics['auc']
    }

    if algorithm.use_gblend:
        return metrics, gbl_losses
    else:
        return metrics


def main():
    parser = argparse.ArgumentParser(description='Train FAS Early Fusion DG')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--source_domains', nargs='+', required=True,
                        help='Source domains for training')
    parser.add_argument('--target_domain', type=str, required=True,
                        help='Target domain for validation')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (overrides config file if provided)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory (overrides config file if provided)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()
    config = get_config()
    config.training.source_domains = args.source_domains
    config.training.target_domain = args.target_domain
    if args.log_dir:
        config.logging.log_dir = args.log_dir
    if args.checkpoint_dir:
        config.checkpointing.checkpoint_dir = args.checkpoint_dir
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.checkpointing.checkpoint_dir, exist_ok=True)
    modality_map = {'rgb': 'r', 'depth': 'd', 'ir': 'i'}
    modality_str = ''.join([modality_map.get(m, m[0]) for m in sorted(config.dataset.modalities)])
    source_str = '+'.join(args.source_domains)
    target_str = args.target_domain
    log_filename = f"{modality_str}_{source_str}_to_{target_str}_log.txt"
    results_filename = f"{modality_str}_{source_str}_to_{target_str}_results.txt"
    log_file_path = os.path.join(config.logging.log_dir, log_filename)
    results_file_path = os.path.join(config.logging.log_dir, results_filename)
    log_file = open(log_file_path, 'w')
    log_file.write(f"Log file: {log_file_path}\n")
    log_file.write(f"Results file: {results_file_path}\n")
    log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Modalities: {config.dataset.modalities}\n")
    log_file.write(f"Source domains: {args.source_domains}\n")
    log_file.write(f"Target domain: {args.target_domain}\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()

    print(f"Log file: {log_file_path}")
    print(f"Results file: {results_file_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Modalities: {config.dataset.modalities}")
    print(f"Source domains: {args.source_domains}")
    print(f"Target domain: {args.target_domain}")
    print_config(config)
    set_random_seed(config.seed)
    save_config(config, os.path.join(config.logging.log_dir, 'config.yaml'))
    print("\n" + "=" * 60)
    print("Creating FAS Model")
    print("=" * 60)
    model, feature_dims = create_fas_model(config)
    print("\n" + "=" * 60)
    print("Creating Early Fusion DG Algorithm")
    print("=" * 60)
    algorithm = EarlyFusionDG(
        model=model,   
        config=config.algorithm,
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
            source=True   
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
    dataset_info.append(f"Val datasets (source):")
    for i, (domain, ds) in enumerate(zip(config.training.source_domains, val_datasets_source)):
        info = f"  Domain {domain}: {len(ds)} samples"
        dataset_info.append(info)
    dataset_info.append(f"Test dataset (target {config.training.target_domain}): {len(val_dataset_target)} samples")

    for line in dataset_info:
        print(line)
        log_file.write(line + "\n")
    log_file.flush()
 
    if config.dataset.use_extracted_features:
        from datasets.base import MultiModalCollator
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
            collate_fn=collate_fn
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

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("Starting Training\n")
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
        print("-" * 80)
        log_file.write(epoch_msg + "\n")
        log_file.write("-" * 80 + "\n")

        # Train
        train_result = train_epoch(
            algorithm, train_loaders[0], train_loaders[1],
            epoch, config
        )
        if algorithm.use_gblend:
            train_losses, train_metrics, gbl_train_losses = train_result
        else:
            train_losses, train_metrics = train_result
            gbl_train_losses = None

        # Print and log training results
        train_msg = f"Train - Loss: {train_losses['total_loss']:.4f}, HTER: {train_metrics['hter']:.4f}, AUC: {train_metrics['auc']:.4f}"
        modal_losses_msg = f"  Modal losses: {', '.join([f'{k}: {v:.4f}' for k, v in train_losses.items() if k != 'total_loss'])}"

        print(train_msg)
        print(modal_losses_msg)
        log_file.write(train_msg + "\n")
        log_file.write(modal_losses_msg + "\n")

        # Validate
        if (epoch + 1) % config.training.val_frequency == 0:
            val_result_source = validate_two_domains(
                algorithm, val_loaders_source[0], val_loaders_source[1], config
            )
            if algorithm.use_gblend:
                val_metrics_source, gbl_val_losses = val_result_source
            else:
                val_metrics_source = val_result_source
                gbl_val_losses = None

            val_source_msg = f"Val Source - HTER: {val_metrics_source['hter']:.4f}, AUC: {val_metrics_source['auc']:.4f}"
            print(val_source_msg)
            log_file.write(val_source_msg + "\n")
            test_metrics = validate(algorithm, val_loader_target, config)
            test_msg = f"Test (Target) - HTER: {test_metrics['hter']:.4f}, AUC: {test_metrics['auc']:.4f}"
            print(test_msg)
            log_file.write(test_msg + "\n")
            log_file.flush()
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
            if val_metrics_source['hter'] < best_val_hter:
                best_val_hter = val_metrics_source['hter']
                best_test_hter = test_metrics['hter']
                best_val_auc = val_metrics_source['auc']
                best_test_auc = test_metrics['auc']
                best_epoch = epoch + 1

                best_msg = f"\n*** New best model! Val HTER: {best_val_hter:.4f}, Val AUC: {best_val_auc:.4f}, Test HTER: {best_test_hter:.4f}, Test AUC: {best_test_auc:.4f} ***"
                print(best_msg)
                log_file.write(best_msg + "\n")

                if config.checkpointing.save_best:
                    checkpoint_path = os.path.join(
                        config.checkpointing.checkpoint_dir,
                        'best_model.pth'
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': algorithm.model.state_dict(),
                        'optimizer_state_dict': algorithm.optimizer.state_dict(),
                        'val_metrics': val_metrics_source,
                        'test_metrics': test_metrics,
                        'config': config
                    }, checkpoint_path)
                    save_msg = f"Saved best model to {checkpoint_path}"
                    print(save_msg)
                    log_file.write(save_msg + "\n")
                log_file.flush()

        # Save checkpoint
        if (epoch + 1) % config.training.save_frequency == 0:
            checkpoint_path = os.path.join(
                config.checkpointing.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': algorithm.model.state_dict(),
                'optimizer_state_dict': algorithm.optimizer.state_dict(),
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    if config.checkpointing.save_last:
        final_path = os.path.join(
            config.checkpointing.checkpoint_dir,
            'final_model.pth'
        )
        torch.save({
            'epoch': config.training.num_epochs - 1,
            'model_state_dict': algorithm.model.state_dict(),
            'optimizer_state_dict': algorithm.optimizer.state_dict(),
            'config': config
        }, final_path)
        print(f"\nSaved final model: {final_path}")

  
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best source validation - HTER: {best_val_hter:.4f}, AUC: {best_val_auc:.4f}")
    print(f"Corresponding target test - HTER: {best_test_hter:.4f}, AUC: {best_test_auc:.4f}")
    print("=" * 80)

    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("Training Completed!\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Best epoch: {best_epoch}\n")
    log_file.write(f"Best source validation - HTER: {best_val_hter:.4f}, AUC: {best_val_auc:.4f}\n")
    log_file.write(f"Corresponding target test - HTER: {best_test_hter:.4f}, AUC: {best_test_auc:.4f}\n")
    log_file.write("=" * 80 + "\n")
    log_file.close()

     
    with open(results_file_path, 'w') as results_file:
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Experiment: {modality_str} modalities, Source: {source_str}, Target: {target_str}\n")
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Modalities: {config.dataset.modalities}\n")
        results_file.write(f"Source domains: {config.training.source_domains}\n")
        results_file.write(f"Target domain: {config.training.target_domain}\n")
        results_file.write("\n")
        results_file.write("Training Configuration:\n")
        results_file.write(f"  Epochs: {config.training.num_epochs}\n")
        results_file.write(f"  Batch size: {config.training.batch_size}\n")
        results_file.write(f"  Learning rate: {config.training.learning_rate}\n")
        results_file.write(f"  Weight decay: {config.training.weight_decay}\n")
        results_file.write(f"  Optimizer: {config.training.optimizer_type}\n")
        results_file.write(f"  Use extracted features: {config.dataset.use_extracted_features}\n")
        results_file.write(f"  Use GBL: {config.algorithm.use_gblend}\n")
        results_file.write(f"  Use contrastive loss: {config.algorithm.use_contras}\n")
        results_file.write(f"  Use modal translation: {config.algorithm.use_modtrans}\n")
        results_file.write(f"  Use mixup: {config.algorithm.use_mixup}\n")
        results_file.write(f"  Fusion type: {config.algorithm.fusion_type}\n")
        results_file.write(f"  Alpha contrast: {config.algorithm.alpha_contrast}\n")
        results_file.write(f"  Alpha trans: {config.algorithm.alpha_trans}\n")
        results_file.write(f"  ld (modal gap weight): {config.algorithm.ld}\n")
        results_file.write("\n")
        results_file.write("=" * 80 + "\n")
        results_file.write("FINAL RESULTS\n")
        results_file.write("=" * 80 + "\n")
        results_file.write(f"Best epoch: {best_epoch}\n")
        results_file.write(f"Best source validation:\n")
        results_file.write(f"  HTER: {best_val_hter:.4f}\n")
        results_file.write(f"  AUC:  {best_val_auc:.4f}\n")
        results_file.write(f"Corresponding target test:\n")
        results_file.write(f"  HTER: {best_test_hter:.4f}\n")
        results_file.write(f"  AUC:  {best_test_auc:.4f}\n")
        results_file.write("=" * 80 + "\n")

    print(f"\nResults saved to: {results_file_path}")
    print(f"Training log saved to: {log_file_path}")


if __name__ == '__main__':
    main()