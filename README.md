# MMDG-Bench: Unified Benchmark for Multi-Modal Domain Generalization 

MMDG-Bench is a PyTorch benchmark suite for multi-modal domain generalization. 

## Overview

This benchmark addresses the challenging problem of Multi-Modal Domain Generalization (MMDG), where models:
1. Handle multiple input modalities (e.g., RGB, Depth, IR, Audio, etc.)
2. Generalize across different domains (train on source domains, test on unseen target domains) 
It comprises two complementary frameworks: DG then MML (D2M) and MML then DG (M2D), alongside unified experimental protocols for action recognition and face anti-spoofing, with high flexibility to integrate additional algorithms and tasks.


## Frameworks
#### D2M framework (Early Fusion)
D2M learns domain-invariant representations per modality before cross-modal fusion.
#### M2D framework (late Fusion)
M2D aligns modalities then enhances domain generalization on fused features.

## Available Tasks
The benchmark currently supports two tasks:
### Face Anti-Spoofing (FAS)
Binary classification to distinguish real faces from spoofing attacks across different sensors and capture conditions.
- **Modalities**: RGB, Depth, IR
- **Datasets**: CASIA-SURF, CASIA-CeFA, WMCA 
- **Domains**: each dataset is one domain

### Action Recognition (AR)
Multi-class classification of human actions from video and audio.
- **Modalities**: RGB video, Audio
- **Datasets**: Epic-Kitchens, HAC
- **Domains**: three different kitchens D1, D2, D3 for Epic-Kitchens, and three domains H, A, C for HAC dataset



## Domain Generalization (DG) Methods
Located in `algorithms/DG_methods/`:
- **ERM**: Empirical Risk Minimization
- **MIRO**: Mutual-Information Regularization for domain generalization
- **MMD**: Maximum Mean Discrepancy for domain alignment
- **Mixup**: Inter-domain mixup augmentation
- **Domain Contrastive Loss**: Contrastive learning for domain-invariant features

## Multi-Modal Learning (MML) Methods
Located in `algorithms/MMmethods/`:
- **GBL (Generalized Blending)**: Dynamic modality weighting based on generalization performance
- **CMT (Cross-Modal Translation)**: Translates between modalities for semantic consistency
- **Modal Contrastive Loss**: Supervised contrastive learning across modalities
- **Bottleneck Fusion**: Compact fusion through bottleneck architectures
 

## Key Features
1. **Modular Design**: Easily combine DG and MML methods under unified framework
2. **ViT Backbones**: Support ViT and VideoMAE backbones for strong baseline performance
3. **Flexible Configuration**: YAML-based configuration system for reproducible experiments
4. **Comprehensive Evaluation**: Standard metrics for each task (HTER, AUC for FAS; accuracy for action recognition)
5. **Pre-extracted Features**: Support for pre-computed features to speed up experimentation

 
## Quick Start
### 1. Prepare Datasets
Organize your datasets in the following structure:
```
/path/to/FAS/
CeFA/
SURF/
WMCA/

/path/to/ActionRecognition/
epic_kitchens/
hac/
```

### 2. Configure Experiment

Edit the configuration file (e.g., `configs/fas_early_fusion.yaml`):

```yaml
dataset:
  data_root: "/path/to/FAS"
  modalities: ["rgb", "depth", "ir"]

training:
  source_domains: ["CeFA", "SURF"]
  target_domain: "WMCA"
  batch_size: 32
  num_epochs: 30
  learning_rate: 1.0e-4

algorithm:
  use_gblend: true
  use_contras: true
  use_modtrans: true
```

### 3. Train a Model

#### Face Anti-Spoofing with Early Fusion
```bash
python scripts/train_fas_early_fusion.py \
    --config configs/fas_early_fusion.yaml \
    --modalities rgb depth \
    --source_domains CeFA SURF \
    --target_domain WMCA
```

#### Face Anti-Spoofing with Late Fusion
```bash
python scripts/train_fas_late_fusion.py \
    --config configs/fas_late_fusion.yaml \
    --modalities rgb depth \
    --source_domains CeFA SURF \
    --target_domain WMCA
```

#### Action Recognition with Late Fusion
```bash
python scripts/train_late_fusion.py \
    --config configs/action_recognition_late_fusion.yaml \
    --dataset HAC\
    --source_domains domain1 domain2 \
    --target_domain domain3
```

  
 
 
 