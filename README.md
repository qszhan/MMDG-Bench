# Welcome to MMDG-Bench
MMDG-Bench is a PyTorch benchmark suite for multi-modal domain generalization (MMDG). 
 

## Overview
### Frameworks
#### D2M framework 
D2M learns domain-invariant representations per modality before cross-modal fusion.
<img width="8176" height="2200" alt="d2m (1)" src="https://github.com/user-attachments/assets/c2ecd100-06dc-4cd5-b97e-a0d9fd4b100e" />

 

#### M2D framework  
M2D aligns modalities then enhances domain generalization on fused features.
![Overview](images/m2d.png)


### MMDG variants
#### Domain Generalization (DG) Methods
Located in `algorithms/DG_methods/`:
- **ERM**: Empirical Risk Minimization
- **MIRO**: Mutual-Information Regularization for domain generalization
- **MMD**: Maximum Mean Discrepancy for domain alignment
- **Mixup**: Inter-domain mixup augmentation
- **Domain Contrastive Loss**: Contrastive learning for domain-invariant features

#### Multi-Modal Learning (MML) Methods
Located in `algorithms/MMmethods/`:
- **GBL (Generalized Blending)**: Dynamic modality weighting based on generalization performance
- **CMT (Cross-Modal Translation)**: Translates between modalities for semantic consistency
- **Modal Contrastive Loss**: Supervised contrastive learning across modalities
- **Bottleneck Fusion**: Compact fusion through bottleneck architectures
 
### Available Tasks
The benchmark currently supports two tasks:
#### Action Recognition (AR) 
Multi-class classification of human actions from video, audio and optical flow.
- **Modalities**:  video, audio and optical flow.
- **Datasets**: Epic-Kitchens, HAC
- **Domains**: three different kitchens D1, D2, D3 for Epic-Kitchens, and three domains H, A, C for HAC dataset

#### Face Anti-Spoofing (FAS)
Binary classification to distinguish real faces from spoofing attacks across different sensors and capture conditions.
- **Modalities**: RGB, Depth, IR
- **Datasets**: CASIA-SURF, CASIA-CeFA, WMCA 
- **Domains**: each dataset is one domain

  
## Quick Start
### Environments

The code was tested using Python 3.9.21, PyTorch 2.1.2+cu121 (CUDA 12.1), and NVIDIA H100 and H20.

Dependencies:
- mmcv 1.7.2
- mmaction2 0.13.0
###  1. Data preparation
* AR task
  - EPIC-Kitchen dataset: follow the guidance [here](https://github.com/donghao51/SimMMDG) 
  - HAC dataset: follow the guidance [here](https://github.com/donghao51/SimMMDG) 
* FAS task
  - [CASIA-SURF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_A_Dataset_and_Benchmark_for_Large-Scale_Multi-Modal_Face_Anti-Spoofing_CVPR_2019_paper.pdf)
  - [CASIA-CeFA](https://openaccess.thecvf.com/content/WACV2021/html/Liu_CASIA-SURF_CeFA_A_Benchmark_for_Multi-Modal_Cross-Ethnicity_Face_Anti-Spoofing_WACV_2021_paper.html)
  - [WMCA](https://ieeexplore.ieee.org/document/8714076)

### 2. Download pretrained weights  
####  Download pretrained weights CNN backbone for AR 
- Download Audio model [link](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), rename it as vggsound_avgpool.pth.tar and place under the third_party/pretrained_models directory

- Download SlowFast model for RGB modality [link](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) and place under the third_party/pretrained_models directory

- Download SlowOnly model for Flow modality [link](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) and place under the third_party/pretrained_models directory

####  Download pretrained weights of ViT backbone for AR 
- Download AST for audio using [link](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) 
- Download VideoMAEV2 for video and flow using [link](https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/distill/vit_b_k710_dl_from_giant.pth)

####  Download pretrained weights for FAS
Download the vit follow the [link](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth)

### 3. Configure Experiment

All experiments are driven by YAML files in `configs/`. Each file controls training setup and which MMDG variant to run.

#### Understanding the 10 Variants

The benchmark produces **10 variants** by exhaustively pairing 5 DG methods with 2 frameworks:

| # | Variant | Framework | Config file | DG flag |
|---|---------|-----------|-------------|---------|
| ① | MM + ERM | D2M | `*_early.yaml` | all DG flags `false` |
| ② | MM + Mixup | D2M | `*_early.yaml` | `use_mixup: true` |
| ③ | MM + MMD | D2M | `*_early.yaml` | `use_mmd: true` |
| ④ | MM + CL | D2M | `*_early.yaml` | `use_domain_contrastive: true` |
| ⑤ | MM + MIRO | D2M | `*_early.yaml` | `use_miro: true` |
| ⑥ | MM + ERM | M2D | `*_late.yaml` | all DG flags `false` |
| ⑦ | MM + Mixup | M2D | `*_late.yaml` | `use_mixup: true` |
| ⑧ | MM + MMD | M2D | `*_late.yaml` | `use_mmd: true` |
| ⑨ | MM + CL | M2D | `*_late.yaml` | `use_domain_contrastive: true` |
| ⑩ | MM + MIRO | M2D | `*_late.yaml` | `use_miro: true` |

**MM** denotes the unified MML setting where all four MML components are enabled simultaneously across all variants. **Framework** is selected by the config file suffix (`_early` → D2M, `_late` → M2D). **DG method** is selected by setting exactly one DG flag to `true` (all `false` = ERM).

#### MML Flags — always `true` for all 10 variants

These four keys under `algorithm:` are fixed across all variants:

```yaml
# configs/hac_early.yaml
algorithm:
  use_gblend: true          # GBL: adaptive modality weighting
  use_contras: true         # Modal Contrastive Loss
  use_modtrans: true        # Cross-Modal Translation (CMT)
  fusion_type: "bottleneck" # Bottleneck Fusion
```

#### DG Flags — set exactly one to `true`

Only one DG flag should be `true` at a time; all others must be `false`. Setting all to `false` corresponds to ERM (variants ① and ⑥).

```yaml
# configs/hac_early.yaml  (shown here as MM + ERM, variant ①)
algorithm:
  use_mixup: false               # → true for variants ②/⑦  MM + Mixup
  use_mmd: false                 # → true for variants ③/⑧  MM + MMD
  use_domain_contrastive: false  # → true for variants ④/⑨  MM + CL
  use_miro: false                # → true for variants ⑤/⑩  MM + MIRO
```

**Example — MM + Mixup under D2M (variant ②):**

```yaml
# configs/hac_early.yaml
algorithm:
  # MML components (always enabled)
  use_gblend: true
  use_contras: true
  use_modtrans: true
  fusion_type: "bottleneck"

  # DG method (only one true)
  use_mixup: true
  mixup_alpha: 1
  mixup_weight: 0.1
  use_mmd: false
  use_domain_contrastive: false
  use_miro: false
```

To run the same DG method under M2D (variant ⑦), switch to `configs/hac_late.yaml` with identical DG flag settings.

#### Other Key Fields

```yaml
dataset:
  name: "hac"
  data_root: "/path/to/HAC/"           # update to your local path
  modalities: ["rgb", "flow", "audio"]
  domains: ["human", "cartoon", "animal"]

training:
  source_domains: ["human", "cartoon"]  # overridden by --source_domains
  target_domain: "animal"               # overridden by --target_domain
  num_epochs: 15
  learning_rate: 1.0e-4
  optimizer_type: "sam"                 # "adam" or "sam"
```

Pretrained backbone paths and MMAction2 config paths must also be set under `dataset:`. See `configs/hac_early.yaml` for the full reference.

### 4. Train a Model

#### Action Recognition with CNN backbone

The following example command runs a single experiment for HAC under the **D2M framework** (early fusion) with video + audio modalities:

```bash
# D2M framework — HAC, video + audio, target domain: human
CUDA_VISIBLE_DEVICES=0 python scripts/train_early_fusion.py \
    --config configs/hac_early.yaml \
    --source_domains cartoon animal \
    --target_domain human \
    --modalities rgb audio

# M2D framework — HAC, video + audio, target domain: human
CUDA_VISIBLE_DEVICES=0 python scripts/train_late_fusion.py \
    --config configs/hac_late.yaml \
    --source_domains cartoon animal \
    --target_domain human \
    --modalities rgb audio
```

The benchmark evaluates four modality combinations — `rgb+audio`, `rgb+flow`, `audio+flow`, and `rgb+flow+audio` — across all leave-one-domain-out splits. Ready-to-run shell scripts covering all four combinations are provided in `scripts_running_files/`:

| Script | Dataset | Framework |
|--------|---------|-----------|
| `scripts_running_files/hac_early.sh` | HAC | D2M |
| `scripts_running_files/hac_late.sh` | HAC | M2D |
| `scripts_running_files/epic_early.sh` | EPIC-Kitchens | D2M |
| `scripts_running_files/epic_late.sh` | EPIC-Kitchens | M2D |

Each script assigns one GPU per modality combination and supports both `parallel` and `sequential` execution modes. Edit the `CONFIGURATION SECTION` at the top of the script to set GPU assignments, batch size, and execution mode before running:

```bash
bash scripts_running_files/hac_early.sh
```

#### Action Recognition with ViT backbone

Uses VideoMAEv2 (RGB/Flow) and AST (Audio) instead of SlowFast/SlowOnly/VGGSound. Scripts are `train_early_fusion_ac_vit.py` (D2M) and `train_late_fusion_ac_vit.py` (M2D); configs are `*_early_vit.yaml` / `*_late_vit.yaml`.

```bash
# D2M framework — EPIC-Kitchens, all modalities, target domain: D2
python scripts/train_early_fusion_ac_vit.py \
    --config configs/epic_early_vit.yaml \
    --source_domains D1 D3 \
    --target_domain D2

# M2D framework — EPIC-Kitchens, all modalities, target domain: D2
python scripts/train_late_fusion_ac_vit.py \
    --config configs/epic_late_vit.yaml \
    --source_domains D1 D3 \
    --target_domain D2

# D2M framework — HAC, all modalities, target domain: cartoon
python scripts/train_early_fusion_ac_vit.py \
    --config configs/hac_early_vit.yaml \
    --dataset HAC \
    --source_domains human animal \
    --target_domain cartoon

# M2D framework — HAC, all modalities, target domain: cartoon
python scripts/train_late_fusion_ac_vit.py \
    --config configs/hac_late_vit.yaml \
    --dataset HAC \
    --source_domains human animal \
    --target_domain cartoon
```

The same leave-one-domain-out protocol applies. Shell scripts for all splits are provided in `scripts_running_files/`:

| Script | Dataset | Framework |
|--------|---------|-----------|
| `scripts_running_files/hac_early_vit.sh` | HAC | D2M (ViT) |
| `scripts_running_files/hac_late_vit.sh` | HAC | M2D (ViT) |
| `scripts_running_files/epic_early_vit.sh` | EPIC-Kitchens | D2M (ViT) |
| `scripts_running_files/epic_late_vit.sh` | EPIC-Kitchens | M2D (ViT) |

```bash
bash scripts_running_files/hac_early_vit.sh
```

#### Face Anti-Spoofing

Uses a shared ViT-Base backbone across RGB, Depth, and IR modalities. Scripts are `train_fas_early_fusion.py` (D2M) and `train_fas_late_fusion.py` (M2D); configs are `fas_early.yaml` / `fas_late.yaml`. The same 10 variants apply by toggling the DG flags in the config.

```bash
# D2M framework — target domain: WMCA (modalities set in config)
CUDA_VISIBLE_DEVICES=2 python scripts/train_fas_early_fusion.py \
    --config configs/fas_early.yaml \
    --source_domains CeFA SURF \
    --target_domain WMCA

# M2D framework — target domain: WMCA (modalities set in config)
CUDA_VISIBLE_DEVICES=1 python scripts/train_fas_late_fusion.py \
    --config configs/fas_late.yaml \
    --source_domains CeFA SURF \
    --target_domain WMCA
```

Three leave-one-domain-out splits are evaluated (target ∈ {CeFA, SURF, WMCA}). Key config fields:

```yaml
dataset:
  modalities: ["rgb", "depth"]        # or add "ir" for tri-modal
  domains: ["CeFA", "SURF", "WMCA"]

model:
  shared_backbone: true               # single ViT shared across modalities
  freeze_vit_backbone: true
  freeze_vit_layers: -1               # full freeze; set 0-11 for partial

training:
  source_domains: ["CeFA", "SURF"]    # overridden by --source_domains
  target_domain: "WMCA"               # overridden by --target_domain
  num_epochs: 20
  learning_rate: 1.0e-4
  optimizer_type: "sam"
```

Shell scripts for all splits and modality combinations are in `scripts_running_files/`:

| Script | Framework |
|--------|-----------|
| `scripts_running_files/fas_early.sh` | D2M |
| `scripts_running_files/fas_late.sh` | M2D |

```bash
bash scripts_running_files/fas_early.sh
```
