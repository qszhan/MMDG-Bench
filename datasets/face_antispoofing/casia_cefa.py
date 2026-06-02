"""
CASIA-CeFA Face Anti-Spoofing Dataset

CASIA-CeFA (Cross-ethnicity Face Anti-spoofing) dataset with RGB, Depth, and IR modalities.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
import torch

from .base_fas import BaseFASDataset


class CASIACeFADataset(BaseFASDataset):
    """
    CASIA-CeFA Face Anti-Spoofing Dataset

    Dataset Structure:
        data_root/
        └── extracted_features/
            └── 4@1_train_features.pkl  # Pre-extracted ViT features

    Feature file format:
        {
            'rgb_features': np.ndarray [N, 768],
            'depth_features': np.ndarray [N, 768],
            'ir_features': np.ndarray [N, 768],
            'labels': np.ndarray [N],  # 0=live, 1=spoof
            'paths': List[str] [N]
        }

    Args:
        split: 'train' or 'val' or 'test'
        domains: List of domain names (for CASIA-CeFA, we use ['CeFA'])
        modalities: List of modality names ['rgb', 'depth', 'ir']
        data_root: Root directory containing the dataset
        use_extracted_features: Whether to use pre-extracted features (default: True)
        source: Whether this is a source domain (True) or target domain (False)
                - (split='train', source=True): Load train features (training on source)
                - (split='val', source=True): Load val features (validation on source)
                - (split='test', source=False): Load train features (test on target)
                - (split='test', source=True): Load val features (test on source)
        **kwargs: Additional arguments
    """

    def __init__(self,
                 split: str,
                 domains: List[str],
                 modalities: List[str],
                 data_root: str,
                 use_extracted_features: bool = True,
                 source: bool = True,
                 **kwargs):

        self.dataset_name = 'CASIA-CeFA'
        self.use_extracted_features = use_extracted_features

        # Initialize empty attributes (will be populated by _load_cefa_data)
        self.labels = torch.tensor([])
        self.features = {}
        self.paths = []
        self.raw_dataset = None  # For raw image loading

        # Feature file paths
        # Features are already pre-split, no need to split in code
        self.feature_files = {
            'train': 'extracted_features/4@1_train_features.pkl',
            'val': 'extracted_features/4@1_dev_features.pkl',
        }

        # Protocol file paths (for raw image loading)
        self.protocol_files = {
            'train': '/data_raid/algorithm/workSpace/adamwang/QianshanZhan/MMDG/datasets/FAS/Modal_Protocols/4@1_train_3frames.txt',
            'val': '/data_raid/algorithm/workSpace/adamwang/QianshanZhan/MMDG/datasets/FAS/Modal_Protocols/4@1_dev_3frames.txt',
            # 'test': '/data_raid/algorithm/workSpace/adamwang/QianshanZhan/MMDG/datasets/FAS/Modal_Protocols/CASIA-SURF_CeFA_test.txt'
        }

        # Store split and source for use in _load_cefa_data
        self.split = split
        self.source = source
        self.data_root = data_root
        self.modalities = modalities

        # Load data before calling super().__init__()
        # This is necessary because BaseMMDGDataset.__init__() calls _load_samples()
        # which needs self.labels to exist
        if use_extracted_features:
            self._load_cefa_data()
        else:
            self._load_cefa_raw()

        super().__init__(
            split=split,
            domains=domains,
            modalities=modalities,
            data_root=data_root,
            use_extracted_features=use_extracted_features,
            **kwargs
        )

    def _load_cefa_data(self):
        """
        Load CASIA-CeFA features from pickle file

        Loading logic (following HAC pattern):
        - If (split == 'train' and source) or (split == 'test' and not source):
            Load 'train' features
        - Otherwise:
            Load 'val' features
        """

        # Determine which feature file to load based on split and source
        if (self.split == 'train' and self.source) or (self.split == 'test' and not self.source):
            feature_key = 'train'
        else:
            feature_key = 'val'

        feature_file = os.path.join(self.data_root, 'extracted_features', self.feature_files[feature_key])

        if not os.path.exists(feature_file):
            raise FileNotFoundError(
                f"Feature file not found: {feature_file}\n"
                f"Please extract features first using extract_features.py"
            )

        print(f"Loading CeFA features (split={self.split}, source={self.source}) from: {feature_file}")

        with open(feature_file, 'rb') as f:
            data = pickle.load(f)

        # Extract features and labels
        self.features = {}

        if 'rgb' in self.modalities:
            self.features['rgb'] = torch.from_numpy(data['rgb_features']).float()

        if 'depth' in self.modalities:
            self.features['depth'] = torch.from_numpy(data['depth_features']).float()

        if 'ir' in self.modalities:
            self.features['ir'] = torch.from_numpy(data['ir_features']).float()

        # Labels: 0=live, 1=spoof
        self.labels = torch.from_numpy(data['labels']).long()

        # Paths (for debugging/visualization)
        self.paths = data['paths']

        # No need to split data here - features are already pre-split
        print(f"  Loaded {len(self.labels)} samples")
        print(f"  Modalities: {list(self.features.keys())}")
        for modality, feats in self.features.items():
            print(f"    {modality}: {feats.shape}")
        print(f"  Labels: {self.labels.shape}")
        print(f"  Live samples: {(self.labels == 0).sum().item()}")
        print(f"  Spoof samples: {(self.labels == 1).sum().item()}")

    def _load_cefa_raw(self):
        """
        Load CASIA-CeFA raw images using Flex-Modal-FAS loaders

        Two-step logic:
        1. Select protocol file (data source) based on HAC pattern
        2. Select loader type (augmentation) based on split
        """
        # Import only when needed (to avoid requiring imgaug when using pre-extracted features)
        from torchvision import transforms
        from .Load_FAS_MultiModal import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout
        from .Load_FAS_MultiModal_test import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest

        # Step 1: Select protocol file based on HAC pattern (data source)
        if (self.split == 'train' and self.source) or (self.split == 'test' and not self.source):
            protocol_key = 'train'  # Use train data
        else:
            protocol_key = 'val'    # Use val data


        # protocol_file = os.path.join(self.data_root, self.dataset_name, 'phase1', self.protocol_files[protocol_key])
        protocol_file = self.protocol_files[protocol_key]
        print(f"Loading CeFA raw images (split={self.split}, source={self.source})")
        print(f"  Protocol: {protocol_file} (using '{protocol_key}' data)")
        whole_path  = os.path.join(self.data_root, self.dataset_name, 'phase1')
        # Step 2: Select loader type based on split (augmentation)
        if self.split == 'train':
            # Training: Use Spoofing_train with augmentations
            self.raw_dataset = Spoofing_train(
                protocol_file,
                whole_path,
                transform=transforms.Compose([
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Cutout(),
                    Normaliztion()
                ])
            )
            print(f"  Loader: Spoofing_train (with augmentations)")
        else:
            # Validation/Testing: Use Spoofing_valtest without augmentations
            self.raw_dataset = Spoofing_valtest(
                protocol_file,
                whole_path,
                transform=transforms.Compose([
                    Normaliztion_valtest(),
                    ToTensor_valtest()
                ])
            )
            print(f"  Loader: Spoofing_valtest (without augmentations)")

        print(f"  Loaded {len(self.raw_dataset)} samples")

    def __len__(self) -> int:
        """Return number of samples"""
        if self.use_extracted_features:
            return len(self.labels)
        else:
            return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a sample

        Args:
            idx: Sample index

        Returns:
            modality_inputs: Dict of {modality_name: tensor [768] or [3, 224, 224]}
            label: Binary label (0=live, 1=spoof)
        """
        if self.use_extracted_features:
            # Return pre-extracted features
            modality_inputs = {}
            for modality in self.modalities:
                modality_inputs[modality] = self.features[modality][idx]

            label = self.labels[idx]

            return modality_inputs, label
        else:
            # Return raw images
            sample = self.raw_dataset[idx]

            # Convert Flex-Modal-FAS format to our format
            # Flex-Modal-FAS returns: {'image_x': rgb, 'image_x_depth': depth, 'image_x_ir': ir, 'spoofing_label': label, ...}
            modality_inputs = {}
            if 'rgb' in self.modalities:
                modality_inputs['rgb'] = sample['image_x']
            if 'depth' in self.modalities:
                modality_inputs['depth'] = sample['image_x_depth']
            if 'ir' in self.modalities:
                modality_inputs['ir'] = sample['image_x_ir']

            label = sample['spoofing_label']

            return modality_inputs, label

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample"""
        return {
            'index': idx,
            'path': self.paths[idx],
            'label': self.labels[idx].item(),
            'dataset': self.dataset_name,
            'split': self.split
        }

    # Implement abstract methods from BaseMMDGDataset
    # (These are required but not used since we override __getitem__)

    def _validate_modalities(self):
        """Validate that requested modalities are supported"""
        for modality in self.modalities:
            if modality not in self.SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Unsupported modality: {modality}. "
                    f"Supported: {self.SUPPORTED_MODALITIES}"
                )

    def _load_samples(self) -> List:
        """Return list of sample indices (required by base class)"""
        # Not used since we override __getitem__, but required by abstract base
        return list(range(len(self.labels)))

    def _load_modality_data(self, sample_idx: int, modality: str) -> torch.Tensor:
        """Load data for a specific modality (required by base class)"""
        # Not used since we override __getitem__, but required by abstract base
        if self.use_extracted_features:
            return self.features[modality][sample_idx]
        else:
            raise NotImplementedError("Raw image loading not implemented")

    def _get_label(self, sample_idx: int) -> int:
        """Get label for a sample (required by base class)"""
        # Not used since we override __getitem__, but required by abstract base
        return self.labels[sample_idx].item()

    def _get_sample_domain(self, sample: int) -> str:
        """Get domain for a sample (required by base class)"""
        # Not used since we override __getitem__, but required by abstract base
        # For CeFA, all samples belong to 'CeFA' domain
        return self.dataset_name