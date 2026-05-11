"""
Base Dataset Class for Face Anti-Spoofing

This provides common functionality for all FAS datasets.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image
import pickle

from datasets.base import BaseMMDGDataset


class BaseFASDataset(BaseMMDGDataset):
    """
    Base class for Face Anti-Spoofing datasets

    Common features across all FAS datasets:
    - Binary classification (live=0 vs spoof=1)
    - Multi-modal: RGB + Depth + IR
    - Static images (no temporal sequence needed)
    - Multiple domains/protocols for domain generalization

    Supports two modes:
    1. Load pre-extracted ViT features (.pkl files)
    2. Load raw images on-the-fly
    """

    SUPPORTED_MODALITIES = ['rgb', 'depth', 'ir']

    def __init__(self,
                 split: str,
                 domains: List[str],
                 modalities: List[str],
                 data_root: str,
                 use_extracted_features: bool = False,
                 features_dir: Optional[str] = None,
                 transform=None,
                 **kwargs):
        """
        Args:
            split: 'train', 'val', or 'test'
            domains: List of domain names (e.g., ['4@1', '4@2'] for CeFA)
            modalities: List of modalities to use (e.g., ['rgb', 'depth', 'ir'])
            data_root: Root directory of dataset
            use_extracted_features: If True, load pre-extracted ViT features
            features_dir: Directory containing extracted .pkl files
            transform: Transform to apply to images (only used when loading raw images)
        """
        self.use_extracted_features = use_extracted_features
        self.features_dir = features_dir
        self.transform = transform

        # If not using extracted features and no transform provided, use default
        if not use_extracted_features and transform is None:
            self.transform = self._get_default_transform(split)

        # Cache for loaded features (lazy loading)
        self._features_cache = {}

        # Call parent class with task_type
        super().__init__(
            split=split,
            domains=domains,
            modalities=modalities,
            task_type='face_antispoofing',
            data_root=data_root,
            **kwargs
        )

    def _get_default_transform(self, split: str):
        """
        Get default transform for FAS images

        Uses ImageNet normalization since we're using ViT pretrained on ImageNet
        """
        from torchvision import transforms

        if split == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                # Optional: add more augmentation
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet mean
                    std=[0.229, 0.224, 0.225]     # ImageNet std
                )
            ])
        else:  # val or test
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _load_image(self, img_path: str) -> Image.Image:
        """
        Load image from path

        Args:
            img_path: Path to image file

        Returns:
            PIL Image in RGB format
        """
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            return Image.new('RGB', (224, 224), (0, 0, 0))

    def _replicate_channels(self, img: Image.Image, target_channels: int = 3) -> Image.Image:
        """
        Replicate single-channel image to multi-channel

        Depth and IR images are often single-channel, but ViT expects 3 channels.

        Args:
            img: PIL Image (can be 'L' mode for grayscale)
            target_channels: Number of channels to replicate to (default 3)

        Returns:
            PIL Image in RGB format
        """
        if img.mode == 'L':
            # Convert grayscale to RGB by replicating channels
            return Image.merge('RGB', [img, img, img])
        elif img.mode == 'RGB':
            return img
        else:
            # Convert other modes to RGB
            return img.convert('RGB')

    def _load_extracted_features(self, domain: str) -> Dict[str, Any]:
        """
        Load pre-extracted features for a domain (lazy loading with cache)

        Args:
            domain: Domain name (e.g., '4@1' for CeFA)

        Returns:
            Dictionary containing:
                'rgb_features': np.ndarray [N, 768]
                'depth_features': np.ndarray [N, 768]
                'ir_features': np.ndarray [N, 768]
                'labels': np.ndarray [N]
                'paths': List[str] (length N)
        """
        # Check cache first
        if domain in self._features_cache:
            return self._features_cache[domain]

        # Construct feature file path
        # Default naming: {domain}_train_features.pkl or {domain}_test_features.pkl
        feature_filename = f"{domain}_{self.split}_features.pkl"
        feature_path = Path(self.features_dir) / feature_filename

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}\n"
                f"Make sure you've extracted features and placed them in {self.features_dir}"
            )

        # Load features
        print(f"Loading features from {feature_path}...")
        with open(feature_path, 'rb') as f:
            features = pickle.load(f)

        # Cache for future use
        self._features_cache[domain] = features

        print(f"  Loaded {len(features['labels'])} samples")
        print(f"  RGB features: {features['rgb_features'].shape}")
        print(f"  Depth features: {features['depth_features'].shape}")
        print(f"  IR features: {features['ir_features'].shape}")

        return features

    def get_num_classes(self) -> int:
        """
        Return number of classes for FAS

        Note: FAS is binary classification, but we return 1 here because
        we use BCEWithLogitsLoss which expects num_classes=1
        """
        return 1

    def __getitem__(self, idx: int):
        """
        Get a sample

        Returns:
            If using extracted features:
                modality_dict: Dict[str, torch.Tensor]
                    {
                        'rgb': [768],
                        'depth': [768],
                        'ir': [768]
                    }
                label: int (0=live, 1=spoof)

            If loading raw images:
                modality_dict: Dict[str, torch.Tensor]
                    {
                        'rgb': [3, 224, 224],
                        'depth': [3, 224, 224],
                        'ir': [3, 224, 224]
                    }
                label: int (0=live, 1=spoof)
        """
        if self.use_extracted_features:
            # Use pre-extracted features
            sample = self.samples[idx]
            domain = sample.get('domain') or sample.get('protocol')

            if domain is None:
                raise ValueError(
                    f"Sample at index {idx} missing 'domain' or 'protocol' key. "
                    f"Available keys: {sample.keys()}"
                )

            # Load features for this domain (lazy loading with cache)
            features = self._load_extracted_features(domain)

            # Get features for this specific sample
            # Assumes samples are in the same order as when features were extracted
            sample_idx_in_features = sample.get('feature_idx', idx)

            modality_dict = {}
            for modality in self.modalities:
                feat_key = f'{modality}_features'
                if feat_key not in features:
                    raise KeyError(f"Feature key '{feat_key}' not found in loaded features")

                feature_array = features[feat_key][sample_idx_in_features]
                modality_dict[modality] = torch.from_numpy(feature_array).float()

            label = int(features['labels'][sample_idx_in_features])

            return modality_dict, label

        else:
            # Load raw images on-the-fly
            return super().__getitem__(idx)

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.samples)