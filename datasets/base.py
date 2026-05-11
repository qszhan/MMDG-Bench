"""
Base dataset class for MMDG-Bench

Provides abstract interface for multi-modal domain generalization datasets
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod


class BaseMMDGDataset(Dataset, ABC):
    """
    Base class for multi-modal domain generalization datasets

    All task-specific datasets should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        split: str,
        domains: List[str],
        modalities: List[str],
        task_type: str,
        data_root: str,
        **kwargs
    ):
        """
        Args:
            split: 'train', 'val', or 'test'
            domains: List of domain names to include (e.g., ['D1', 'D2'])
            modalities: List of modalities to use (e.g., ['rgb', 'flow', 'audio'])
            task_type: Task type identifier (e.g., 'action_recognition')
            data_root: Root directory of dataset
            **kwargs: Additional task-specific arguments
        """
        super().__init__()

        self.split = split
        self.domains = domains
        self.modalities = modalities
        self.task_type = task_type
        self.data_root = data_root

        # Validate modalities
        self._validate_modalities()

        # Load dataset samples
        self.samples = self._load_samples()

    @abstractmethod
    def _validate_modalities(self):
        """Validate that requested modalities are supported for this task"""
        pass

    @abstractmethod
    def _load_samples(self) -> List[Any]:
        """
        Load and return list of samples

        Returns:
            List of sample metadata (format depends on task)
        """
        pass

    @abstractmethod
    def _load_modality_data(self, sample_idx: int, modality: str) -> torch.Tensor:
        """
        Load data for a specific modality

        Args:
            sample_idx: Index of the sample
            modality: Modality name (e.g., 'rgb', 'flow', 'audio')

        Returns:
            Tensor containing modality data
        """
        pass

    @abstractmethod
    def _get_label(self, sample_idx: int) -> int:
        """
        Get label for a sample

        Args:
            sample_idx: Index of the sample

        Returns:
            Label (integer)
        """
        pass

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Get a single sample

        Args:
            index: Sample index

        Returns:
            Tuple of (modality_dict, label) where:
                - modality_dict: Dict mapping modality names to tensors
                - label: Integer label
        """
        modality_dict = {}

        for modality in self.modalities:
            try:
                modality_dict[modality] = self._load_modality_data(index, modality)
            except Exception as e:
                print(f"Warning: Failed to load {modality} for sample {index}: {e}")
                # Return placeholder (zeros) for missing modality
                modality_dict[modality] = self._get_modality_placeholder(modality)

        label = self._get_label(index)

        return modality_dict, label

    def _get_modality_placeholder(self, modality: str) -> torch.Tensor:
        """
        Return a placeholder tensor for missing modality data

        Args:
            modality: Modality name

        Returns:
            Zero tensor with appropriate shape
        """
        # Default placeholder - subclasses should override for specific shapes
        return torch.tensor(0)

    def __len__(self) -> int:
        return len(self.samples)

    def get_num_classes(self) -> int:
        """Return number of classes for this task"""
        raise NotImplementedError("Subclasses must implement get_num_classes()")

    def get_modality_dims(self) -> Dict[str, Tuple]:
        """
        Return feature dimensions for each modality

        Returns:
            Dict mapping modality names to dimension tuples
        """
        raise NotImplementedError("Subclasses must implement get_modality_dims()")

    def get_domain_info(self) -> Dict[str, Any]:
        """
        Return information about domains

        Returns:
            Dict with domain statistics and metadata
        """
        domain_sample_counts = {}
        for sample in self.samples:
            domain = self._get_sample_domain(sample)
            domain_sample_counts[domain] = domain_sample_counts.get(domain, 0) + 1

        return {
            'domains': self.domains,
            'sample_counts': domain_sample_counts,
            'total_samples': len(self.samples)
        }

    @abstractmethod
    def _get_sample_domain(self, sample: Any) -> str:
        """Extract domain identifier from a sample"""
        pass


class MultiModalCollator:
    """
    Custom collate function for multi-modal data

    Handles batching of different modalities with potentially different shapes
    """

    def __init__(self, modalities: list[str]):
        self.modalities = modalities

    def _to_tensor(self, x):
       
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        raise TypeError(f"Unsupported modality data type: {type(x)}")

    def __call__(self, batch):
        batched_modalities = {}
        labels = []

        for modality in self.modalities:
            modality_data = [item[0][modality] for item in batch]

            
            if len(modality_data) > 0 and isinstance(modality_data[0], dict):
                imgs_data = [self._to_tensor(d['imgs']) for d in modality_data]
                try:
                    batched_modalities[modality] = torch.stack(imgs_data, dim=0)
                except RuntimeError:
                    
                    batched_modalities[modality] = imgs_data

            else:
                tensor_data = [self._to_tensor(x) for x in modality_data]
                try:
                    batched_modalities[modality] = torch.stack(tensor_data, dim=0)
                except RuntimeError:
                    batched_modalities[modality] = tensor_data

        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return batched_modalities, labels