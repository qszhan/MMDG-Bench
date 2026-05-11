"""
Base algorithm class for domain generalization
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class BaseAlgorithm(ABC):
    """
    Base class for domain generalization algorithms

    All DG algorithms should inherit from this class and implement:
    - compute_loss: Compute the total loss given inputs
    - update: Update model parameters
    """

    def __init__(self, model, config):
        """
        Args:
            model: The multi-modal model
            config: Algorithm configuration
        """
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize optimizer (to be set by subclasses)
        self.optimizer = None

    def compute_loss(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the total loss for training

        This method is optional. Subclasses can override it if they use single-domain training.
        For multi-domain training algorithms, implement custom training methods instead.

        Args:
            modality_inputs: Dict mapping modality names to input tensors
            labels: Ground truth labels
            domain_ids: Domain identifiers (optional)
            **kwargs: Additional algorithm-specific arguments

        Returns:
            total_loss: Scalar loss tensor
            loss_dict: Dict of individual loss components for logging
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_loss. "
            "This algorithm may use a different training paradigm (e.g., multi-domain training)."
        )

    @abstractmethod
    def update(self, loss: torch.Tensor):
        """
        Update model parameters given the loss

        Args:
            loss: Total loss tensor
        """
        pass

    def predict(self, modality_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Make predictions

        Args:
            modality_inputs: Dict mapping modality names to input tensors

        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(modality_inputs)
        return predictions

    def train_step(
        self,
        modality_inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Perform one training step

        Args:
            modality_inputs: Dict mapping modality names to input tensors
            labels: Ground truth labels
            domain_ids: Domain identifiers (optional)
            **kwargs: Additional arguments

        Returns:
            Dict of losses for logging
        """
        self.model.train()

        # Compute loss
        total_loss, loss_dict = self.compute_loss(
            modality_inputs, labels, domain_ids, **kwargs
        )

        # Update parameters
        self.update(total_loss)

        return loss_dict

    def get_parameters(self):
        """Get all trainable parameters"""
        return self.model.parameters()

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])