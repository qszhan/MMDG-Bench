"""
Face Anti-Spoofing (FAS) Task Definition
"""

import torch.nn as nn
from typing import Dict, List, Any

from .base import BaseTask, TaskRegistry


@TaskRegistry.register('face_antispoofing')
class FaceAntiSpoofingTask(BaseTask):
    """
    Face anti-spoofing task with multi-modal inputs

    Typical modalities: RGB, Depth, Infrared
    Output: Binary classification (live vs spoof)
    Datasets: CASIA-CeFA, CASIA-SURF, WMCA, OULU-NPU
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config()

    def get_supported_modalities(self) -> List[str]:
        """Return supported modalities for FAS"""
        return ['rgb', 'depth', 'ir']

    def get_supported_datasets(self) -> List[str]:
        """Return supported FAS datasets"""
        return ['casia_cefa', 'casia_surf', 'wmca', 'oulu_npu']

    def get_loss_function(self) -> nn.Module:
        """Return binary cross-entropy loss for FAS"""
        # BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
        return nn.BCEWithLogitsLoss()

    def get_evaluation_metrics(self) -> List[str]:
        """Return evaluation metrics for FAS"""
        return [
            'auc',          # Area Under ROC Curve
            'eer',          # Equal Error Rate
            'hter',         # Half Total Error Rate
            'apcer',        # Attack Presentation Classification Error Rate
            'bpcer',        # Bona Fide Presentation Classification Error Rate
            'accuracy'      # Overall accuracy
        ]

    def get_input_requirements(self) -> Dict[str, Any]:
        """Return input requirements for FAS"""
        return {
            'modality_dims': {
                'rgb': {
                    'shape': '(batch, channels, height, width)',
                    'expected': '(N, 3, 224, 224)',
                    'type': 'RGB image',
                    'note': 'Static image (no temporal dimension)'
                },
                'depth': {
                    'shape': '(batch, channels, height, width)',
                    'expected': '(N, 3, 224, 224)',
                    'type': 'Depth map (replicated to 3 channels)',
                    'note': 'Original depth may be single channel'
                },
                'ir': {
                    'shape': '(batch, channels, height, width)',
                    'expected': '(N, 3, 224, 224)',
                    'type': 'Infrared image (replicated to 3 channels)',
                    'note': 'Original IR may be single channel'
                }
            },
            'preprocessing': {
                'rgb': [
                    'Resize to 224x224',
                    'Normalization (ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
                    'Optional: Random horizontal flip for training',
                    'Optional: Color jitter for training'
                ],
                'depth': [
                    'Resize to 224x224',
                    'Normalize to [0, 1] range',
                    'Replicate to 3 channels if single channel',
                    'Optional: Random horizontal flip for training'
                ],
                'ir': [
                    'Resize to 224x224',
                    'Normalize to [0, 1] range',
                    'Replicate to 3 channels if single channel',
                    'Optional: Random horizontal flip for training'
                ]
            },
            'temporal_info': {
                'required': False,
                'note': 'FAS typically uses single frames (not temporal sequences like action recognition)'
            },
            'label_format': {
                'type': 'binary',
                'encoding': '0=live/bonafide, 1=spoof/attack',
                'output_shape': '(batch, 1) for BCEWithLogitsLoss',
                'note': 'Use BCEWithLogitsLoss with num_classes=1'
            },
            'feature_extraction': {
                'backbone': 'ViT-Base (google/vit-base-patch16-224)',
                'feature_dim': 768,
                'extraction_point': 'Class token [:, 0, :] after last layer',
                'pretrained': 'ImageNet-21k'
            }
        }

    def validate_config(self):
        """Validate FAS-specific configuration"""
        super().validate_config()

        # Check num_classes is 1 (binary classification with BCEWithLogitsLoss)
        if hasattr(self.config, 'task') and hasattr(self.config.task, 'num_classes'):
            if self.config.task.num_classes != 1:
                raise ValueError(
                    f"FAS task expects num_classes=1 (binary with BCEWithLogitsLoss), "
                    f"got {self.config.task.num_classes}"
                )