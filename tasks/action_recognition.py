"""
Action Recognition Task Definition
"""

import torch.nn as nn
from typing import Dict, List, Any

from .base import BaseTask, TaskRegistry


@TaskRegistry.register('action_recognition')
class ActionRecognitionTask(BaseTask):
    """
    Action recognition task with multi-modal inputs

    Typical modalities: RGB video, optical flow, audio
    Output: Multi-class classification (action/verb classes)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config()

    def get_supported_modalities(self) -> List[str]:
        """Return supported modalities"""
        return ['rgb', 'flow', 'audio']

    def get_supported_datasets(self) -> List[str]:
        """Return supported datasets"""
        return ['epic_kitchens', 'hac']

    def get_loss_function(self) -> nn.Module:
        """Return cross-entropy loss for multi-class classification"""
        return nn.CrossEntropyLoss()

    def get_evaluation_metrics(self) -> List[str]:
        """Return evaluation metrics for action recognition"""
        return [
            'accuracy',      # Top-1 accuracy
            'top5_accuracy', # Top-5 accuracy
            'per_class_accuracy',  # Per-class accuracy
            'confusion_matrix'     # Confusion matrix
        ]

    def get_input_requirements(self) -> Dict[str, Any]:
        """Return input requirements"""
        return {
            'modality_dims': {
                'rgb': {
                    'shape': '(batch, channels, time, height, width)',
                    'expected': '(N, 3, T, 224, 224)',
                    'type': '3D CNN input'
                },
                'flow': {
                    'shape': '(batch, channels, time, height, width)',
                    'expected': '(N, 2, T, 224, 224)',
                    'type': '3D CNN input - optical flow (u,v)'
                },
                'audio': {
                    'shape': '(batch, freq_bins, time_bins)',
                    'expected': '(N, 257, 512)',
                    'type': 'Spectrogram'
                }
            },
            'preprocessing': {
                'rgb': [
                    'Frame sampling',
                    'Resize to 224x224',
                    'Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
                    'Optional: Random crop, flip for training'
                ],
                'flow': [
                    'Frame sampling (typically half of RGB frames)',
                    'Resize to 224x224',
                    'Normalization',
                ],
                'audio': [
                    'Extract audio segment aligned with video',
                    'Resample to fixed length (160000 samples)',
                    'Compute spectrogram (nperseg=512, noverlap=353)',
                    'Log-scale and normalize',
                    'Optional: Add noise and masking for training'
                ]
            },
            'temporal_info': {
                'required': True,
                'typical_clip_length': '8-16 frames',
                'sampling_strategy': 'Dense or sparse sampling'
            }
        }


# Placeholder tasks for future implementation

@TaskRegistry.register('face_antispoofing')
class FaceAntiSpoofingTask(BaseTask):
    """
    Face anti-spoofing task (PLACEHOLDER)

    TODO: Implement when adding face anti-spoofing datasets
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        raise NotImplementedError(
            "Face anti-spoofing task not yet implemented. "
            "This is a placeholder for future extension."
        )

    def get_supported_modalities(self) -> List[str]:
        return ['rgb', 'depth', 'ir']

    def get_supported_datasets(self) -> List[str]:
        return []  # To be filled when implemented

    def get_loss_function(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def get_evaluation_metrics(self) -> List[str]:
        return ['hter', 'auc', 'eer', 'apcer', 'bpcer']

    def get_input_requirements(self) -> Dict[str, Any]:
        return {'placeholder': True}


@TaskRegistry.register('emotion_recognition')
class EmotionRecognitionTask(BaseTask):
    """
    Emotion recognition task (PLACEHOLDER)

    TODO: Implement when adding emotion recognition datasets
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        raise NotImplementedError(
            "Emotion recognition task not yet implemented. "
            "This is a placeholder for future extension."
        )

    def get_supported_modalities(self) -> List[str]:
        return ['rgb', 'audio', 'text']

    def get_supported_datasets(self) -> List[str]:
        return []  # To be filled when implemented

    def get_loss_function(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def get_evaluation_metrics(self) -> List[str]:
        return ['accuracy', 'f1_score', 'confusion_matrix']

    def get_input_requirements(self) -> Dict[str, Any]:
        return {'placeholder': True}
