"""Task definitions for MMDG-Bench"""

from .base import BaseTask, TaskRegistry
from .action_recognition import ActionRecognitionTask, EmotionRecognitionTask
from .fas import FaceAntiSpoofingTask

__all__ = [
    'BaseTask',
    'TaskRegistry',
    'ActionRecognitionTask',
    'FaceAntiSpoofingTask'
]