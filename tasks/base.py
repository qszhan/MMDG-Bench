"""
Base task definition for MMDG-Bench
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch.nn as nn


class BaseTask(ABC):
    """
    Base class for all tasks in MMDG-Bench

    Defines the interface for task-specific configurations and behaviors
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Task configuration dictionary
        """
        self.config = config
        self.task_name = config.get('task_name', 'unknown')
        self.num_classes = config.get('num_classes', 2)

    @abstractmethod
    def get_supported_modalities(self) -> List[str]:
        """Return list of supported modalities for this task"""
        pass

    @abstractmethod
    def get_supported_datasets(self) -> List[str]:
        """Return list of supported dataset names"""
        pass

    @abstractmethod
    def get_loss_function(self) -> nn.Module:
        """Return the appropriate loss function for this task"""
        pass

    @abstractmethod
    def get_evaluation_metrics(self) -> List[str]:
        """Return list of evaluation metric names"""
        pass

    @abstractmethod
    def get_input_requirements(self) -> Dict[str, Any]:
        """
        Return input requirements for this task

        Returns:
            Dict containing:
                - 'modality_dims': Expected dimensions for each modality
                - 'preprocessing': Required preprocessing steps
                - etc.
        """
        pass

    def validate_config(self):
        """Validate task configuration"""
        required_keys = ['task_name', 'num_classes']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Task config missing required key: {key}")


class TaskRegistry:
    """Registry for all available tasks"""

    _tasks = {}

    @classmethod
    def register(cls, task_name: str):
        """
        Decorator to register a task class

        Usage:
            @TaskRegistry.register('action_recognition')
            class ActionRecognitionTask(BaseTask):
                ...
        """
        def decorator(task_class):
            if task_name in cls._tasks:
                raise ValueError(f"Task '{task_name}' already registered")
            cls._tasks[task_name] = task_class
            return task_class
        return decorator

    @classmethod
    def get_task(cls, task_name: str, config: Dict[str, Any]) -> BaseTask:
        """
        Get task instance by name

        Args:
            task_name: Name of the task
            config: Task configuration

        Returns:
            Task instance
        """
        if task_name not in cls._tasks:
            raise ValueError(
                f"Task '{task_name}' not found. "
                f"Available tasks: {list(cls._tasks.keys())}"
            )
        return cls._tasks[task_name](config)

    @classmethod
    def list_tasks(cls) -> List[str]:
        """Return list of registered task names"""
        return list(cls._tasks.keys())

    @classmethod
    def get_task_info(cls, task_name: str) -> Dict[str, Any]:
        """Get information about a registered task"""
        if task_name not in cls._tasks:
            raise ValueError(f"Task '{task_name}' not found")

        task_class = cls._tasks[task_name]
        # Create temporary instance to get info
        dummy_config = {'task_name': task_name, 'num_classes': 2}
        task_instance = task_class(dummy_config)

        return {
            'name': task_name,
            'supported_modalities': task_instance.get_supported_modalities(),
            'supported_datasets': task_instance.get_supported_datasets(),
            'evaluation_metrics': task_instance.get_evaluation_metrics(),
        }