"""
Configuration utilities for MMDG-Bench
"""

import yaml
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration class with dict-like and attribute access"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._convert_to_attributes(config_dict)

    def _convert_to_attributes(self, d: Dict, prefix: str = ''):
        """Recursively convert dict to attributes"""
        for key, value in d.items():
            if isinstance(value, dict):
                # Create nested Config object
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        self._config[key] = value
        setattr(self, key, value)

    def get(self, key, default=None):
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary"""
        return self._config

    def update(self, other: Dict[str, Any]):
        """Update configuration with another dict"""
        self._config.update(other)
        self._convert_to_attributes(self._config)

    def __repr__(self):
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object
    """

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """
    Merge two configurations (override takes precedence)

    Args:
        base_config: Base configuration
        override_config: Override configuration dict

    Returns:
        Merged Config object
    """
    merged = base_config.to_dict().copy()

    def deep_update(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_update(base[key], value)
            else:
                base[key] = value

    deep_update(merged, override_config)
    return Config(merged)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MMDG-Bench Training')

    # Config file
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')

    # Task and dataset
    parser.add_argument('--task', type=str, default=None,
                        help='Task name (overrides config)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (overrides config)')

    # Domain generalization setup
    parser.add_argument('--source_domains', nargs='+', default=None,
                        help='Source domains for training (e.g., D1 D2)')
    parser.add_argument('--target_domain', type=str, default=None,
                        help='Target domain for evaluation (e.g., D3)')

    # Algorithm
    parser.add_argument('--algorithm', type=str, default=None,
                        help='DG algorithm name (e.g., simmdg, miro, erm)')

    # Modalities
    parser.add_argument('--modalities', nargs='+', default=None,
                        help='Modalities to use (e.g., rgb flow audio)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning_rate', '--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs')

    # Model
    parser.add_argument('--fusion_type', type=str, default=None,
                        help='Fusion type (concat, attention, late)')

    # Logging
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Logging directory')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='W&B project name')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    # Testing
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing (requires --resume)')

    # GPU
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')

    return parser.parse_args()


def get_config() -> Config:
    """
    Get final configuration by merging config file and command line args

    Returns:
        Final Config object
    """
    args = parse_args()

    # Load base config from file
    config = load_config(args.config)

    # Override with command line arguments
    overrides = {}

    # Task settings
    if args.task:
        overrides['task'] = overrides.get('task', {})
        overrides['task']['name'] = args.task

    if args.dataset:
        overrides['dataset'] = overrides.get('dataset', {})
        overrides['dataset']['name'] = args.dataset

    if args.modalities:
        overrides['dataset'] = overrides.get('dataset', {})
        overrides['dataset']['modalities'] = args.modalities

    # Domain setup
    if args.source_domains:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['source_domains'] = args.source_domains

    if args.target_domain:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['target_domain'] = args.target_domain

    # Algorithm
    if args.algorithm:
        overrides['algorithm'] = overrides.get('algorithm', {})
        overrides['algorithm']['name'] = args.algorithm

    # Training params
    if args.batch_size:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['batch_size'] = args.batch_size

    if args.learning_rate:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['learning_rate'] = args.learning_rate

    if args.num_epochs:
        overrides['training'] = overrides.get('training', {})
        overrides['training']['num_epochs'] = args.num_epochs

    # Model
    if args.fusion_type:
        overrides['model'] = overrides.get('model', {})
        overrides['model']['fusion_type'] = args.fusion_type

    # Logging
    if args.log_dir:
        overrides['logging'] = overrides.get('logging', {})
        overrides['logging']['log_dir'] = args.log_dir

    if args.use_wandb:
        overrides['logging'] = overrides.get('logging', {})
        overrides['logging']['use_wandb'] = True

    if args.wandb_project:
        overrides['logging'] = overrides.get('logging', {})
        overrides['logging']['wandb_project'] = args.wandb_project

    # Checkpointing
    if args.checkpoint_dir:
        overrides['checkpointing'] = overrides.get('checkpointing', {})
        overrides['checkpointing']['checkpoint_dir'] = args.checkpoint_dir

    # Reproducibility
    if args.seed is not None:
        overrides['seed'] = args.seed

    # Merge configs
    if overrides:
        config = merge_configs(config, overrides)

    # Store additional runtime args
    config.gpu = args.gpu
    config.resume = args.resume
    config.test_only = args.test_only

    return config


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)


def print_config(config: Config):
    """Pretty print configuration"""
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(yaml.dump(config.to_dict(), default_flow_style=False, indent=2))
    print("=" * 80)
