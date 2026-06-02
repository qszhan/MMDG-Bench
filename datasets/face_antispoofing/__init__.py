"""
Face Anti-Spoofing Datasets

Supports both:
1. Pre-extracted features (MMDG_Bench format)
2. Raw images (Flex-Modal-FAS format)

Public API:
- get_fas_dataset(config, split, domains): High-level interface with config object
- get_fas_dataset_raw(split, domains, modalities, data_root, use_extracted_features, source): Low-level interface
- CASIACeFADataset, CASIASURFDataset, WMCADataset: Dataset classes supporting both modes
"""

# Import the actual dataset classes from their modules
from .casia_cefa import CASIACeFADataset
from .casia_surf import CASIASURFDataset
from .wmca import WMCADataset


__all__ = [
    'get_fas_dataset',
    'get_fas_dataset_raw',
    'CASIACeFADataset',
    'CASIASURFDataset',
    'WMCADataset',
]



def get_fas_dataset_raw(split, domains, modalities, data_root, use_extracted_features=True, source=True):
    """
    Low-level dataset factory for FAS datasets (accepts individual parameters)

    Supports two modes:
    1. use_extracted_features=True: Load pre-extracted ViT features (MMDG_Bench format)
    2. use_extracted_features=False: Load raw images (Flex-Modal-FAS format)

    Args:
        split: 'train', 'val', or 'test'
        domains: List of domain names (e.g., ['CeFA'], ['SURF'], ['WMCA'])
        modalities: List of modalities (e.g., ['rgb', 'depth', 'ir'])
        data_root: Root directory for data
            - If use_extracted_features=True: Path to pre-extracted features
            - If use_extracted_features=False: Path to Flex-Modal-FAS data
        use_extracted_features: Whether to use pre-extracted features or raw images
        source: Whether this is a source domain (True) or target domain (False)
                - (split='train', source=True): Load train features/data
                - (split='val', source=True): Load val features/data
                - (split='test', source=False): Load train features/data (test on target)
                - (split='test', source=True): Load val features/data (test on source)

    Returns:
        dataset: PyTorch Dataset instance
            - If use_extracted_features=True: Returns (modality_dict, label) tuples with features [768]
            - If use_extracted_features=False: Returns (modality_dict, label) tuples with images [3, 224, 224]
    """

    # Validate inputs
    if len(domains) != 1:
        raise ValueError(
            f"Currently only single domain per dataset is supported. Got {domains}"
        )

     

    domain = domains[0]

    # Dataset class mapping
    dataset_map = {
        'CeFA': CASIACeFADataset,
        'SURF': CASIASURFDataset,
        'WMCA': WMCADataset
    }

    if domain not in dataset_map:
        raise ValueError(
            f"Unknown domain: {domain}. Supported domains: {list(dataset_map.keys())}"
        )

    dataset_class = dataset_map[domain]

    mode_str = "pre-extracted features" if use_extracted_features else "raw images"
    print(f"Creating {domain} dataset ({mode_str}): split={split}, source={source}")

    # Create dataset instance (handles both modes internally)
    dataset = dataset_class(
        split=split,
        domains=domains,
        modalities=modalities,
        data_root=data_root,
        use_extracted_features=use_extracted_features,
        source=source
    )

    print(f"  Dataset created: {len(dataset)} samples")

    return dataset


def get_fas_dataset(config, split, domains, source=True):
    """
    High-level dataset factory (accepts config object)

    This is a convenience wrapper that extracts parameters from the config object.
    Maintains compatibility with existing code.

    Args:
        config: Config object with dataset settings
        split: 'train', 'val', or 'test'
        domains: List of domain names (e.g., ['CeFA'], ['SURF'], ['WMCA'])
        source: Whether this is a source domain (True) or target domain (False)

    Returns:
        dataset: FAS dataset instance
    """
    return get_fas_dataset_raw(
        split=split,
        domains=domains,
        modalities=config.dataset.modalities,
        data_root=config.dataset.data_root,
        use_extracted_features=config.dataset.use_extracted_features,
        source=source
    )

