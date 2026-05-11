"""
WMCA HDF5 Data Loader

Loads WMCA face anti-spoofing data from HDF5 format.
Combines RGB from RGB station (true color) with Depth+IR from CDIT station.

HDF5 Structure:
    - preprocessed-face-station_RGB/date/video.hdf5
        Each frame: [3, 128, 128] - True RGB
    - preprocessed-face-station_CDIT/date/video.hdf5
        Each frame: [4, 128, 128] - Color + Depth + IR + Thermal

Protocol File Format:
    date/video.hdf5 frame_idx label
    Example: 12.02.18/005_03_000_0_01.hdf5 12 0
"""

import os
import cv2
import numpy as np
import h5py
from torch.utils.data import Dataset


class Spoofing_train_WMCA_HDF5(Dataset):
    """
    WMCA HDF5 training dataset (with data augmentation)

    Args:
        list_file: Protocol file path (format: hdf5_path frame_idx label)
        root_dir: Root directory containing WMCA folder
        transform: Data augmentation transforms (from Load_FAS_MultiModal)
    """

    def __init__(self, list_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Two stations
        self.rgb_dir = os.path.join(root_dir, 'WMCA', 'preprocessed-face-station_RGB')
        self.cdit_dir = os.path.join(root_dir, 'WMCA', 'preprocessed-face-station_CDIT')

        # Load protocol file
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue

                rel_path = parts[0]  # e.g., "12.02.18/005_03_000_0_01.hdf5"
                frame_idx = int(parts[1])  # e.g., 12
                label = int(parts[2])  # 0=live, 1=spoof

                self.samples.append((rel_path, frame_idx, label))

        print(f"[WMCA HDF5] Loaded {len(self.samples)} samples from {list_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, frame_idx, label = self.samples[idx]

        # Full HDF5 paths
        rgb_path = os.path.join(self.rgb_dir, rel_path)
        cdit_path = os.path.join(self.cdit_dir, rel_path)

        # Read from RGB station (true 3-channel RGB)
        try:
            with h5py.File(rgb_path, 'r') as f:
                frame_key = f'Frame_{frame_idx}'
                rgb_array = f[frame_key]['array'][:]  # [3, 128, 128]
        except Exception as e:
            # print(f"[ERROR] Failed to read RGB from {rgb_path}, frame={frame_idx}: {e}")
            rgb_array = np.zeros((3, 128, 128), dtype=np.uint8)

        # Read from CDIT station (Depth and IR)
        try:
            with h5py.File(cdit_path, 'r') as f:
                frame_key = f'Frame_{frame_idx}'
                cdit_array = f[frame_key]['array'][:]  # [4, 128, 128]
        except Exception as e:
            # print(f"[ERROR] Failed to read CDIT from {cdit_path}, frame={frame_idx}: {e}")
            cdit_array = np.zeros((4, 128, 128), dtype=np.uint8)

        # Extract modalities
        # RGB: from RGB station (channels already in RGB order)
        image_x = rgb_array.transpose(1, 2, 0)  # [3, 128, 128] -> [128, 128, 3]
        image_x = cv2.resize(image_x, (224, 224))  # Resize to 224x224 for ViT

        # Depth: from CDIT channel 1
        depth_gray = cdit_array[1]  # [128, 128]
        image_x_depth = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2RGB)  # [128, 128, 3]
        image_x_depth = cv2.resize(image_x_depth, (224, 224))  # Resize to 224x224

        # IR: from CDIT channel 2
        ir_gray = cdit_array[2]  # [128, 128]
        image_x_ir = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)  # [128, 128, 3]
        image_x_ir = cv2.resize(image_x_ir, (224, 224))  # Resize to 224x224

        # Create map_x1 (depth map for supervision)
        # For live (label=1): map should be all ones
        # For spoof (label=0): map should be all zeros
        if label == 1:  # live
            map_x1 = np.ones((28, 28))
        else:  # spoof
            map_x1 = np.zeros((28, 28))

        # Create sample dict (compatible with Flex-Modal-FAS format)
        sample = {
            'image_x': image_x,
            'image_x_depth': image_x_depth,
            'image_x_ir': image_x_ir,
            'spoofing_label': label,
            'map_x1': map_x1
        }

        # Apply transforms (augmentation)
        if self.transform:
            sample = self.transform(sample)

        return sample


class Spoofing_valtest_WMCA_HDF5(Dataset):
    """
    WMCA HDF5 validation/test dataset (without data augmentation)

    Args:
        list_file: Protocol file path (format: hdf5_path frame_idx label)
        root_dir: Root directory containing WMCA folder
        transform: Normalization transforms (no augmentation)
    """

    def __init__(self, list_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Two stations
        self.rgb_dir = os.path.join(root_dir, 'WMCA', 'preprocessed-face-station_RGB')
        self.cdit_dir = os.path.join(root_dir, 'WMCA', 'preprocessed-face-station_CDIT')

        # Load protocol file
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue

                rel_path = parts[0]
                frame_idx = int(parts[1])
                label = int(parts[2])

                self.samples.append((rel_path, frame_idx, label))

        print(f"[WMCA HDF5] Loaded {len(self.samples)} samples from {list_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, frame_idx, label = self.samples[idx]

        # Full HDF5 paths
        rgb_path = os.path.join(self.rgb_dir, rel_path)
        cdit_path = os.path.join(self.cdit_dir, rel_path)

        # Read from RGB station
        try:
            with h5py.File(rgb_path, 'r') as f:
                frame_key = f'Frame_{frame_idx}'
                rgb_array = f[frame_key]['array'][:]  # [3, 128, 128]
        except Exception as e:
            # print(f"[ERROR] Failed to read RGB from {rgb_path}, frame={frame_idx}: {e}")
            rgb_array = np.zeros((3, 128, 128), dtype=np.uint8)

        # Read from CDIT station
        try:
            with h5py.File(cdit_path, 'r') as f:
                frame_key = f'Frame_{frame_idx}'
                cdit_array = f[frame_key]['array'][:]  # [4, 128, 128]
        except Exception as e:
            # print(f"[ERROR] Failed to read CDIT from {cdit_path}, frame={frame_idx}: {e}")
            cdit_array = np.zeros((4, 128, 128), dtype=np.uint8)

        # Extract modalities
        image_x = rgb_array.transpose(1, 2, 0)  # [3, 128, 128] -> [128, 128, 3]
        image_x = cv2.resize(image_x, (224, 224))  # Resize to 224x224 for ViT

        depth_gray = cdit_array[1]
        image_x_depth = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2RGB)
        image_x_depth = cv2.resize(image_x_depth, (224, 224))  # Resize to 224x224

        ir_gray = cdit_array[2]
        image_x_ir = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2RGB)
        image_x_ir = cv2.resize(image_x_ir, (224, 224))  # Resize to 224x224

        # Create map_x1 (depth map for supervision)
        if label == 1:  # live
            map_x1 = np.ones((28, 28))
        else:  # spoof
            map_x1 = np.zeros((28, 28))

        # Create sample dict
        sample = {
            'image_x': image_x,
            'image_x_depth': image_x_depth,
            'image_x_ir': image_x_ir,
            'spoofing_label': label,
            'map_x1': map_x1
        }

        # Apply transforms (normalization only, no augmentation)
        if self.transform:
            sample = self.transform(sample)

        return sample