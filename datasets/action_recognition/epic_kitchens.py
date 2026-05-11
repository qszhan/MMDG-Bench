"""
EPIC-Kitchens dataset for action recognition
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import signal
from typing import List, Dict, Any, Tuple

# Import MMAction2 pipeline
import sys
import os

# Add third_party directory to path for mmaction
_bench_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_third_party_path = os.path.join(_bench_root, 'third_party')
if _third_party_path not in sys.path:
    sys.path.insert(0, _third_party_path)

from mmaction.datasets.pipelines import Compose

from ..base import BaseMMDGDataset


class EPICKitchensDataset(BaseMMDGDataset):
    """
    EPIC-Kitchens dataset for multi-modal action recognition

    Supports RGB video, optical flow, and audio modalities
    Domains: D1, D2, D3 (different environments)
    """

    # Supported modalities for this dataset
    SUPPORTED_MODALITIES = ['rgb', 'flow', 'audio']

    def __init__(
        self,
        split: str,
        domains: List[str],
        modalities: List[str],
        data_root: str,
        cfg_video=None,
        cfg_flow=None,
        sample_dur: int = 10,
        audio_sample_rate: int = 16000,
        audio_segment_length: int = 160000,
        **kwargs
    ):
        """
        Args:
            split: 'train' or 'test'
            domains: List of domains ['D1', 'D2', 'D3']
            modalities: List of modalities to use
            data_root: Path to EPIC-KITCHENS directory
            cfg_video: MMAction2 config for video pipeline
            cfg_flow: MMAction2 config for flow pipeline
            sample_dur: Duration for audio sampling
            audio_sample_rate: Audio sampling rate
            audio_segment_length: Fixed length for audio segments
        """
        # Store modality-specific parameters
        self.cfg_video = cfg_video
        self.cfg_flow = cfg_flow
        self.sample_dur = sample_dur
        self.audio_sample_rate = audio_sample_rate
        self.audio_segment_length = audio_segment_length
        self.interval = 9  # For audio augmentation

        # Initialize pipelines if configs provided
        self.pipeline_video = None
        self.pipeline_flow = None

        if 'rgb' in modalities and cfg_video is not None:
            if split == 'train':
                self.pipeline_video = Compose(cfg_video.data.train.pipeline)
            else:
                self.pipeline_video = Compose(cfg_video.data.val.pipeline)

        if 'flow' in modalities and cfg_flow is not None:
            if split == 'train':
                self.pipeline_flow = Compose(cfg_flow.data.train.pipeline)
            else:
                self.pipeline_flow = Compose(cfg_flow.data.val.pipeline)

        # Call parent constructor (will call _load_samples)
        super().__init__(
            split=split,
            domains=domains,
            modalities=modalities,
            task_type='action_recognition',
            data_root=data_root,
            **kwargs
        )

    def _validate_modalities(self):
        """Validate that requested modalities are supported"""
        for modality in self.modalities:
            if modality not in self.SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Modality '{modality}' not supported. "
                    f"Choose from {self.SUPPORTED_MODALITIES}"
                )

    def _load_samples(self) -> List[Tuple]:
        """
        Load samples from pickle files

        Returns:
            List of tuples: (video_id, start_frame, stop_frame,
                           start_timestamp, stop_timestamp, label)
        """
        samples = []

        for domain in self.domains:
            pkl_file = os.path.join(
                self.data_root,
                f"{domain}_{self.split}.pkl"
            )

            if not os.path.exists(pkl_file):
                raise FileNotFoundError(f"Dataset file not found: {pkl_file}")

            df = pd.read_pickle(pkl_file)

            for _, row in df.iterrows():
                video_id = f"{domain}/{row['video_id']}"
                sample = (
                    video_id,
                    row['start_frame'],
                    row['stop_frame'],
                    row['start_timestamp'],
                    row['stop_timestamp'],
                    int(row['verb_class'])
                )
                samples.append(sample)

        print(f"Loaded {len(samples)} samples from domains {self.domains} ({self.split})")
        return samples

    def _get_sample_domain(self, sample: Tuple) -> str:
        """Extract domain from sample (e.g., 'D1' from 'D1/P08_01')"""
        video_id = sample[0]
        return video_id.split('/')[0]

    def _get_label(self, sample_idx: int) -> int:
        """Get verb class label"""
        return self.samples[sample_idx][-1]

    def _load_modality_data(self, sample_idx: int, modality: str) -> torch.Tensor:
        """Load data for specific modality"""
        if modality == 'rgb':
            return self._load_rgb(sample_idx)
        elif modality == 'flow':
            return self._load_flow(sample_idx)
        elif modality == 'audio':
            return self._load_audio(sample_idx)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _load_rgb(self, sample_idx: int) -> Dict:
        """Load RGB video frames"""
        sample = self.samples[sample_idx]
        video_path = os.path.join(self.data_root, 'rgb', self.split, sample[0])

        if self.cfg_video is None:
            raise ValueError("cfg_video must be provided to load RGB data")

        filename_tmpl = self.cfg_video.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality = self.cfg_video.data.train.get('modality', 'RGB')
        start_index = self.cfg_video.data.train.get('start_index', int(sample[1]))

        data = dict(
            frame_dir=video_path,
            total_frames=int(sample[2] - sample[1]),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality
        )

        # Apply MMAction2 pipeline
        data = self.pipeline_video(data)
        return data

    def _load_flow(self, sample_idx: int) -> Dict:
        """Load optical flow"""
        sample = self.samples[sample_idx]
        flow_path = os.path.join(self.data_root, 'flow', self.split, sample[0])

        if self.cfg_flow is None:
            raise ValueError("cfg_flow must be provided to load flow data")

        filename_tmpl = self.cfg_flow.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality = self.cfg_flow.data.train.get('modality', 'Flow')
        start_index = self.cfg_flow.data.train.get('start_index', int(np.ceil(sample[1] / 2)))

        flow = dict(
            frame_dir=flow_path,
            total_frames=int((sample[2] - sample[1]) / 2),
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality
        )

        # Apply MMAction2 pipeline
        flow = self.pipeline_flow(flow)
        return flow

    def _load_audio(self, sample_idx: int) -> np.ndarray:
        """Load and process audio spectrogram"""
        sample = self.samples[sample_idx]
        audio_path = os.path.join(
            self.data_root, 'rgb', self.split, f"{sample[0]}.wav"
        )

        # Read audio file
        samples_audio, samplerate = sf.read(audio_path)
        duration = len(samples_audio) / samplerate

        # Parse timestamps
        start_sec = self._parse_timestamp(sample[3])
        stop_sec = self._parse_timestamp(sample[4])

        # Extract audio segment
        # start_sample = int(start_sec / duration * len(samples_audio))
        # end_sample = int(stop_sec / duration * len(samples_audio))
        # samples_audio = samples_audio[start_sample:end_sample]
        start_sample_f = start_sec / duration * len(samples_audio)
        end_sample_f = stop_sec / duration * len(samples_audio)
        start_sample = int(np.round(start_sample_f))
        end_sample = int(np.round(end_sample_f))
        samples_audio = samples_audio[start_sample:end_sample]
        # Resample to fixed length
        resamples = samples_audio[:self.audio_segment_length]
        while len(resamples) < self.audio_segment_length:
            resamples = np.tile(resamples, 10)[:self.audio_segment_length]

        # Clip values
        resamples = np.clip(resamples, -1.0, 1.0)

        # Compute spectrogram
        frequencies, times, spectrogram = signal.spectrogram(
            resamples, samplerate, nperseg=512, noverlap=353
        )
        spectrogram = np.log(spectrogram + 1e-7)

        # Normalize
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = (spectrogram - mean) / (std + 1e-9)

        # Apply augmentation during training
        if self.split == 'train':
            # Add noise
            noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
            spectrogram = spectrogram + noise

            # Mask random time segment
            start_mask = np.random.choice(256 - self.interval, (1,))[0]
            spectrogram[start_mask:(start_mask + self.interval), :] = 0

        return spectrogram.astype(np.float32)

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string (HH:MM:SS) to seconds"""
        parts = timestamp_str.split(':')
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return (hours * 60 + minutes) * 60 + seconds

    def _get_modality_placeholder(self, modality: str) -> torch.Tensor:
        """Return placeholder for missing modality"""
        # Return 0 as placeholder (will be handled by model)
        return torch.tensor(0)

    def get_num_classes(self) -> int:
        """Return number of verb classes in EPIC-Kitchens"""
        # EPIC-Kitchens has 8 verb classes in the domain adaptation split
        return 8

    def get_modality_dims(self) -> Dict[str, Tuple]:
        """Return feature dimensions for each modality"""
        return {
            'rgb': (3, 8, 224, 224),  # (C, T, H, W) - depends on config
            'flow': (2, 8, 224, 224),  # (C, T, H, W) - u,v channels
            'audio': (257, 512)        # (freq_bins, time_bins)
        }