"""
HAC (Human-Animal-Computer) dataset for action recognition
 
Key differences from EPIC-Kitchens:
- 7 classes instead of 8
- Domains: human, cartoon, animal (instead of D1, D2, D3)
- Uses CSV files for metadata (instead of pickle files)
- Reads full video files (instead of frame folders)
- Different file organization structure
"""

import os
import csv
import numpy as np
import soundfile as sf
import torch
from scipy import signal
from typing import List, Dict, Any, Tuple, Optional

# Import MMAction2 pipeline
import sys

# Add third_party directory to path
_bench_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_third_party_path = os.path.join(_bench_root, 'third_party')
if _third_party_path not in sys.path:
    sys.path.insert(0, _third_party_path)

# HAC uses a modified mmaction with RawFrameDecode that reads from video arrays
# To avoid hook registration conflicts, we temporarily rename mmaction to mmaction_original
# and make mmaction_hac available as mmaction
_saved_mmaction = sys.modules.get('mmaction', None)
_saved_mmaction_datasets = sys.modules.get('mmaction.datasets', None)
_saved_mmaction_pipelines = sys.modules.get('mmaction.datasets.pipelines', None)

# Clear any existing mmaction modules
for key in list(sys.modules.keys()):
    if key.startswith('mmaction'):
        del sys.modules[key]

# Now import from mmaction_hac, but it will be registered as mmaction
import mmaction_hac as mmaction
sys.modules['mmaction'] = mmaction

from mmaction.datasets.pipelines import Compose

# Restore original mmaction if it existed (though we'll keep using mmaction_hac for HAC)
# This prevents conflicts when other modules try to import mmaction
if _saved_mmaction is not None:
    sys.modules['mmaction_original'] = _saved_mmaction

from torch.utils.data import Dataset

# Import imageio for video reading
try:
    import imageio.v3 as iio
except ImportError:
    try:
        import imageio as iio
    except ImportError:
        raise ImportError("imageio is required for HAC dataset. Install with: pip install imageio imageio-ffmpeg")


def get_spectrogram_piece(samples, start_time, end_time, duration, samplerate, training=False):
    """
    Extract and process spectrogram from audio samples

    Args:
        samples: Audio samples
        start_time: Start time in seconds
        end_time: End time in seconds
        duration: Total duration
        samplerate: Sample rate
        training: Whether in training mode (for augmentation)

    Returns:
        Processed spectrogram
    """
    start1 = start_time / duration * len(samples)
    end1 = end_time / duration * len(samples)
    start1 = int(np.round(start1))
    end1 = int(np.round(end1))
    samples = samples[start1:end1]

    resamples = samples[:160000]
    if len(resamples) == 0:
        resamples = np.zeros((160000))
    while len(resamples) < 160000:
        resamples = np.tile(resamples, 10)[:160000]

    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)

    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    interval = 9
    if training is True:
        noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
        spectrogram = spectrogram + noise
        start1 = np.random.choice(256 - interval, (1,))[0]
        spectrogram[start1:(start1 + interval), :] = 0

    return spectrogram


class HACDataset(Dataset):
    """
    HAC (Human-Animal-Computer) dataset for multi-modal action recognition

    Supports RGB video, optical flow, and audio modalities
    Domains: human, cartoon, animal (different action execution agents)

    Note: This class directly inherits from torch.utils.data.Dataset instead of
    BaseMMDGDataset to maintain compatibility with the original HAC dataloader
    implementation.
    """

    # Supported modalities for this dataset
    SUPPORTED_MODALITIES = ['rgb', 'flow', 'audio']

    # Domain names for HAC
    VALID_DOMAINS = ['human', 'cartoon', 'animal']

    def __init__(
        self,
        split: str,
        domains: List[str],
        modalities: List[str],
        data_root: str,
        cfg_video=None,
        cfg_flow=None,
        audio_sample_rate: int = 16000,
        audio_segment_length: int = 160000,
        load_train_split: bool = False,
        **kwargs
    ):
        """
        Args:
            split: 'train' or 'test'
            domains: List of domains ['human', 'cartoon', 'animal']
            modalities: List of modalities to use ['rgb', 'flow', 'audio']
            data_root: Path to HAC directory
            cfg_video: MMAction2 config for video pipeline
            cfg_flow: MMAction2 config for flow pipeline
            audio_sample_rate: Audio sampling rate (default: 16000)
            audio_segment_length: Fixed length for audio segments (default: 160000)
            load_train_split: If True and split=='test', also load train CSV.
        """
        # Store modality-specific parameters
        self.cfg_video = cfg_video
        self.cfg_flow = cfg_flow
        self.audio_sample_rate = audio_sample_rate
        self.audio_segment_length = audio_segment_length
        self.interval = 9  # For audio augmentation
        self.load_train_split = load_train_split
        # Validate domains
        for domain in domains:
            if domain not in self.VALID_DOMAINS:
                raise ValueError(f"Invalid domain '{domain}'. Must be one of {self.VALID_DOMAINS}")

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

        # Store basic parameters
        self.split = split
        self.domains = domains
        self.modalities = modalities
        self.data_root = data_root

        # Create video path base for temporary storage
        self.video_path_base = os.path.join(data_root, 'HAC')
        if not os.path.exists(self.video_path_base):
            os.makedirs(self.video_path_base, exist_ok=True)

        # Validate and load samples
        self._validate_modalities()
        self.samples = self._load_samples()

    def _validate_modalities(self):
        """Validate that requested modalities are supported"""
        for modality in self.modalities:
            if modality not in self.SUPPORTED_MODALITIES:
                raise ValueError(
                    f"Modality '{modality}' not supported for HAC dataset. "
                    f"Supported modalities: {self.SUPPORTED_MODALITIES}"
                )

    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load samples from HAC CSV files

        CSV format: [video_filename, label]
        File location: {data_root}/HAC_Splits/HAC_{split}_only_{domain}.csv

        Returns:
            List of sample dictionaries with keys: 'video_id', 'label', 'domain'
        """
        samples = []
        # Debug: print load_train_split value
        print(f"DEBUG _load_samples: split={self.split}, domains={self.domains}, load_train_split={self.load_train_split}")
        # import pdb; pdb.set_trace()
        for domain in self.domains:
            # Construct CSV path
            csv_path = os.path.join(
                self.data_root,
                'HAC_Splits',
                f'HAC_{self.split}_only_{domain}.csv'
            )

            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # Read CSV file
            with open(csv_path, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    if len(row) >= 2:
                        video_id = row[0]
                        label = int(row[1])

                        samples.append({
                            'video_id': video_id,
                            'label': label,
                            'domain': domain
                        })

            # For test split, also include train samples if source=False
            # This follows the original implementation logic
            if self.split == 'test' and self.load_train_split:
                train_csv_path = os.path.join(
                    self.data_root,
                    'HAC_Splits',
                    f'HAC_train_only_{domain}.csv'
                )
                print(train_csv_path)
                if os.path.exists(train_csv_path):
                    with open(train_csv_path, 'r') as f:
                        csv_reader = csv.reader(f)
                        for row in csv_reader:
                            if len(row) >= 2:
                                video_id = row[0]
                                label = int(row[1])

                                samples.append({
                                    'video_id': video_id,
                                    'label': label,
                                    'domain': domain
                                })

        return samples

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset

        Args:
            index: Sample index

        Returns:
            Dictionary containing:
                - 'rgb': RGB video data (if 'rgb' in modalities)
                - 'flow': Optical flow data (if 'flow' in modalities)
                - 'audio': Audio spectrogram (if 'audio' in modalities)
                - 'label': Action label
                - 'domain': Domain name
        """
        sample = self.samples[index]
        video_id = sample['video_id']
        label = sample['label']
        domain = sample['domain']
        modality_dict = {}
        # result = {
        #     'label': label,
        #     'domain': domain
        # }

        # Base paths
        domain_prefix = domain + '/'
        video_path = os.path.join(self.video_path_base, video_id, video_id + '-')

        # Track frame indices for audio synchronization
        frame_inds = None
        frame_inds_flow = None

        # Load RGB video if requested
        if 'rgb' in self.modalities:
            video_file = os.path.join(self.data_root, domain_prefix, 'videos', video_id)

            if not os.path.exists(video_file):
                raise FileNotFoundError(f"Video file not found: {video_file}")

            # Read video using imageio
            vid = iio.imread(video_file, plugin="pyav")
            frame_num = vid.shape[0]
            start_frame = 0
            end_frame = frame_num - 1

            # Prepare data dict for MMAction2 pipeline
            filename_tmpl = self.cfg_video.data.val.get('filename_tmpl', '{:06}.jpg')
            modality = self.cfg_video.data.val.get('modality', 'RGB')
            start_index = self.cfg_video.data.val.get('start_index', start_frame)

            data = dict(
                frame_dir=video_path,
                total_frames=end_frame - start_frame,
                label=-1,
                start_index=start_index,
                video=vid,
                frame_num=frame_num,
                filename_tmpl=filename_tmpl,
                modality=modality
            )

            # Apply pipeline
            data, frame_inds = self.pipeline_video(data)
            # result['rgb'] = data
            modality_dict['rgb'] = data

        # Load optical flow if requested
        if 'flow' in self.modalities:
            video_file_x = os.path.join(self.data_root, domain_prefix, 'flow', video_id[:-4] + '_flow_x.mp4')
            video_file_y = os.path.join(self.data_root, domain_prefix, 'flow', video_id[:-4] + '_flow_y.mp4')

            if not os.path.exists(video_file_x) or not os.path.exists(video_file_y):
                raise FileNotFoundError(f"Flow files not found: {video_file_x} or {video_file_y}")

            # Read flow videos
            vid_x = iio.imread(video_file_x, plugin="pyav")
            vid_y = iio.imread(video_file_y, plugin="pyav")

            frame_num = vid_x.shape[0]
            start_frame = 0
            end_frame = frame_num - 1

            # Prepare data dict for MMAction2 pipeline
            filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
            modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
            start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame)

            flow = dict(
                frame_dir=video_path,
                total_frames=end_frame - start_frame,
                label=-1,
                start_index=start_index_flow,
                video=vid_x,
                video_y=vid_y,
                frame_num=frame_num,
                filename_tmpl=filename_tmpl_flow,
                modality=modality_flow
            )

            # Apply pipeline
            flow, frame_inds_flow = self.pipeline_flow(flow)
            # result['flow'] = flow
            modality_dict['flow'] = flow

        # Load audio if requested
        if 'audio' in self.modalities:
            audio_path = os.path.join(self.data_root, domain_prefix, 'audio', video_id[:-4] + '.wav')

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Determine time range from video frame indices
            if frame_inds is not None:
                start_time = frame_inds[0] / 24.0  # Assuming 24 fps
                end_time = frame_inds[-1] / 24.0
            elif frame_inds_flow is not None:
                start_time = frame_inds_flow[0] / 24.0
                end_time = frame_inds_flow[-1] / 24.0
            else:
                # Default to full audio
                start_time = 0
                end_time = 10.0  # Default 10 seconds

            # Read audio
            samples, samplerate = sf.read(audio_path)
            duration = len(samples) / samplerate

            # Extract spectrogram
            training = (self.split == 'train')
            spectrogram = get_spectrogram_piece(samples, start_time, end_time, duration, samplerate, training=training)

            # result['audio'] = spectrogram.astype(np.float32)
            modality_dict['audio'] = spectrogram.astype(np.float32)

        return modality_dict, label

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.samples)

    @property
    def num_classes(self) -> int:
        """Return the number of action classes for HAC dataset"""
        return 7