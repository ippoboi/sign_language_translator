import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import logging
from typing import Optional, Dict, List, Union
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageDataset(Dataset):
    """
    Dataset class for Sign Language Recognition
    Loads preprocessed landmark data from .npz files and handles multiple hands,
    variable sequence lengths, and data augmentation.
    
    Args:
        data_dir (str): Root directory containing the dataset
        split (str): Dataset split ('train', 'val', 'test')
        transform (Optional[object]): Transform to apply to landmarks
        max_seq_length (Optional[int]): Maximum sequence length for padding
        cache_size (int): Number of samples to cache in memory
        normalize (bool): Whether to normalize landmarks
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[object] = None,
        max_seq_length: Optional[int] = None,
        cache_size: int = 1000,
        normalize: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.cache_size = cache_size
        
        # Data cache
        self._cache: Dict[int, tuple] = {}
        
        # Load samples and label mapping
        self.samples = self._load_samples()
        self.label_map = self._load_label_map()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} set")
        logger.info(f"Found {len(self.label_map)} unique signs")

    def _load_samples(self) -> List[Path]:
        """Load all .npz files from the processed/{split} directory"""
        split_dir = self.data_dir / 'processed' / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
            
        samples = list(split_dir.glob('**/*.npz'))
        if not samples:
            raise ValueError(f"No samples found in {split_dir}")
            
        return samples

    def _load_label_map(self) -> Dict[str, int]:
        """Load label mapping from preprocessing stats"""
        stats_path = self.data_dir / 'interim' / 'preprocessing_stats.json'
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            label_map = stats.get('label_map', {})
            if not label_map:
                raise ValueError("Empty label map found")
            return label_map
        except FileNotFoundError:
            raise FileNotFoundError(f"Label map not found: {stats_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in label map: {stats_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to have zero mean and unit variance
        landmarks shape: (frames, num_hands, 21, 3) or (frames, 21, 3)
        """
        if len(landmarks.shape) == 4:
            # Multiple hands
            mean = np.mean(landmarks, axis=(0, 1, 2), keepdims=True)
            std = np.std(landmarks, axis=(0, 1, 2), keepdims=True)
        else:
            # Single hand
            mean = np.mean(landmarks, axis=(0, 1), keepdims=True)
            std = np.std(landmarks, axis=(0, 1), keepdims=True)
            
        std = np.where(std == 0, 1e-6, std)  # Prevent division by zero
        return (landmarks - mean) / std

    def _validate_sequence_length(self, sequence: np.ndarray, min_length: int = 1) -> np.ndarray:
        """Validate sequence length and optionally trim"""
        if len(sequence) < min_length:
            raise ValueError(f"Sequence length {len(sequence)} is less than minimum {min_length}")
            
        if self.max_seq_length and len(sequence) > self.max_seq_length:
            # Randomly sample max_seq_length frames if sequence is too long
            indices = sorted(random.sample(range(len(sequence)), self.max_seq_length))
            return sequence[indices]
        return sequence

    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Pad sequence to max_seq_length if specified"""
        if self.max_seq_length is None or len(sequence) >= self.max_seq_length:
            return sequence
            
        pad_length = self.max_seq_length - len(sequence)
        pad_shape = (pad_length,) + sequence.shape[1:]
        padding = np.zeros(pad_shape)
        return np.concatenate([sequence, padding])

    def _process_multiple_hands(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Process landmarks when multiple hands are present
        Strategy: Take primary hand (first hand) if multiple present
        """
        if len(landmarks.shape) == 3:  # Single hand
            return np.expand_dims(landmarks, axis=1)
            
        # Already in correct shape (frames, num_hands, 21, 3)
        return landmarks[:, :1, :, :]  # Take primary hand

    def __getitem__(self, idx: int) -> tuple:
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]

        # Load and process data
        try:
            sample_path = self.samples[idx]
            if not sample_path.exists():
                raise FileNotFoundError(f"Sample not found: {sample_path}")

            data = np.load(sample_path)
            landmarks = data['landmarks']
            label = data['label']

            # Process landmarks
            landmarks = self._process_multiple_hands(landmarks)
            landmarks = self._validate_sequence_length(landmarks)
            
            if self.normalize:
                landmarks = self._normalize_landmarks(landmarks)
                
            if self.max_seq_length:
                landmarks = self._pad_sequence(landmarks)

            # Apply transforms if any
            if self.transform:
                landmarks = self.transform(landmarks)

            # Convert to tensors
            landmarks = torch.FloatTensor(landmarks)
            label = torch.LongTensor([label])[0]

            # Update cache
            if len(self._cache) < self.cache_size:
                self._cache[idx] = (landmarks, label)

            return landmarks, label

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise

class SignLanguageTransform:
    """
    Transform class for data augmentation with improved robustness
    """
    def __init__(
        self,
        rotation_range: float = 15,
        scale_range: float = 0.1,
        translation_range: float = 0.1,
        noise_factor: float = 0.05
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_factor = noise_factor

    def _get_rotation_matrix(self, angle: float) -> np.ndarray:
        """Generate 3D rotation matrix around y-axis (assuming front-facing camera)"""
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply random rotation, scaling, translation, and noise to landmarks
        landmarks: numpy array of shape (frames, num_hands, 21, 3)
        """
        # Make a copy to avoid modifying original data
        transformed = landmarks.copy()

        # Random rotation
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        rotation_matrix = self._get_rotation_matrix(angle)

        # Random scaling
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)

        # Random translation
        translation = np.random.uniform(
            -self.translation_range,
            self.translation_range,
            size=3
        )

        # Random noise
        noise = np.random.normal(
            0,
            self.noise_factor,
            transformed.shape
        )

        # Apply transformations
        for i in range(len(transformed)):
            # Apply rotation
            transformed[i] = transformed[i] @ rotation_matrix
            # Apply scaling
            transformed[i] *= scale
            # Apply translation
            transformed[i] += translation
            # Add noise
            transformed[i] += noise[i]

        return transformed