from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    embed_dim: int = 64
    spatial_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    num_classes: int = 50  # Will be set based on dataset

@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""
    # Data settings
    data_dir: Path = Path("data")
    max_seq_length: int = 100
    batch_size: int = 32
    num_workers: int = 4
    
    # Training hyperparameters
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Optimizer settings
    optimizer: str = "adam"
    scheduler: str = "cosine"  # cosine or step
    warmup_epochs: int = 5
    
    # Augmentation settings
    rotation_range: float = 15.0
    scale_range: float = 0.1
    translation_range: float = 0.1
    noise_factor: float = 0.05
    
    # Device settings for M3 Pro
    device: str = "mps"  # For Mac M3 Pro with Metal GPU support
    # Will fallback to "cpu" if MPS is not available
    
    # M3 Pro Specific Settings
    batch_size: int = 64  # M3 Pro can handle larger batches
    num_workers: int = 6  # Utilize M3 Pro's multiple cores
    pin_memory: bool = True  # Faster data transfer
    use_mixed_precision: bool = True  # Enable mixed precision for M3
    prefetch_factor: int = 2  # Prefetch batches for better GPU utilization
    
    # Logging and checkpointing
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    experiment_name: str = "sign_language_model"
    log_interval: int = 10  # Log every N batches

@dataclass
class Config:
    """Complete configuration"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()