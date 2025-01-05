#!/usr/bin/env python3
import argparse
import logging
import yaml
from pathlib import Path
import torch
import warnings

from training.config import Config
from training.trainer import SignLanguageTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing the dataset')
    parser.add_argument('--checkpoint', type=str,
                      help='Path to checkpoint to resume training from')
    parser.add_argument('--experiment_name', type=str, default='sign_language_model',
                      help='Name of the experiment for logging')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    
    return parser.parse_args()

def load_config(config_path: str) -> Config:
    """Load and update configuration"""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Create config instance
    config = Config()
    
    # Update model config
    for key, value in config_dict.get('model', {}).items():
        setattr(config.model, key, value)
    
    # Update training config
    for key, value in config_dict.get('training', {}).items():
        setattr(config.training, key, value)
    
    return config

def setup_environment(debug: bool = False):
    """Setup training environment"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        warnings.filterwarnings('always')  # Show all warnings
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    else:
        warnings.filterwarnings('ignore')  # Ignore warnings in production mode

def load_checkpoint(checkpoint_path: str, trainer: SignLanguageTrainer):
    """Load checkpoint and update trainer state"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Update trainer state
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    trainer.start_epoch = checkpoint['epoch'] + 1
    trainer.best_val_acc = checkpoint.get('best_val_acc', 0)
    
    logger.info(f"Resumed from epoch {trainer.start_epoch}")

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Setup environment
    setup_environment(args.debug)
    
    # Load and update config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Update config with command line arguments
    config.training.data_dir = Path(args.data_dir)
    config.training.experiment_name = args.experiment_name
    
    # Create trainer
    trainer = SignLanguageTrainer(config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        load_checkpoint(args.checkpoint, trainer)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception("Error occurred during training")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

if __name__ == "__main__":
    main()