import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import wandb
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple

from .config import Config
from ..models.sign_model import SignLanguageModel
from ..dataset.sign_dataset import SignLanguageDataset, SignLanguageTransform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignLanguageTrainer:
    def __init__(self, config: Config):
        self.config = config
        self._setup_directories()
        self._setup_logging()
        
        # Initialize metrics tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def _setup_directories(self):
        """Create necessary directories"""
        self.config.training.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_logging(self):
        """Initialize wandb logging"""
        wandb.init(
            project=self.config.training.experiment_name,
            config={
                "model": vars(self.config.model),
                "training": vars(self.config.training)
            }
        )
        
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders"""
        # Create transforms
        transform = SignLanguageTransform(
            rotation_range=self.config.training.rotation_range,
            scale_range=self.config.training.scale_range,
            translation_range=self.config.training.translation_range,
            noise_factor=self.config.training.noise_factor
        )
        
        # Create datasets
        train_dataset = SignLanguageDataset(
            data_dir=self.config.training.data_dir,
            split='train',
            transform=transform,
            max_seq_length=self.config.training.max_seq_length
        )
        
        val_dataset = SignLanguageDataset(
            data_dir=self.config.training.data_dir,
            split='val',
            transform=None,  # No augmentation for validation
            max_seq_length=self.config.training.max_seq_length
        )
        
        # Update number of classes if not set
        if self.config.model.num_classes == 50:  # Default value
            self.config.model.num_classes = len(train_dataset.label_map)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _setup_device(self) -> str:
        """Setup the appropriate device for training"""
        if self.config.training.device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif self.config.training.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _create_model(self) -> SignLanguageModel:
        """Initialize model"""
        # Update device based on availability
        self.config.training.device = self._setup_device()
        logger.info(f"Using device: {self.config.training.device}")
        
        # Enable Metal Performance Shaders (MPS) optimizations if available
        if self.config.training.device == "mps":
            # Enable async GPU operations
            torch.mps.set_per_process_memory_fraction(0.7)  # Prevent memory spikes
            torch.mps.empty_cache()  # Clear GPU memory before starting
            
            if self.config.training.use_mixed_precision:
                # Enable automatic mixed precision for MPS
                torch.set_float32_matmul_precision('medium')  # Balance accuracy and speed
        model = SignLanguageModel(
            num_classes=self.config.model.num_classes,
            embed_dim=self.config.model.embed_dim,
            spatial_dim=self.config.model.spatial_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_layers,
            dropout=self.config.model.dropout
        )
        return model.to(self.config.training.device)
    
    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Initialize optimizer"""
        if self.config.training.optimizer.lower() == "adam":
            optimizer = Adam(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            optimizer = SGD(
                model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        return optimizer
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Initialize learning rate scheduler"""
        if self.config.training.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.epochs,
                eta_min=1e-6
            )
        else:
            scheduler = StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        return scheduler
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Dict[str, float]:
        """Train one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            # Move to device
            data, target = data.to(self.config.training.device), target.to(self.config.training.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Compute metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            if batch_idx % self.config.training.log_interval == 0:
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        # Compute epoch metrics
        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_acc': 100. * correct / total
        }
        
        return metrics
    
    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in val_loader:
            # Move to device
            data, target = data.to(self.config.training.device), target.to(self.config.training.device)
            
            # Forward pass
            logits, _ = model(data)
            loss = criterion(logits, target)
            
            # Compute metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        # Compute validation metrics
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_acc': 100. * correct / total
        }
        
        return metrics
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.config.training.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if current model is best
        if is_best:
            best_model_path = self.config.training.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)
    
    def train(self):
        """Main training loop"""
        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders()
        
        # Create model, optimizer, scheduler, and criterion
        model = self._create_model()
        optimizer = self._create_optimizer(model)
        scheduler = self._create_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()
        
        logger.info("Starting training...")
        for epoch in range(self.config.training.epochs):
            # Train epoch
            train_metrics = self._train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_metrics = self._validate(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            wandb.log(metrics, step=epoch)
            
            # Check for improvement
            is_best = val_metrics['val_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['val_acc']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self._save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - "
                f"Train Acc: {train_metrics['train_acc']:.2f}% - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Val Acc: {val_metrics['val_acc']:.2f}%"
            )
        
        wandb.finish()
        logger.info("Training completed!")