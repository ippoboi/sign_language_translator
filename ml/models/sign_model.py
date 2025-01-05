import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LandmarkEmbedding(nn.Module):
    """
    Converts raw 3D landmarks into learned embeddings
    Input: (batch_size, frames, 21, 3) -> Output: (batch_size, frames, 21, embed_dim)
    """
    def __init__(self, input_dim: int = 3, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, frames, 21, 3)
        # Reshape for linear layer
        batch_size, frames, landmarks, _ = x.shape
        x = x.view(-1, 3)  # Combine all dimensions except last
        x = self.embed(x)  # Apply embedding
        # Restore original dimensions with new embed_dim
        return x.view(batch_size, frames, landmarks, -1)

class SpatialFeatureExtractor(nn.Module):
    """
    Extracts spatial relationships between landmarks using 1D convolutions
    Input: (batch_size, frames, 21, embed_dim) -> Output: (batch_size, frames, spatial_dim)
    """
    def __init__(self, embed_dim: int = 64, spatial_dim: int = 128):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(21, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        self.fc = nn.Linear(64 * embed_dim, spatial_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, frames, 21, embed_dim)
        batch_size, frames, landmarks, embed_dim = x.shape
        
        # Process each frame independently
        x = x.view(-1, landmarks, embed_dim)  # Combine batch and frames
        x = self.conv1d(x)  # Apply 1D convolutions
        x = x.view(batch_size, frames, -1)  # Restore batch and frames
        return self.fc(x)  # Project to final spatial dimension

class TemporalEncoder(nn.Module):
    """
    Processes the temporal dynamics of the sign using LSTM
    Input: (batch_size, frames, spatial_dim) -> Output: (batch_size, frames, hidden_dim)
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, frames, spatial_dim)
        output, _ = self.lstm(x)
        return output  # Shape: (batch_size, frames, hidden_dim*2) due to bidirectional

class AttentionModule(nn.Module):
    """
    Applies attention over the temporal dimension to focus on important frames
    Input: (batch_size, frames, hidden_dim*2) -> Output: (batch_size, hidden_dim*2)
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, frames, hidden_dim*2)
        attention_weights = F.softmax(self.attention(x), dim=1)  # Shape: (batch_size, frames, 1)
        context = torch.sum(attention_weights * x, dim=1)  # Shape: (batch_size, hidden_dim*2)
        return context, attention_weights

class SignLanguageModel(nn.Module):
    """
    Complete sign language recognition model combining all components
    Input: (batch_size, frames, 21, 3) -> Output: (batch_size, num_classes)
    """
    def __init__(
        self, 
        num_classes: int,
        embed_dim: int = 64,
        spatial_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Main components
        self.landmark_embedding = LandmarkEmbedding(input_dim=3, embed_dim=embed_dim)
        self.spatial_features = SpatialFeatureExtractor(embed_dim=embed_dim, spatial_dim=spatial_dim)
        self.temporal_encoder = TemporalEncoder(
            input_dim=spatial_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.attention = AttentionModule(hidden_dim=hidden_dim)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, frames, 21, 3)
        
        # 1. Embed raw landmarks
        x = self.landmark_embedding(x)  # -> (batch_size, frames, 21, embed_dim)
        
        # 2. Extract spatial features
        x = self.spatial_features(x)  # -> (batch_size, frames, spatial_dim)
        
        # 3. Encode temporal dynamics
        x = self.temporal_encoder(x)  # -> (batch_size, frames, hidden_dim*2)
        
        # 4. Apply attention
        x, attention_weights = self.attention(x)  # -> (batch_size, hidden_dim*2)
        
        # 5. Classify
        logits = self.classifier(x)  # -> (batch_size, num_classes)
        
        return logits, attention_weights