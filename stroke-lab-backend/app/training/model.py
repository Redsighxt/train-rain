"""
PyTorch model architectures for stroke classification

Implements various neural network architectures optimized for stroke invariant
feature classification, including CNNs, Transformers, and hybrid models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import math
import logging

from app.core.config import get_training_settings

training_settings = get_training_settings()
logger = logging.getLogger(__name__)


class StrokeFeatureCNN(nn.Module):
    """
    1D Convolutional Neural Network for stroke feature classification.
    
    Designed to work with pre-computed stroke invariant features.
    Uses 1D convolutions to capture local patterns in feature sequences.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = [128, 256, 512],
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize the CNN model.
        
        Args:
            input_dim: Dimension of input feature vector
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(StrokeFeatureCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Reshape input to treat features as sequence for 1D convolution
        # Input: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        
        layers = []
        current_dim = 1  # Single channel initially
        
        # Convolutional layers
        for i, hidden_dim in enumerate(hidden_dims):
            # 1D Convolution
            conv = nn.Conv1d(
                in_channels=current_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
                bias=not use_batch_norm
            )
            layers.append(conv)
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Max pooling (reduce sequence length)
            if i < len(hidden_dims) - 1:  # Not on last layer
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            
            current_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Calculate the size after convolutions for the classifier
        # This is approximate and might need adjustment based on input_dim
        conv_output_size = self._calculate_conv_output_size(input_dim, len(hidden_dims))
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"StrokeFeatureCNN initialized: input_dim={input_dim}, num_classes={num_classes}")
    
    def _calculate_conv_output_size(self, input_size: int, num_pools: int) -> int:
        """Calculate output size after convolutions and pooling."""
        size = input_size
        for _ in range(num_pools - 1):  # Pooling layers
            size = size // 2
        return size
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Reshape for 1D convolution: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class StrokeSequenceTransformer(nn.Module):
    """
    Transformer model for stroke sequence classification.
    
    Uses self-attention to capture dependencies in stroke point sequences.
    Designed for variable-length sequences with padding.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout_rate: float = 0.1,
        max_seq_length: int = 200,
        use_positional_encoding: bool = True
    ):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim: Dimension of input features per sequence element
            num_classes: Number of output classes
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout_rate: Dropout probability
            max_seq_length: Maximum sequence length
            use_positional_encoding: Whether to use positional encoding
        """
        super(StrokeSequenceTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        if use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"StrokeSequenceTransformer initialized: d_model={d_model}, num_layers={num_layers}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            src_key_padding_mask: Mask for padded positions
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size, seq_length, _ = x.size()
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Update padding mask for CLS token
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Use CLS token for classification
        cls_output = x[:, 0]  # First token (CLS)
        
        # Classification
        output = self.classifier(cls_output)
        
        return output
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for sequences.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            
        Returns:
            Padding mask of shape (batch_size, seq_length)
        """
        # Assume zero padding - mask positions where all features are zero
        return torch.all(x == 0, dim=-1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class HybridStrokeClassifier(nn.Module):
    """
    Hybrid model combining CNN and Transformer for stroke classification.
    
    Uses CNN for local feature extraction and Transformer for global dependencies.
    """
    
    def __init__(
        self,
        feature_dim: int,
        sequence_dim: int,
        num_classes: int,
        cnn_hidden_dims: list = [128, 256],
        transformer_d_model: int = 256,
        transformer_layers: int = 4,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the hybrid model.
        
        Args:
            feature_dim: Dimension of pre-computed features
            sequence_dim: Dimension of sequence features per point
            num_classes: Number of output classes
            cnn_hidden_dims: Hidden dimensions for CNN branch
            transformer_d_model: Model dimension for Transformer branch
            transformer_layers: Number of Transformer layers
            dropout_rate: Dropout probability
        """
        super(HybridStrokeClassifier, self).__init__()
        
        self.feature_dim = feature_dim
        self.sequence_dim = sequence_dim
        self.num_classes = num_classes
        
        # CNN branch for features
        self.cnn_branch = StrokeFeatureCNN(
            input_dim=feature_dim,
            num_classes=cnn_hidden_dims[-1],  # Output features, not classes
            hidden_dims=cnn_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        # Replace CNN classifier with feature extractor
        self.cnn_branch.classifier = nn.Identity()
        
        # Transformer branch for sequences
        self.transformer_branch = StrokeSequenceTransformer(
            input_dim=sequence_dim,
            num_classes=transformer_d_model,  # Output features, not classes
            d_model=transformer_d_model,
            num_layers=transformer_layers,
            dropout_rate=dropout_rate
        )
        
        # Replace Transformer classifier with feature extractor
        self.transformer_branch.classifier = nn.Identity()
        
        # Fusion layer
        fusion_input_dim = cnn_hidden_dims[-1] + transformer_d_model
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_input_dim // 2, fusion_input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_input_dim // 4, num_classes)
        )
        
        logger.info(f"HybridStrokeClassifier initialized with fusion dim: {fusion_input_dim}")
    
    def forward(
        self,
        features: torch.Tensor,
        sequences: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Pre-computed features (batch_size, feature_dim)
            sequences: Point sequences (batch_size, seq_length, sequence_dim)
            sequence_mask: Padding mask for sequences
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        # CNN branch
        cnn_features = self.cnn_branch(features)
        
        # Transformer branch
        transformer_features = self.transformer_branch(sequences, sequence_mask)
        
        # Fusion
        fused_features = torch.cat([cnn_features, transformer_features], dim=1)
        output = self.fusion_classifier(fused_features)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block for deep CNN architectures."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class DeepStrokeCNN(nn.Module):
    """
    Deep CNN with residual connections for stroke classification.
    
    Uses residual blocks to enable training of very deep networks.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_blocks: int = 4,
        base_channels: int = 64,
        dropout_rate: float = 0.3
    ):
        super(DeepStrokeCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(1, base_channels, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(base_channels)
        
        # Residual blocks
        layers = []
        current_channels = base_channels
        
        for i in range(num_blocks):
            out_channels = base_channels * (2 ** i)
            layers.append(ResidualBlock(current_channels, out_channels))
            
            if i < num_blocks - 1:  # Add pooling except for last block
                layers.append(nn.MaxPool1d(2))
            
            current_channels = out_channels
        
        self.residual_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(current_channels, current_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(current_channels // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for 1D convolution
        x = x.unsqueeze(1)
        
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Residual blocks
        x = self.residual_layers(x)
        
        # Global pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_model(
    model_type: str,
    input_dim: int,
    num_classes: int,
    model_config: Optional[Dict[str, Any]] = None
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ("cnn", "transformer", "hybrid", "deep_cnn")
        input_dim: Input dimension
        num_classes: Number of classes
        model_config: Optional model configuration
        
    Returns:
        Initialized PyTorch model
    """
    config = model_config or {}
    
    if model_type.lower() == "cnn":
        return StrokeFeatureCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            **config
        )
    
    elif model_type.lower() == "transformer":
        return StrokeSequenceTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            **config
        )
    
    elif model_type.lower() == "deep_cnn":
        return DeepStrokeCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            **config
        )
    
    elif model_type.lower() == "hybrid":
        # Hybrid model requires both feature and sequence dimensions
        feature_dim = config.get("feature_dim", input_dim)
        sequence_dim = config.get("sequence_dim", 5)  # x, y, pressure, velocity, time
        
        return HybridStrokeClassifier(
            feature_dim=feature_dim,
            sequence_dim=sequence_dim,
            num_classes=num_classes,
            **{k: v for k, v in config.items() if k not in ["feature_dim", "sequence_dim"]}
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get the size of a model in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)  # Convert to MB


def summarize_model(model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Size of input tensor (excluding batch dimension)
        
    Returns:
        Dictionary containing model summary information
    """
    total_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    
    # Try to compute FLOPs (requires a forward pass)
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_size).to(device)
        
        # Count operations (simplified)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        
        flops = "Not computed"
    except Exception:
        flops = "Error computing"
    
    summary = {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": total_params,
        "model_size_mb": model_size_mb,
        "input_size": input_size,
        "estimated_flops": flops
    }
    
    return summary


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    feature_dim = 49  # Example feature dimension
    sequence_dim = 5  # x, y, pressure, velocity, time
    num_classes = 26  # A-Z
    
    models = {
        "CNN": create_model("cnn", feature_dim, num_classes),
        "Transformer": create_model("transformer", sequence_dim, num_classes),
        "Deep CNN": create_model("deep_cnn", feature_dim, num_classes),
        "Hybrid": create_model("hybrid", feature_dim, num_classes, {
            "feature_dim": feature_dim,
            "sequence_dim": sequence_dim
        })
    }
    
    for name, model in models.items():
        print(f"\n{name} Model:")
        if name == "Hybrid":
            summary = summarize_model(model, (feature_dim,))  # Simplified for hybrid
        elif "CNN" in name:
            summary = summarize_model(model, (feature_dim,))
        else:  # Transformer
            summary = summarize_model(model, (200, sequence_dim))  # seq_len=200
        
        print(f"  Parameters: {summary['total_parameters']:,}")
        print(f"  Size: {summary['model_size_mb']:.2f} MB")
