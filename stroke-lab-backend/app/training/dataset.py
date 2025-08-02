"""
PyTorch Dataset classes for stroke data training

Provides dataset implementations for loading and preprocessing stroke data
for machine learning training with various augmentation strategies.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import random

from app.db.models import StrokeData
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class StrokeFeatureDataset(Dataset):
    """
    PyTorch Dataset for stroke feature vectors.
    
    Loads pre-computed stroke features and provides them for training.
    Supports various data augmentation techniques specific to stroke data.
    """
    
    def __init__(
        self,
        stroke_data: List[StrokeData],
        labels: Optional[List[str]] = None,
        feature_scaler: Optional[StandardScaler] = None,
        label_encoder: Optional[LabelEncoder] = None,
        augment: bool = False,
        augment_prob: float = 0.5,
        normalize_features: bool = True
    ):
        """
        Initialize the stroke dataset.
        
        Args:
            stroke_data: List of StrokeData objects
            labels: Optional list of specific labels to include
            feature_scaler: Optional pre-fitted StandardScaler
            label_encoder: Optional pre-fitted LabelEncoder
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying augmentation to each sample
            normalize_features: Whether to normalize feature vectors
        """
        self.stroke_data = stroke_data
        self.augment = augment
        self.augment_prob = augment_prob
        self.normalize_features = normalize_features
        
        # Filter by labels if specified
        if labels:
            self.stroke_data = [
                stroke for stroke in stroke_data 
                if stroke.label in labels
            ]
        
        if len(self.stroke_data) == 0:
            raise ValueError("No stroke data available for the specified criteria")
        
        # Prepare features and labels
        self.features, self.labels_raw = self._extract_features_and_labels()
        
        # Setup label encoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels_raw)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(self.labels_raw)
        
        # Setup feature scaler
        if feature_scaler is None and normalize_features:
            self.feature_scaler = StandardScaler()
            self.features = self.feature_scaler.fit_transform(self.features)
        elif feature_scaler is not None:
            self.feature_scaler = feature_scaler
            self.features = self.feature_scaler.transform(self.features)
        else:
            self.feature_scaler = None
        
        # Convert to torch tensors
        self.features = torch.FloatTensor(self.features)
        self.encoded_labels = torch.LongTensor(self.encoded_labels)
        
        logger.info(f"Dataset initialized with {len(self)} samples, {len(self.label_encoder.classes_)} classes")
        logger.info(f"Feature dimension: {self.features.shape[1]}")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
    
    def _extract_features_and_labels(self) -> Tuple[np.ndarray, List[str]]:
        """Extract feature vectors and labels from stroke data."""
        features = []
        labels = []
        
        for stroke in self.stroke_data:
            # Get feature vector (computed property from StrokeData model)
            feature_vector = stroke.feature_vector
            
            if len(feature_vector) > 0:
                features.append(feature_vector)
                labels.append(stroke.label)
        
        if len(features) == 0:
            raise ValueError("No valid feature vectors found in stroke data")
        
        # Ensure all feature vectors have the same length
        max_length = max(len(f) for f in features)
        
        padded_features = []
        for feature_vector in features:
            if len(feature_vector) < max_length:
                # Pad with zeros
                padded = np.pad(feature_vector, (0, max_length - len(feature_vector)), 'constant')
            else:
                padded = np.array(feature_vector[:max_length])
            padded_features.append(padded)
        
        return np.array(padded_features), labels
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        features = self.features[idx].clone()
        label = self.encoded_labels[idx]
        
        # Apply augmentation if enabled
        if self.augment and random.random() < self.augment_prob:
            features = self._augment_features(features)
        
        return features, label
    
    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to feature vector.
        
        Args:
            features: Original feature vector
            
        Returns:
            Augmented feature vector
        """
        # Make a copy to avoid modifying original
        augmented = features.clone()
        
        # Gaussian noise augmentation
        if random.random() < 0.3:
            noise_std = 0.05 * torch.std(augmented)
            noise = torch.normal(0, noise_std, augmented.shape)
            augmented += noise
        
        # Scale augmentation (simulate different writing speeds)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)
            # Apply to geometric features (first few components)
            augmented[:10] *= scale_factor
        
        # Feature dropout (randomly zero out some features)
        if random.random() < 0.2:
            dropout_prob = 0.1
            mask = torch.rand(augmented.shape) > dropout_prob
            augmented *= mask.float()
        
        return augmented
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.encoded_labels.numpy()),
            y=self.encoded_labels.numpy()
        )
        
        return torch.FloatTensor(class_weights)
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get mapping from encoded labels to original string labels."""
        return {
            i: label for i, label in enumerate(self.label_encoder.classes_)
        }
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dataset features."""
        features_np = self.features.numpy()
        
        return {
            "num_samples": len(self),
            "num_features": features_np.shape[1],
            "num_classes": len(self.label_encoder.classes_),
            "feature_mean": np.mean(features_np, axis=0).tolist(),
            "feature_std": np.std(features_np, axis=0).tolist(),
            "feature_min": np.min(features_np, axis=0).tolist(),
            "feature_max": np.max(features_np, axis=0).tolist(),
            "class_distribution": {
                label: int(np.sum(self.encoded_labels.numpy() == i))
                for i, label in enumerate(self.label_encoder.classes_)
            }
        }


class StrokeSequenceDataset(Dataset):
    """
    PyTorch Dataset for stroke sequence data.
    
    Loads raw stroke point sequences for RNN/LSTM/Transformer training.
    Handles variable-length sequences with padding.
    """
    
    def __init__(
        self,
        stroke_data: List[StrokeData],
        max_sequence_length: int = 200,
        labels: Optional[List[str]] = None,
        label_encoder: Optional[LabelEncoder] = None,
        augment: bool = False,
        augment_prob: float = 0.3
    ):
        """
        Initialize the sequence dataset.
        
        Args:
            stroke_data: List of StrokeData objects
            max_sequence_length: Maximum sequence length (longer sequences will be truncated)
            labels: Optional list of specific labels to include
            label_encoder: Optional pre-fitted LabelEncoder
            augment: Whether to apply data augmentation
            augment_prob: Probability of applying augmentation
        """
        self.stroke_data = stroke_data
        self.max_sequence_length = max_sequence_length
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Filter by labels if specified
        if labels:
            self.stroke_data = [
                stroke for stroke in stroke_data 
                if stroke.label in labels
            ]
        
        if len(self.stroke_data) == 0:
            raise ValueError("No stroke data available for the specified criteria")
        
        # Prepare sequences and labels
        self.sequences, self.labels_raw = self._extract_sequences_and_labels()
        
        # Setup label encoder
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels_raw)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(self.labels_raw)
        
        # Convert to torch tensors
        self.sequences = torch.FloatTensor(self.sequences)
        self.encoded_labels = torch.LongTensor(self.encoded_labels)
        
        logger.info(f"Sequence dataset initialized with {len(self)} samples")
        logger.info(f"Max sequence length: {self.max_sequence_length}")
        logger.info(f"Feature dimension per point: {self.sequences.shape[2]}")
    
    def _extract_sequences_and_labels(self) -> Tuple[np.ndarray, List[str]]:
        """Extract point sequences and labels from stroke data."""
        sequences = []
        labels = []
        
        for stroke in self.stroke_data:
            if stroke.raw_points and len(stroke.raw_points) > 0:
                # Convert points to sequence
                sequence = self._points_to_sequence(stroke.raw_points)
                sequences.append(sequence)
                labels.append(stroke.label)
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences found in stroke data")
        
        # Pad/truncate sequences to max_sequence_length
        padded_sequences = []
        for seq in sequences:
            if len(seq) > self.max_sequence_length:
                # Truncate
                padded_seq = seq[:self.max_sequence_length]
            else:
                # Pad with zeros
                padding_length = self.max_sequence_length - len(seq)
                padding = np.zeros((padding_length, seq.shape[1]))
                padded_seq = np.vstack([seq, padding])
            
            padded_sequences.append(padded_seq)
        
        return np.array(padded_sequences), labels
    
    def _points_to_sequence(self, points: List[Dict]) -> np.ndarray:
        """Convert stroke points to sequence array."""
        sequence = []
        
        # Sort points by time
        sorted_points = sorted(points, key=lambda p: p.get('time', 0))
        
        for i, point in enumerate(sorted_points):
            features = [
                point.get('x', 0.0),
                point.get('y', 0.0),
                point.get('pressure', 1.0),
            ]
            
            # Add velocity if we have previous point
            if i > 0:
                prev_point = sorted_points[i-1]
                dt = point.get('time', 0) - prev_point.get('time', 0)
                if dt > 0:
                    dx = point.get('x', 0) - prev_point.get('x', 0)
                    dy = point.get('y', 0) - prev_point.get('y', 0)
                    velocity = np.sqrt(dx*dx + dy*dy) / dt
                else:
                    velocity = 0.0
            else:
                velocity = 0.0
            
            features.append(velocity)
            
            # Add relative time
            if i == 0:
                rel_time = 0.0
            else:
                rel_time = point.get('time', 0) - sorted_points[0].get('time', 0)
            
            features.append(rel_time)
            
            sequence.append(features)
        
        return np.array(sequence)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (sequence, label)
        """
        sequence = self.sequences[idx].clone()
        label = self.encoded_labels[idx]
        
        # Apply augmentation if enabled
        if self.augment and random.random() < self.augment_prob:
            sequence = self._augment_sequence(sequence)
        
        return sequence, label
    
    def _augment_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to sequence."""
        augmented = sequence.clone()
        
        # Find actual sequence length (non-zero rows)
        non_zero_mask = torch.any(augmented != 0, dim=1)
        actual_length = torch.sum(non_zero_mask).item()
        
        if actual_length == 0:
            return augmented
        
        # Translation augmentation
        if random.random() < 0.4:
            tx = random.uniform(-20, 20)
            ty = random.uniform(-20, 20)
            augmented[:actual_length, 0] += tx  # x coordinates
            augmented[:actual_length, 1] += ty  # y coordinates
        
        # Scaling augmentation
        if random.random() < 0.4:
            scale = random.uniform(0.8, 1.2)
            augmented[:actual_length, 0] *= scale  # x coordinates
            augmented[:actual_length, 1] *= scale  # y coordinates
        
        # Rotation augmentation
        if random.random() < 0.3:
            angle = random.uniform(-0.2, 0.2)  # ±~11 degrees
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            x_coords = augmented[:actual_length, 0].clone()
            y_coords = augmented[:actual_length, 1].clone()
            
            augmented[:actual_length, 0] = cos_a * x_coords - sin_a * y_coords
            augmented[:actual_length, 1] = sin_a * x_coords + cos_a * y_coords
        
        # Time jitter
        if random.random() < 0.3:
            time_noise = torch.normal(0, 0.01, (actual_length, 1))
            augmented[:actual_length, 4:5] += time_noise  # relative time
            # Ensure times remain monotonic
            augmented[:actual_length, 4] = torch.cummax(augmented[:actual_length, 4], dim=0)[0]
        
        return augmented


def create_data_loaders(
    stroke_data: List[StrokeData],
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    dataset_type: str = "features",
    labels: Optional[List[str]] = None,
    augment_train: bool = True,
    num_workers: int = 0,
    random_seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        stroke_data: List of StrokeData objects
        batch_size: Batch size for data loaders
        val_split: Validation set split ratio
        test_split: Test set split ratio
        dataset_type: "features" or "sequences"
        labels: Optional list of labels to include
        augment_train: Whether to augment training data
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
        
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    # Set random seed for reproducible splits
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Split data into train/val/test
    train_split = 1.0 - val_split - test_split
    
    if dataset_type == "features":
        # Create full dataset first to get shared encoders
        full_dataset = StrokeFeatureDataset(
            stroke_data=stroke_data,
            labels=labels,
            augment=False,
            normalize_features=True
        )
        
        # Split the data
        train_size = int(train_split * len(full_dataset))
        val_size = int(val_split * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        # If augmentation is needed for training, create augmented version
        if augment_train:
            # Get indices for training data
            train_indices = train_dataset.indices
            train_stroke_data = [stroke_data[i] for i in train_indices]
            
            train_dataset = StrokeFeatureDataset(
                stroke_data=train_stroke_data,
                labels=labels,
                feature_scaler=full_dataset.feature_scaler,
                label_encoder=full_dataset.label_encoder,
                augment=True,
                normalize_features=False  # Already normalized
            )
    
    elif dataset_type == "sequences":
        # Similar process for sequence datasets
        full_dataset = StrokeSequenceDataset(
            stroke_data=stroke_data,
            labels=labels,
            augment=False
        )
        
        train_size = int(train_split * len(full_dataset))
        val_size = int(val_split * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        if augment_train:
            train_indices = train_dataset.indices
            train_stroke_data = [stroke_data[i] for i in train_indices]
            
            train_dataset = StrokeSequenceDataset(
                stroke_data=train_stroke_data,
                labels=labels,
                label_encoder=full_dataset.label_encoder,
                augment=True
            )
    
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")
    
    # Create data loaders
    data_loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    }
    
    # Add dataset info
    data_loaders["info"] = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "total_size": len(full_dataset),
        "num_classes": len(full_dataset.label_encoder.classes_),
        "class_names": list(full_dataset.label_encoder.classes_),
        "dataset_type": dataset_type,
        "feature_dim": full_dataset.features.shape[1] if dataset_type == "features" else full_dataset.sequences.shape[2]
    }
    
    logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    return data_loaders


# Utility functions
def collate_fn_sequences(batch):
    """Custom collate function for variable length sequences."""
    sequences, labels = zip(*batch)
    
    # Stack sequences (already padded in dataset)
    sequences = torch.stack(sequences)
    labels = torch.stack(labels)
    
    return sequences, labels


def analyze_dataset_balance(stroke_data: List[StrokeData]) -> Dict[str, Any]:
    """Analyze class balance in the dataset."""
    label_counts = {}
    for stroke in stroke_data:
        label = stroke.label
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total_samples = len(stroke_data)
    
    analysis = {
        "total_samples": total_samples,
        "num_classes": len(label_counts),
        "class_counts": label_counts,
        "class_percentages": {
            label: (count / total_samples) * 100 
            for label, count in label_counts.items()
        },
        "is_balanced": all(
            0.1 <= (count / total_samples) <= 0.9 
            for count in label_counts.values()
        ),
        "min_class_size": min(label_counts.values()) if label_counts else 0,
        "max_class_size": max(label_counts.values()) if label_counts else 0
    }
    
    return analysis
