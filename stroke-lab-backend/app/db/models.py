"""
Database models for Stroke Lab Backend

Defines SQLAlchemy ORM models for stroke data, trained models, and related entities.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean, 
    ForeignKey, JSON, LargeBinary, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
import enum
import json
from pathlib import Path

from app.db.database import Base
from app.core.config import get_settings

settings = get_settings()


class TrainingStatus(enum.Enum):
    """Enumeration for training status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class ModelType(enum.Enum):
    """Enumeration for model types."""
    CNN = "cnn"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class StrokeData(Base):
    """
    Model for storing stroke data and computed features.
    
    Stores raw stroke points, computed invariants, and derived features
    for machine learning training and analysis.
    """
    __tablename__ = "stroke_data"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(255), unique=True, index=True)  # For frontend reference
    
    # Metadata
    label = Column(String(10), nullable=False, index=True)  # Character label (A-Z, 0-9)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Source information
    source_type = Column(String(50), default="manual")  # manual, image, dataset
    source_path = Column(String(500), nullable=True)  # Original file path if applicable
    image_metadata = Column(JSON, nullable=True)  # Image processing metadata
    
    # Raw stroke data (JSON format)
    raw_points = Column(JSON, nullable=False)  # List of {x, y, time, pressure?}
    processed_points = Column(JSON, nullable=True)  # Normalized/processed points
    
    # Geometric invariants
    arc_length = Column(Float, nullable=True)
    total_turning = Column(Float, nullable=True)
    winding_number = Column(Float, nullable=True)
    writhe = Column(Float, nullable=True)
    
    # Topological invariants
    betti_numbers = Column(JSON, nullable=True)  # {b0, b1}
    persistence_diagram = Column(JSON, nullable=True)  # List of birth-death pairs
    
    # Statistical invariants
    complexity_score = Column(Float, nullable=True)
    regularity_score = Column(Float, nullable=True)
    symmetry_score = Column(Float, nullable=True)
    stability_index = Column(Float, nullable=True)
    
    # Path signature features
    path_signature_level1 = Column(JSON, nullable=True)
    path_signature_level2 = Column(JSON, nullable=True)
    path_signature_level3 = Column(JSON, nullable=True)
    path_signature_level4 = Column(JSON, nullable=True)
    log_signature = Column(JSON, nullable=True)
    
    # Spectral features
    fft_coefficients = Column(JSON, nullable=True)
    wavelet_coefficients = Column(JSON, nullable=True)
    mfcc_features = Column(JSON, nullable=True)
    spectral_centroid = Column(Float, nullable=True)
    spectral_bandwidth = Column(Float, nullable=True)
    
    # 3D signature coordinates
    signature_coordinates = Column(JSON, nullable=True)  # [{x, y, z, weight}]
    signature_axis_mapping = Column(JSON, nullable=True)  # {x: "feature", y: "feature", z: "feature"}
    signature_quality_metrics = Column(JSON, nullable=True)  # Quality scores
    
    # Landmark points
    landmark_points = Column(JSON, nullable=True)  # Detected landmark points
    
    # Quality and validation
    quality_score = Column(Float, nullable=True)  # Overall quality (0-1)
    is_validated = Column(Boolean, default=False)
    validation_notes = Column(Text, nullable=True)
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)
    algorithm_version = Column(String(50), nullable=True)
    
    # Relationships
    # training_sessions = relationship("TrainingSession", back_populates="stroke_data")
    
    @hybrid_property
    def feature_vector(self) -> List[float]:
        """Get the complete feature vector for ML training."""
        features = []
        
        # Geometric features
        features.extend([
            self.arc_length or 0.0,
            self.total_turning or 0.0,
            self.winding_number or 0.0,
            self.writhe or 0.0
        ])
        
        # Statistical features
        features.extend([
            self.complexity_score or 0.0,
            self.regularity_score or 0.0,
            self.symmetry_score or 0.0,
            self.stability_index or 0.0
        ])
        
        # Path signature features (truncated to fixed size)
        if self.path_signature_level1:
            features.extend(self.path_signature_level1[:10])  # First 10 components
        else:
            features.extend([0.0] * 10)
            
        if self.log_signature:
            features.extend(self.log_signature[:20])  # First 20 components
        else:
            features.extend([0.0] * 20)
        
        # Spectral features
        features.extend([
            self.spectral_centroid or 0.0,
            self.spectral_bandwidth or 0.0
        ])
        
        if self.mfcc_features:
            features.extend(self.mfcc_features[:13])  # Standard 13 MFCC coefficients
        else:
            features.extend([0.0] * 13)
        
        return features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stroke data to dictionary format."""
        return {
            "id": self.id,
            "external_id": self.external_id,
            "label": self.label,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "source_type": self.source_type,
            "raw_points": self.raw_points,
            "quality_score": self.quality_score,
            "feature_vector": self.feature_vector,
            "is_validated": self.is_validated
        }


class TrainedModel(Base):
    """
    Model for storing trained machine learning models and metadata.
    
    Tracks model training sessions, performance metrics, and model artifacts.
    """
    __tablename__ = "trained_models"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    
    # Model metadata
    model_type = Column(SQLEnum(ModelType), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Training configuration
    training_config = Column(JSON, nullable=False)  # Hyperparameters, etc.
    dataset_info = Column(JSON, nullable=False)  # Dataset statistics
    
    # Performance metrics
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    training_loss = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    
    # Additional metrics
    precision_scores = Column(JSON, nullable=True)  # Per-class precision
    recall_scores = Column(JSON, nullable=True)     # Per-class recall
    f1_scores = Column(JSON, nullable=True)         # Per-class F1
    confusion_matrix = Column(JSON, nullable=True)  # Confusion matrix
    
    # Model artifacts
    model_path = Column(String(500), nullable=False)  # Path to saved model
    model_size_bytes = Column(Integer, nullable=True)
    model_parameters = Column(Integer, nullable=True)  # Number of parameters
    
    # Training details
    training_duration_seconds = Column(Integer, nullable=True)
    epochs_completed = Column(Integer, nullable=True)
    best_epoch = Column(Integer, nullable=True)
    
    # Status and validation
    is_active = Column(Boolean, default=True)
    is_validated = Column(Boolean, default=False)
    validation_notes = Column(Text, nullable=True)
    
    # Relationships
    training_sessions = relationship("TrainingSession", back_populates="model")
    
    @hybrid_property
    def model_file_exists(self) -> bool:
        """Check if the model file exists on disk."""
        return Path(self.model_path).exists() if self.model_path else False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type.value if self.model_type else None,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "test_accuracy": self.test_accuracy,
            "model_size_bytes": self.model_size_bytes,
            "model_parameters": self.model_parameters,
            "training_duration_seconds": self.training_duration_seconds,
            "epochs_completed": self.epochs_completed,
            "is_active": self.is_active,
            "is_validated": self.is_validated,
            "model_file_exists": self.model_file_exists
        }


class TrainingSession(Base):
    """
    Model for tracking individual training sessions.
    
    Links stroke data to trained models and tracks training progress.
    """
    __tablename__ = "training_sessions"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True)
    
    # Status and timing
    status = Column(SQLEnum(TrainingStatus), default=TrainingStatus.PENDING)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Progress tracking
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, nullable=False)
    current_loss = Column(Float, nullable=True)
    current_accuracy = Column(Float, nullable=True)
    progress_percentage = Column(Float, default=0.0)
    
    # Configuration
    training_config = Column(JSON, nullable=False)
    dataset_split = Column(JSON, nullable=True)  # Train/val/test split info
    
    # Results
    final_metrics = Column(JSON, nullable=True)
    training_history = Column(JSON, nullable=True)  # Loss/accuracy per epoch
    
    # Error handling
    error_message = Column(Text, nullable=True)
    
    # Foreign keys
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=True)
    
    # Relationships
    model = relationship("TrainedModel", back_populates="training_sessions")
    # stroke_data = relationship("StrokeData", back_populates="training_sessions")
    
    @hybrid_property
    def duration_seconds(self) -> Optional[int]:
        """Calculate training duration in seconds."""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert training session to dictionary format."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "status": self.status.value if self.status else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_loss": self.current_loss,
            "current_accuracy": self.current_accuracy,
            "progress_percentage": self.progress_percentage,
            "duration_seconds": self.duration_seconds,
            "model_id": self.model_id
        }


class DatasetImport(Base):
    """
    Model for tracking dataset import operations.
    
    Records information about imported datasets and processing status.
    """
    __tablename__ = "dataset_imports"
    
    # Primary key and identification
    id = Column(Integer, primary_key=True, index=True)
    import_id = Column(String(255), unique=True, index=True)
    
    # Import metadata
    source_path = Column(String(500), nullable=False)
    source_type = Column(String(50), nullable=False)  # kaggle, local, url
    imported_at = Column(DateTime, default=datetime.utcnow)
    
    # Processing status
    total_files = Column(Integer, nullable=True)
    processed_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)
    success_rate = Column(Float, nullable=True)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    progress_percentage = Column(Float, default=0.0)
    current_file = Column(String(500), nullable=True)
    
    # Results
    created_stroke_records = Column(Integer, default=0)
    processing_errors = Column(JSON, nullable=True)  # List of error messages
    labels_processed = Column(JSON, nullable=True)  # List of processed labels
    
    # Configuration used for import
    import_config = Column(JSON, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset import to dictionary format."""
        return {
            "id": self.id,
            "import_id": self.import_id,
            "source_path": self.source_path,
            "source_type": self.source_type,
            "imported_at": self.imported_at.isoformat() if self.imported_at else None,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "success_rate": self.success_rate,
            "created_stroke_records": self.created_stroke_records
        }


# Helper functions for model operations
def create_stroke_data_from_dict(data: Dict[str, Any]) -> StrokeData:
    """Create a StrokeData instance from dictionary data."""
    stroke = StrokeData()
    
    # Basic fields
    for field in ['external_id', 'label', 'source_type', 'source_path']:
        if field in data:
            setattr(stroke, field, data[field])
    
    # JSON fields
    json_fields = [
        'raw_points', 'processed_points', 'image_metadata',
        'betti_numbers', 'persistence_diagram',
        'path_signature_level1', 'path_signature_level2',
        'path_signature_level3', 'path_signature_level4',
        'log_signature', 'fft_coefficients', 'wavelet_coefficients',
        'mfcc_features', 'signature_coordinates', 'signature_axis_mapping',
        'signature_quality_metrics', 'landmark_points'
    ]
    
    for field in json_fields:
        if field in data:
            setattr(stroke, field, data[field])
    
    # Numeric fields
    numeric_fields = [
        'arc_length', 'total_turning', 'winding_number', 'writhe',
        'complexity_score', 'regularity_score', 'symmetry_score',
        'stability_index', 'spectral_centroid', 'spectral_bandwidth',
        'quality_score', 'processing_time_ms'
    ]
    
    for field in numeric_fields:
        if field in data:
            setattr(stroke, field, data[field])
    
    return stroke


def get_label_distribution(db_session) -> Dict[str, int]:
    """Get the distribution of labels in the stroke data."""
    from sqlalchemy import func
    
    result = db_session.query(
        StrokeData.label,
        func.count(StrokeData.id).label('count')
    ).group_by(StrokeData.label).all()
    
    return {label: count for label, count in result}


def get_model_performance_summary(model_id: int, db_session) -> Dict[str, Any]:
    """Get a summary of model performance metrics."""
    model = db_session.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    
    if not model:
        return {}
    
    return {
        "model_info": model.to_dict(),
        "performance": {
            "training_accuracy": model.training_accuracy,
            "validation_accuracy": model.validation_accuracy,
            "test_accuracy": model.test_accuracy,
            "precision_scores": model.precision_scores,
            "recall_scores": model.recall_scores,
            "f1_scores": model.f1_scores
        },
        "training_details": {
            "duration_seconds": model.training_duration_seconds,
            "epochs_completed": model.epochs_completed,
            "best_epoch": model.best_epoch,
            "model_parameters": model.model_parameters
        }
    }
