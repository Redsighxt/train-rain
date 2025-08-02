"""
Configuration settings for Stroke Lab Backend

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application Info
    APP_NAME: str = "Stroke Lab Backend"
    APP_VERSION: str = "1.0.0"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 6969
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "sqlite:///./stroke_lab.db"
    
    # Directories
    MODELS_DIR: Path = Path("./models")
    DATASETS_DIR: Path = Path("./datasets")
    LOGS_DIR: Path = Path("./logs")
    
    # Training Configuration
    MAX_EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    VALIDATION_SPLIT: float = 0.2
    
    # Processing Configuration
    IMAGE_SIZE: int = 128
    MAX_STROKE_POINTS: int = 200
    SMOOTHING_FACTOR: float = 0.5
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Performance
    MAX_WORKERS: int = 4
    CACHE_SIZE: int = 1000
    
    # External APIs
    KAGGLE_API_KEY: Optional[str] = None
    KAGGLE_USERNAME: Optional[str] = None
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    @validator("MODELS_DIR", "DATASETS_DIR", "LOGS_DIR", pre=True)
    def create_directories(cls, v):
        """Ensure directories exist."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @validator("LOG_FILE", pre=True)
    def set_log_file(cls, v, values):
        """Set default log file if not provided."""
        if v is None and "LOGS_DIR" in values:
            return str(values["LOGS_DIR"] / "stroke_lab.log")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


class DatabaseSettings:
    """Database-specific configuration."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.is_sqlite = database_url.startswith("sqlite")
        self.is_postgres = database_url.startswith("postgresql")
        
    @property
    def connect_args(self) -> dict:
        """Connection arguments based on database type."""
        if self.is_sqlite:
            return {"check_same_thread": False}
        return {}


class TrainingSettings:
    """Training-specific configuration."""
    
    def __init__(self, settings: Settings):
        self.max_epochs = settings.MAX_EPOCHS
        self.batch_size = settings.BATCH_SIZE
        self.learning_rate = settings.LEARNING_RATE
        self.validation_split = settings.VALIDATION_SPLIT
        
    @property
    def device(self) -> str:
        """Determine the best available device for training."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# Global settings instance
settings = Settings()

# Derived settings
db_settings = DatabaseSettings(settings.DATABASE_URL)
training_settings = TrainingSettings(settings)


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def get_database_settings() -> DatabaseSettings:
    """Get database settings."""
    return db_settings


def get_training_settings() -> TrainingSettings:
    """Get training settings."""
    return training_settings


# Development helpers
def is_development() -> bool:
    """Check if running in development mode."""
    return settings.DEBUG


def is_production() -> bool:
    """Check if running in production mode."""
    return not settings.DEBUG


# Logging configuration
def setup_logging():
    """Configure logging for the application."""
    import logging
    from pathlib import Path
    
    # Create logs directory
    log_dir = Path(settings.LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.NullHandler(),
        ]
    )
    
    return logging.getLogger("stroke_lab")


# Model paths
def get_model_path(model_name: str) -> Path:
    """Get the full path for a model file."""
    return settings.MODELS_DIR / f"{model_name}.pth"


def get_dataset_path(dataset_name: str) -> Path:
    """Get the full path for a dataset directory."""
    return settings.DATASETS_DIR / dataset_name


# Validation helpers
def validate_image_file(file_path: Path) -> bool:
    """Validate if a file is a supported image format."""
    supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    return file_path.suffix.lower() in supported_formats


def validate_stroke_data(data: dict) -> bool:
    """Validate stroke data format."""
    required_keys = {'x', 'y', 'time'}
    if not isinstance(data, dict):
        return False
    
    points = data.get('points', [])
    if not isinstance(points, list) or len(points) < 2:
        return False
    
    for point in points:
        if not isinstance(point, dict) or not required_keys.issubset(point.keys()):
            return False
        
        try:
            float(point['x'])
            float(point['y'])
            float(point['time'])
        except (ValueError, TypeError):
            return False
    
    return True
