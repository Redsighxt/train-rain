"""
FastAPI endpoints for Stroke Lab Backend

Provides comprehensive API endpoints for dataset management, model training,
stroke analysis, and system management.
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import logging
import json
import asyncio
from pathlib import Path
import aiofiles
import shutil
import numpy as np

from app.db.database import get_db, health_check
from app.db.crud import (
    stroke_crud, model_crud, session_crud, import_crud, composite_ops
)
from app.db.models import TrainingStatus, ModelType
from app.processing.image_to_stroke import ImageToStrokeConverter
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize image converter
image_converter = ImageToStrokeConverter()

# ============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# ============================================================================

class StrokePoint(BaseModel):
    x: float
    y: float
    time: float
    pressure: float = 1.0


class StrokeData(BaseModel):
    label: str = Field(..., description="Character label (A-Z, 0-9)")
    points: List[StrokePoint]
    source_type: str = "manual"
    quality_score: Optional[float] = None


class DatasetImportRequest(BaseModel):
    source_path: str
    source_type: str = "local"
    labels_filter: Optional[List[str]] = None
    min_quality: float = 0.0
    import_config: Optional[Dict[str, Any]] = None
    sample_size: Optional[int] = None  # For testing with limited samples
    progress_callback: Optional[str] = None  # WebSocket or polling endpoint


class TrainingRequest(BaseModel):
    model_name: str
    model_type: str = "cnn"
    description: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    labels: Optional[List[str]] = None
    min_quality: float = 0.7


class ModelResponse(BaseModel):
    id: int
    name: str
    version: str
    model_type: str
    description: Optional[str]
    training_accuracy: Optional[float]
    validation_accuracy: Optional[float]
    is_active: bool
    created_at: datetime


class TrainingStatusResponse(BaseModel):
    session_id: str
    status: str
    current_epoch: int
    total_epochs: int
    current_loss: Optional[float]
    current_accuracy: Optional[float]
    progress_percentage: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database: Dict[str, Any]
    system: Dict[str, Any]


# ============================================================================
# DATASET MANAGEMENT ENDPOINTS
# ============================================================================

async def upload_and_process_files(
    files: List[UploadFile],
    import_id: str,
    db: Session
) -> Dict[str, Any]:
    """Background task to process uploaded files."""
    try:
        # Create upload directory
        upload_dir = settings.DATASETS_DIR / "uploads" / import_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            file_path = upload_dir / file.filename
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            saved_files.append(file_path)
        
        # Update import record
        import_crud.update_progress(db, import_id, 0, 0)
        
        # Process files
        processed_count = 0
        failed_count = 0
        stroke_records = []
        
        for i, file_path in enumerate(saved_files):
            try:
                # Extract label from filename
                label = image_converter._extract_label_from_path(file_path)
                
                # Convert image to strokes
                result = image_converter.convert_image_to_strokes(str(file_path), label)
                
                if result["success"] and result["strokes"]:
                    # Create stroke data record
                    for stroke_data in result["strokes"]:
                        stroke_record = {
                            "external_id": str(uuid.uuid4()),
                            "label": label,
                            "raw_points": stroke_data["points"],
                            "source_type": "image",
                            "source_path": str(file_path),
                            "image_metadata": result["metrics"],
                            "quality_score": result["metrics"]["quality_score"],
                            "processing_time_ms": result["metrics"]["processing_time_ms"],
                            "algorithm_version": result["processing_metadata"]["algorithm_version"]
                        }
                        
                        created_stroke = stroke_crud.create(db, stroke_record)
                        stroke_records.append(created_stroke)
                    
                    processed_count += 1
                else:
                    failed_count += 1
                    logger.warning(f"Failed to process {file_path}: {result.get('error', 'Unknown error')}")
                
                # Update progress
                import_crud.update_progress(db, import_id, processed_count, failed_count)
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing file {file_path}: {e}")
                import_crud.update_progress(db, import_id, processed_count, failed_count)
        
        # Final update
        final_import = import_crud.get_by_import_id(db, import_id)
        if final_import:
            final_import.created_stroke_records = len(stroke_records)
            db.commit()
        
        logger.info(f"Import {import_id} completed: {processed_count} processed, {failed_count} failed")
        
        return {
            "processed_files": processed_count,
            "failed_files": failed_count,
            "stroke_records_created": len(stroke_records)
        }
        
    except Exception as e:
        logger.error(f"Error in background file processing: {e}")
        # Mark import as failed
        import_record = import_crud.get_by_import_id(db, import_id)
        if import_record:
            import_record.processing_errors = [str(e)]
            db.commit()
        raise


def create_dataset_import_endpoints(app: FastAPI):
    """Create dataset import endpoints."""
    
    @app.post("/api/dataset/import", summary="Import dataset from uploaded files")
    async def import_dataset_files(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        source_type: str = Form("upload"),
        db: Session = Depends(get_db)
    ):
        """Import dataset from uploaded image files."""
        try:
            if not files:
                raise HTTPException(status_code=400, detail="No files provided")
            
            # Validate file types
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            for file in files:
                if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unsupported file type: {file.filename}"
                    )
            
            # Create import record
            import_id = str(uuid.uuid4())
            import_data = {
                "import_id": import_id,
                "source_path": f"uploads/{import_id}",
                "source_type": source_type,
                "total_files": len(files),
                "import_config": {"file_count": len(files)}
            }
            
            import_record = import_crud.create(db, import_data)
            
            # Start background processing
            background_tasks.add_task(
                upload_and_process_files, 
                files, 
                import_id, 
                db
            )
            
            return {
                "success": True,
                "import_id": import_id,
                "message": f"Started processing {len(files)} files",
                "status_endpoint": f"/api/dataset/import/{import_id}/status"
            }
            
        except Exception as e:
            logger.error(f"Error starting dataset import: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/dataset/import/{import_id}/status", summary="Get import status")
    async def get_import_status(import_id: str, db: Session = Depends(get_db)):
        """Get the status of a dataset import operation."""
        import_record = import_crud.get_by_import_id(db, import_id)
        
        if not import_record:
            raise HTTPException(status_code=404, detail="Import not found")
        
        return {
            "import_id": import_id,
            "status": "completed" if import_record.success_rate is not None else "processing",
            "total_files": import_record.total_files,
            "processed_files": import_record.processed_files,
            "failed_files": import_record.failed_files,
            "success_rate": import_record.success_rate,
            "created_stroke_records": import_record.created_stroke_records,
            "errors": import_record.processing_errors or []
        }
    
    @app.post("/api/dataset/import/path", summary="Import dataset from file path")
    async def import_dataset_from_path(
        request: DatasetImportRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
    ):
        """Import dataset from a local file path or URL."""
        try:
            source_path = Path(request.source_path)
            if not source_path.exists():
                raise HTTPException(status_code=400, detail="Source path does not exist")
            
            # Create import record
            import_id = str(uuid.uuid4())
            import_data = {
                "import_id": import_id,
                "source_path": request.source_path,
                "source_type": request.source_type,
                "import_config": {
                    "labels_filter": request.labels_filter,
                    "min_quality": request.min_quality,
                    "sample_size": request.sample_size,
                    **(request.import_config or {})
                }
            }
            
            import_record = import_crud.create(db, import_data)
            
            # Start background processing
            background_tasks.add_task(
                process_dataset_directory,
                request.source_path,
                import_id,
                request.labels_filter,
                request.min_quality,
                request.sample_size,
                db
            )
            
            return {
                "success": True,
                "import_id": import_id,
                "message": f"Started processing dataset from {request.source_path}",
                "status_endpoint": f"/api/dataset/import/{import_id}/status"
            }
            
        except Exception as e:
            logger.error(f"Error starting path-based dataset import: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/dataset/import/handwritten", summary="Import HandwrittenCharacters dataset")
    async def import_handwritten_dataset(
        sample_size: Optional[int] = None,
        labels_filter: Optional[List[str]] = None,
        min_quality: float = 0.0,
        background_tasks: BackgroundTasks = None,
        db: Session = Depends(get_db)
    ):
        """Import the HandwrittenCharacters dataset with optional sampling."""
        try:
            # Default path to the HandwrittenCharacters folder
            default_path = Path("/home/rajat/Dev/train-rain/HandwrittenCharacters/handwritten-english-characters-and-digits/combined_folder/train")
            
            if not default_path.exists():
                raise HTTPException(
                    status_code=400, 
                    detail=f"HandwrittenCharacters dataset not found at {default_path}"
                )
            
            # Create import record
            import_id = str(uuid.uuid4())
            import_data = {
                "import_id": import_id,
                "source_path": str(default_path),
                "source_type": "handwritten_dataset",
                "import_config": {
                    "labels_filter": labels_filter,
                    "min_quality": min_quality,
                    "sample_size": sample_size,
                    "dataset_type": "handwritten_characters"
                }
            }
            
            import_record = import_crud.create(db, import_data)
            
            # Start background processing
            background_tasks.add_task(
                process_dataset_directory,
                str(default_path),
                import_id,
                labels_filter,
                min_quality,
                sample_size,
                db
            )
            
            sample_info = f" (sampling {sample_size} files)" if sample_size else ""
            return {
                "success": True,
                "import_id": import_id,
                "message": f"Started processing HandwrittenCharacters dataset{sample_info}",
                "status_endpoint": f"/api/dataset/import/{import_id}/status",
                "dataset_info": {
                    "path": str(default_path),
                    "sample_size": sample_size,
                    "labels_filter": labels_filter,
                    "min_quality": min_quality
                }
            }
            
        except Exception as e:
            logger.error(f"Error starting HandwrittenCharacters dataset import: {e}")
            raise HTTPException(status_code=500, detail=str(e))


async def process_dataset_directory(
    source_path: str,
    import_id: str,
    labels_filter: Optional[List[str]],
    min_quality: float,
    sample_size: Optional[int],
    db: Session
):
    """Background task to process dataset directory with progress tracking."""
    try:
        # Update import record to show processing started
        import_record = import_crud.get_by_import_id(db, import_id)
        if import_record:
            import_record.status = "processing"
            db.commit()
        
        # Use the enhanced image converter to process the dataset with progress
        result = await image_converter.convert_dataset_with_progress(
            source_path,
            str(settings.DATASETS_DIR / "processed" / import_id),
            import_id=import_id,
            db=db,
            sample_size=sample_size,
            labels_filter=labels_filter,
            min_quality=min_quality
        )
        
        # Update import record with final results
        import_record = import_crud.get_by_import_id(db, import_id)
        if import_record:
            import_record.total_files = result["total_files"]
            import_record.processed_files = result["processed_files"]
            import_record.failed_files = result["failed_files"]
            import_record.success_rate = (
                result["processed_files"] / result["total_files"] * 100
                if result["total_files"] > 0 else 0
            )
            import_record.processing_errors = result["errors"]
            import_record.status = "completed"
            db.commit()
        
        logger.info(f"Directory import {import_id} completed: {result['processed_files']}/{result['total_files']} files")
        
    except Exception as e:
        logger.error(f"Error processing dataset directory: {e}")
        import_record = import_crud.get_by_import_id(db, import_id)
        if import_record:
            import_record.processing_errors = [str(e)]
            import_record.status = "failed"
            db.commit()


# ============================================================================
# STROKE DATA ENDPOINTS
# ============================================================================

def create_stroke_data_endpoints(app: FastAPI):
    """Create stroke data management endpoints."""
    
    @app.post("/api/strokes", summary="Create new stroke data")
    async def create_stroke_data(stroke_data: StrokeData, db: Session = Depends(get_db)):
        """Create a new stroke data record."""
        try:
            # Convert to database format
            db_data = {
                "external_id": str(uuid.uuid4()),
                "label": stroke_data.label,
                "raw_points": [point.dict() for point in stroke_data.points],
                "source_type": stroke_data.source_type,
                "quality_score": stroke_data.quality_score or 0.8
            }
            
            created_stroke = stroke_crud.create(db, db_data)
            return {
                "success": True,
                "stroke_id": created_stroke.id,
                "external_id": created_stroke.external_id
            }
            
        except Exception as e:
            logger.error(f"Error creating stroke data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/strokes", summary="List stroke data")
    async def list_stroke_data(
        skip: int = 0,
        limit: int = 100,
        label: Optional[str] = None,
        validated_only: bool = False,
        db: Session = Depends(get_db)
    ):
        """List stroke data with optional filtering."""
        try:
            strokes = stroke_crud.get_all(
                db, skip=skip, limit=limit, 
                label_filter=label, validated_only=validated_only
            )
            
            total_count = stroke_crud.get_count(
                db, label_filter=label, validated_only=validated_only
            )
            
            return {
                "strokes": [stroke.to_dict() for stroke in strokes],
                "total_count": total_count,
                "skip": skip,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"Error listing stroke data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/strokes/{stroke_id}", summary="Get stroke data by ID")
    async def get_stroke_data(stroke_id: int, db: Session = Depends(get_db)):
        """Get stroke data by ID."""
        stroke = stroke_crud.get_by_id(db, stroke_id)
        if not stroke:
            raise HTTPException(status_code=404, detail="Stroke data not found")
        
        return stroke.to_dict()
    
    @app.delete("/api/strokes/{stroke_id}", summary="Delete stroke data")
    async def delete_stroke_data(stroke_id: int, db: Session = Depends(get_db)):
        """Delete stroke data by ID."""
        success = stroke_crud.delete(db, stroke_id)
        if not success:
            raise HTTPException(status_code=404, detail="Stroke data not found")
        
        return {"success": True, "message": "Stroke data deleted"}
    
    @app.get("/api/strokes/statistics", summary="Get stroke data statistics")
    async def get_stroke_statistics(db: Session = Depends(get_db)):
        """Get comprehensive stroke data statistics."""
        try:
            stats = composite_ops.get_dataset_statistics(db)
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stroke statistics: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRAINING MANAGEMENT ENDPOINTS
# ============================================================================

def create_training_endpoints(app: FastAPI):
    """Create model training and management endpoints."""

    @app.post("/api/train", summary="Start model training")
    async def start_training(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
    ):
        """Start a new model training session."""
        try:
            # Validate model type
            try:
                model_type_enum = ModelType(request.model_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model type: {request.model_type}"
                )

            # Check if we have enough data
            training_data = stroke_crud.get_training_data(
                db,
                labels=request.labels,
                min_quality=request.min_quality
            )

            if len(training_data) < 10:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient training data: {len(training_data)} samples (minimum 10 required)"
                )

            # Create training session
            session_id = str(uuid.uuid4())
            training_config = {
                "model_name": request.model_name,
                "model_type": request.model_type,
                "labels": request.labels,
                "min_quality": request.min_quality,
                "epochs": settings.MAX_EPOCHS,
                "batch_size": settings.BATCH_SIZE,
                "learning_rate": settings.LEARNING_RATE,
                **(request.training_config or {})
            }

            session_data = {
                "session_id": session_id,
                "status": TrainingStatus.PENDING,
                "total_epochs": training_config.get("epochs", settings.MAX_EPOCHS),
                "training_config": training_config
            }

            session = session_crud.create(db, session_data)

            # Start background training
            background_tasks.add_task(
                run_training_session,
                session_id,
                request,
                training_data,
                db
            )

            return {
                "success": True,
                "session_id": session_id,
                "message": f"Started training {request.model_name}",
                "training_data_count": len(training_data),
                "status_endpoint": f"/api/train/{session_id}/status"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/train/{session_id}/status", summary="Get training status")
    async def get_training_status(session_id: str, db: Session = Depends(get_db)):
        """Get the status of a training session."""
        session = session_crud.get_by_session_id(db, session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Training session not found")

        response = TrainingStatusResponse(
            session_id=session.session_id,
            status=session.status.value if session.status else "unknown",
            current_epoch=session.current_epoch,
            total_epochs=session.total_epochs,
            current_loss=session.current_loss,
            current_accuracy=session.current_accuracy,
            progress_percentage=session.progress_percentage
        )

        return response

    @app.post("/api/train/{session_id}/stop", summary="Stop training session")
    async def stop_training(session_id: str, db: Session = Depends(get_db)):
        """Stop a running training session."""
        session = session_crud.get_by_session_id(db, session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Training session not found")

        if session.status != TrainingStatus.IN_PROGRESS:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot stop session with status: {session.status.value}"
            )

        # Update session status
        session.status = TrainingStatus.STOPPED
        session.completed_at = datetime.utcnow()
        db.commit()

        return {"success": True, "message": "Training session stopped"}

    @app.get("/api/train/sessions", summary="List training sessions")
    async def list_training_sessions(
        limit: int = 10,
        db: Session = Depends(get_db)
    ):
        """List recent training sessions."""
        sessions = session_crud.get_recent_sessions(db, limit)

        return {
            "sessions": [session.to_dict() for session in sessions]
        }

    @app.get("/api/train/active", summary="Get active training sessions")
    async def get_active_training_sessions(db: Session = Depends(get_db)):
        """Get all currently active training sessions."""
        active_sessions = session_crud.get_active_sessions(db)

        return {
            "active_sessions": [session.to_dict() for session in active_sessions],
            "count": len(active_sessions)
        }


async def run_training_session(
    session_id: str,
    request: TrainingRequest,
    training_data: list,
    db: Session
):
    """Background task to run model training."""
    try:
        # Update session to in progress
        session = session_crud.get_by_session_id(db, session_id)
        if not session:
            logger.error(f"Training session {session_id} not found")
            return

        session.status = TrainingStatus.IN_PROGRESS
        session.started_at = datetime.utcnow()
        db.commit()

        # Prepare dataset
        dataset_split = composite_ops.prepare_training_dataset(
            db,
            labels=request.labels,
            min_quality=request.min_quality
        )

        train_data = dataset_split["train"]
        val_data = dataset_split["validation"]
        test_data = dataset_split["test"]

        logger.info(f"Training with {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")

        # Simulate training process (replace with actual training logic)
        total_epochs = session.total_epochs

        for epoch in range(1, total_epochs + 1):
            # Simulate training time
            await asyncio.sleep(0.2)  # 200ms per epoch for demo

            # Check if session was stopped
            session = session_crud.get_by_session_id(db, session_id)
            if session.status == TrainingStatus.STOPPED:
                logger.info(f"Training session {session_id} was stopped")
                return

            # Simulate training metrics
            progress = epoch / total_epochs

            # Loss decreases over time with some noise
            base_loss = 2.0 * (1 - progress) + 0.1
            current_loss = max(0.01, base_loss + np.random.normal(0, 0.1))

            # Accuracy increases over time with some noise
            base_accuracy = 0.5 + 0.45 * progress
            current_accuracy = min(0.98, base_accuracy + np.random.normal(0, 0.02))

            progress_percentage = progress * 100

            # Update session progress
            session_crud.update_progress(
                db, session_id, epoch, current_loss, current_accuracy, progress_percentage
            )

            logger.debug(f"Training {session_id}: Epoch {epoch}/{total_epochs}, Loss: {current_loss:.4f}, Acc: {current_accuracy:.4f}")

        # Training completed successfully
        final_metrics = {
            "final_accuracy": current_accuracy,
            "final_loss": current_loss,
            "train_samples": len(train_data),
            "validation_samples": len(val_data),
            "test_samples": len(test_data)
        }

        training_history = {
            "epochs": total_epochs,
            "final_epoch": total_epochs,
            "training_time_seconds": total_epochs * 0.2
        }

        # Create model record
        model_data = {
            "name": request.model_name,
            "version": "1.0",
            "model_type": ModelType(request.model_type.lower()),
            "description": request.description or f"Trained {request.model_type} model",
            "training_config": session.training_config,
            "dataset_info": {
                "total_samples": len(training_data),
                "train_samples": len(train_data),
                "validation_samples": len(val_data),
                "test_samples": len(test_data),
                "labels": request.labels or "all"
            },
            "training_accuracy": current_accuracy,
            "validation_accuracy": current_accuracy * 0.95,  # Simulate validation
            "test_accuracy": current_accuracy * 0.92,  # Simulate test
            "training_loss": current_loss,
            "validation_loss": current_loss * 1.1,
            "model_path": str(settings.MODELS_DIR / f"{request.model_name}_{session_id}.pth"),
            "model_size_bytes": 2100000,  # ~2.1 MB
            "model_parameters": 847000,
            "training_duration_seconds": int(total_epochs * 0.2),
            "epochs_completed": total_epochs,
            "best_epoch": int(total_epochs * 0.8)  # Best at 80% through training
        }

        created_model = model_crud.create(db, model_data)

        # Complete the training session
        session_crud.complete_session(
            db, session_id, final_metrics, training_history, created_model.id
        )

        logger.info(f"Training session {session_id} completed successfully")

    except Exception as e:
        logger.error(f"Training session {session_id} failed: {e}")
        session_crud.fail_session(db, session_id, str(e))


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

def create_model_endpoints(app: FastAPI):
    """Create model management endpoints."""
    
    @app.get("/api/models", summary="List trained models")
    async def list_models(
        skip: int = 0,
        limit: int = 20,
        active_only: bool = True,
        model_type: Optional[str] = None,
        db: Session = Depends(get_db)
    ):
        """List trained models with optional filtering."""
        try:
            model_type_enum = None
            if model_type:
                try:
                    model_type_enum = ModelType(model_type.lower())
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}")

            models = model_crud.get_all(
                db, skip=skip, limit=limit,
                active_only=active_only, model_type=model_type_enum
            )

            return {
                "models": [model.to_dict() for model in models],
                "count": len(models)
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/models/{model_id}", summary="Get model details")
    async def get_model(model_id: int, db: Session = Depends(get_db)):
        """Get detailed information about a specific model."""
        model = model_crud.get_by_id(db, model_id)

        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        return model.to_dict()

    @app.get("/api/models/best", summary="Get best performing model")
    async def get_best_model(
        metric: str = "validation_accuracy",
        db: Session = Depends(get_db)
    ):
        """Get the best performing model based on specified metric."""
        try:
            model = model_crud.get_best_model(db, metric)

            if not model:
                raise HTTPException(status_code=404, detail="No models found")

            return model.to_dict()

        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/models/{model_id}", summary="Delete model")
    async def delete_model(model_id: int, db: Session = Depends(get_db)):
        """Delete a trained model."""
        success = model_crud.delete(db, model_id)

        if not success:
            raise HTTPException(status_code=404, detail="Model not found")

        return {"success": True, "message": "Model deleted"}

    @app.post("/api/models/cleanup", summary="Cleanup old models")
    async def cleanup_old_models(
        keep_latest: int = 5,
        db: Session = Depends(get_db)
    ):
        """Deactivate old models, keeping only the latest N."""
        try:
            deactivated_count = model_crud.deactivate_old_models(db, keep_latest)

            return {
                "success": True,
                "deactivated_count": deactivated_count,
                "message": f"Deactivated {deactivated_count} old models"
            }

        except Exception as e:
            logger.error(f"Error cleaning up models: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL EXPORT ENDPOINTS
# ============================================================================

def create_model_export_endpoints(app: FastAPI):
    """Create model export endpoints."""
    
    @app.get("/api/models/{model_id}/export", summary="Export trained model")
    async def export_model(model_id: int, db: Session = Depends(get_db)):
        """Export a trained model file for download."""
        try:
            model = model_crud.get_by_id(db, model_id)

            if not model:
                raise HTTPException(status_code=404, detail="Model not found")

            if not model.is_active:
                raise HTTPException(status_code=400, detail="Cannot export inactive model")

            # Create export package
            export_data = {
                "model_info": {
                    "name": model.name,
                    "version": model.version,
                    "model_type": model.model_type.value if model.model_type else "unknown",
                    "description": model.description,
                    "created_at": model.created_at.isoformat() if model.created_at else None
                },
                "performance_metrics": {
                    "training_accuracy": model.training_accuracy,
                    "validation_accuracy": model.validation_accuracy,
                    "test_accuracy": model.test_accuracy,
                    "training_loss": model.training_loss,
                    "validation_loss": model.validation_loss,
                    "precision_scores": model.precision_scores,
                    "recall_scores": model.recall_scores,
                    "f1_scores": model.f1_scores
                },
                "training_details": {
                    "training_config": model.training_config,
                    "dataset_info": model.dataset_info,
                    "training_duration_seconds": model.training_duration_seconds,
                    "epochs_completed": model.epochs_completed,
                    "best_epoch": model.best_epoch,
                    "model_parameters": model.model_parameters
                },
                "model_artifacts": {
                    "model_path": model.model_path,
                    "model_size_bytes": model.model_size_bytes,
                    "algorithm_version": "1.0"
                },
                "export_metadata": {
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "export_version": "1.0",
                    "exported_by": "stroke_lab_backend",
                    "format": "pytorch_state_dict"
                }
            }

            # Check if model file exists
            model_file_path = Path(model.model_path) if model.model_path else None

            if model_file_path and model_file_path.exists():
                # Return actual model file
                return FileResponse(
                    path=str(model_file_path),
                    filename=f"{model.name}_v{model.version}.pth",
                    media_type="application/octet-stream",
                    headers={
                        "Content-Disposition": f"attachment; filename={model.name}_v{model.version}.pth",
                        "X-Model-Info": json.dumps(export_data["model_info"]),
                        "X-Model-Metrics": json.dumps(export_data["performance_metrics"])
                    }
                )
            else:
                # Return JSON export with metadata (for simulation)
                return JSONResponse(
                    content=export_data,
                    headers={
                        "Content-Disposition": f"attachment; filename={model.name}_v{model.version}_metadata.json"
                    }
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error exporting model {model_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/models/{model_id}/export/metadata", summary="Get model export metadata")
    async def get_model_export_metadata(model_id: int, db: Session = Depends(get_db)):
        """Get metadata for model export without downloading the file."""
        try:
            model = model_crud.get_by_id(db, model_id)

            if not model:
                raise HTTPException(status_code=404, detail="Model not found")

            # Calculate export size estimate
            base_metadata_size = 2048  # Base JSON metadata size
            model_file_size = model.model_size_bytes or 0
            total_size = base_metadata_size + model_file_size

            metadata = {
                "model_id": model.id,
                "model_name": model.name,
                "model_version": model.version,
                "export_format": "pytorch_state_dict",
                "estimated_size_bytes": total_size,
                "estimated_size_mb": round(total_size / 1024 / 1024, 2),
                "includes_model_file": model.model_file_exists,
                "model_file_path": model.model_path,
                "export_available": model.is_active,
                "performance_summary": {
                    "validation_accuracy": model.validation_accuracy,
                    "model_parameters": model.model_parameters,
                    "training_duration": model.training_duration_seconds
                },
                "export_urls": {
                    "download": f"/api/models/{model_id}/export",
                    "metadata_only": f"/api/models/{model_id}/export/metadata"
                }
            }

            return metadata

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting export metadata for model {model_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/models/export/batch", summary="Export multiple models")
    async def export_models_batch(
        model_ids: List[int],
        format: str = "json",
        db: Session = Depends(get_db)
    ):
        """Export multiple models in a batch operation."""
        try:
            if len(model_ids) > 10:
                raise HTTPException(
                    status_code=400,
                    detail="Maximum 10 models can be exported in a single batch"
                )

            export_results = []
            total_size = 0

            for model_id in model_ids:
                model = model_crud.get_by_id(db, model_id)

                if model and model.is_active:
                    model_data = {
                        "model_id": model.id,
                        "name": model.name,
                        "version": model.version,
                        "performance": {
                            "validation_accuracy": model.validation_accuracy,
                            "test_accuracy": model.test_accuracy
                        },
                        "size_bytes": model.model_size_bytes or 0,
                        "export_status": "ready"
                    }

                    export_results.append(model_data)
                    total_size += model.model_size_bytes or 0
                else:
                    export_results.append({
                        "model_id": model_id,
                        "export_status": "failed",
                        "error": "Model not found or inactive"
                    })

            batch_metadata = {
                "batch_id": str(uuid.uuid4()),
                "requested_models": len(model_ids),
                "exportable_models": len([r for r in export_results if r.get("export_status") == "ready"]),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "export_format": format,
                "models": export_results,
                "batch_created_at": datetime.utcnow().isoformat()
            }

            if format == "json":
                return JSONResponse(
                    content=batch_metadata,
                    headers={
                        "Content-Disposition": f"attachment; filename=model_batch_{batch_metadata['batch_id']}.json"
                    }
                )
            else:
                raise HTTPException(status_code=400, detail="Only 'json' format is currently supported")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in batch model export: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SYSTEM HEALTH ENDPOINTS
# ============================================================================

def create_system_endpoints(app: FastAPI):
    """Create system health and information endpoints."""
    
    @app.get("/api/health", summary="System health check")
    async def health_check_endpoint():
        """Comprehensive system health check."""
        try:
            # Database health
            db_health = health_check()
            
            # System info
            system_info = {
                "version": settings.APP_VERSION,
                "debug_mode": settings.DEBUG,
                "models_dir": str(settings.MODELS_DIR),
                "datasets_dir": str(settings.DATASETS_DIR)
            }
            
            overall_status = "healthy" if db_health["status"] == "healthy" else "unhealthy"
            
            return HealthResponse(
                status=overall_status,
                timestamp=datetime.utcnow(),
                version=settings.APP_VERSION,
                database=db_health,
                system=system_info
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow(),
                version=settings.APP_VERSION,
                database={"status": "error", "error": str(e)},
                system={"error": str(e)}
            )
    
    @app.get("/api/info", summary="System information")
    async def get_system_info():
        """Get system information and configuration."""
        return {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "debug": settings.DEBUG,
            "database_url": settings.DATABASE_URL.replace("sqlite:///", "sqlite:///(hidden)"),
            "max_epochs": settings.MAX_EPOCHS,
            "batch_size": settings.BATCH_SIZE,
            "learning_rate": settings.LEARNING_RATE,
            "image_size": settings.IMAGE_SIZE,
            "max_stroke_points": settings.MAX_STROKE_POINTS
        }


# ============================================================================
# INITIALIZE ALL ENDPOINTS
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Stroke Lab Backend",
        description="Advanced backend for stroke invariant research laboratory",
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Create all endpoint groups
    create_dataset_import_endpoints(app)
    create_stroke_data_endpoints(app)
    create_training_endpoints(app)
    create_model_endpoints(app)
    create_model_export_endpoints(app)
    create_system_endpoints(app)
    
    return app


# Create the FastAPI app instance
app = create_app()


# Root endpoint
@app.get("/", summary="API Root")
async def root():
    """API root endpoint with basic information."""
    return {
        "message": "Stroke Lab Backend API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/api/health"
    }
