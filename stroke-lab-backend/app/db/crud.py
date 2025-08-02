"""
CRUD operations for Stroke Lab Backend

Provides Create, Read, Update, Delete operations for all database models.
"""

from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from datetime import datetime, timedelta
import uuid
import logging

from app.db.models import (
    StrokeData, TrainedModel, TrainingSession, DatasetImport,
    TrainingStatus, ModelType, create_stroke_data_from_dict
)

logger = logging.getLogger(__name__)


# ============================================================================
# STROKE DATA CRUD OPERATIONS
# ============================================================================

class StrokeDataCRUD:
    """CRUD operations for stroke data."""
    
    @staticmethod
    def create(db: Session, stroke_data: Dict[str, Any]) -> StrokeData:
        """Create a new stroke data record."""
        try:
            # Generate external ID if not provided
            if 'external_id' not in stroke_data:
                stroke_data['external_id'] = str(uuid.uuid4())
            
            stroke = create_stroke_data_from_dict(stroke_data)
            stroke.created_at = datetime.utcnow()
            
            db.add(stroke)
            db.commit()
            db.refresh(stroke)
            
            logger.info(f"Created stroke data record with ID: {stroke.id}")
            return stroke
            
        except Exception as e:
            logger.error(f"Error creating stroke data: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def get_by_id(db: Session, stroke_id: int) -> Optional[StrokeData]:
        """Get stroke data by ID."""
        return db.query(StrokeData).filter(StrokeData.id == stroke_id).first()
    
    @staticmethod
    def get_by_external_id(db: Session, external_id: str) -> Optional[StrokeData]:
        """Get stroke data by external ID."""
        return db.query(StrokeData).filter(StrokeData.external_id == external_id).first()
    
    @staticmethod
    def get_by_label(db: Session, label: str, limit: int = 100) -> List[StrokeData]:
        """Get stroke data by label."""
        return db.query(StrokeData).filter(
            StrokeData.label == label
        ).limit(limit).all()
    
    @staticmethod
    def get_all(
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        label_filter: Optional[str] = None,
        validated_only: bool = False
    ) -> List[StrokeData]:
        """Get all stroke data with optional filtering."""
        query = db.query(StrokeData)
        
        if label_filter:
            query = query.filter(StrokeData.label == label_filter)
        
        if validated_only:
            query = query.filter(StrokeData.is_validated == True)
        
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def get_count(
        db: Session, 
        label_filter: Optional[str] = None,
        validated_only: bool = False
    ) -> int:
        """Get count of stroke data records."""
        query = db.query(func.count(StrokeData.id))
        
        if label_filter:
            query = query.filter(StrokeData.label == label_filter)
        
        if validated_only:
            query = query.filter(StrokeData.is_validated == True)
        
        return query.scalar()
    
    @staticmethod
    def update(db: Session, stroke_id: int, update_data: Dict[str, Any]) -> Optional[StrokeData]:
        """Update stroke data record."""
        try:
            stroke = db.query(StrokeData).filter(StrokeData.id == stroke_id).first()
            if not stroke:
                return None
            
            for field, value in update_data.items():
                if hasattr(stroke, field):
                    setattr(stroke, field, value)
            
            stroke.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(stroke)
            
            logger.info(f"Updated stroke data record ID: {stroke_id}")
            return stroke
            
        except Exception as e:
            logger.error(f"Error updating stroke data {stroke_id}: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def delete(db: Session, stroke_id: int) -> bool:
        """Delete stroke data record."""
        try:
            stroke = db.query(StrokeData).filter(StrokeData.id == stroke_id).first()
            if not stroke:
                return False
            
            db.delete(stroke)
            db.commit()
            
            logger.info(f"Deleted stroke data record ID: {stroke_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting stroke data {stroke_id}: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def get_label_distribution(db: Session) -> Dict[str, int]:
        """Get distribution of labels in stroke data."""
        result = db.query(
            StrokeData.label,
            func.count(StrokeData.id).label('count')
        ).group_by(StrokeData.label).all()
        
        return {label: count for label, count in result}
    
    @staticmethod
    def get_quality_statistics(db: Session) -> Dict[str, float]:
        """Get quality statistics for stroke data."""
        stats = db.query(
            func.avg(StrokeData.quality_score).label('avg_quality'),
            func.min(StrokeData.quality_score).label('min_quality'),
            func.max(StrokeData.quality_score).label('max_quality'),
            func.count(StrokeData.id).label('total_count')
        ).filter(StrokeData.quality_score.isnot(None)).first()
        
        return {
            'average_quality': float(stats.avg_quality) if stats.avg_quality else 0.0,
            'min_quality': float(stats.min_quality) if stats.min_quality else 0.0,
            'max_quality': float(stats.max_quality) if stats.max_quality else 0.0,
            'total_count': int(stats.total_count) if stats.total_count else 0
        }
    
    @staticmethod
    def get_training_data(
        db: Session, 
        labels: Optional[List[str]] = None,
        min_quality: float = 0.0,
        validated_only: bool = True
    ) -> List[StrokeData]:
        """Get stroke data suitable for training."""
        query = db.query(StrokeData).filter(
            StrokeData.quality_score >= min_quality
        )
        
        if labels:
            query = query.filter(StrokeData.label.in_(labels))
        
        if validated_only:
            query = query.filter(StrokeData.is_validated == True)
        
        return query.all()


# ============================================================================
# TRAINED MODEL CRUD OPERATIONS
# ============================================================================

class TrainedModelCRUD:
    """CRUD operations for trained models."""
    
    @staticmethod
    def create(db: Session, model_data: Dict[str, Any]) -> TrainedModel:
        """Create a new trained model record."""
        try:
            model = TrainedModel(**model_data)
            model.created_at = datetime.utcnow()
            
            db.add(model)
            db.commit()
            db.refresh(model)
            
            logger.info(f"Created trained model record with ID: {model.id}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating trained model: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def get_by_id(db: Session, model_id: int) -> Optional[TrainedModel]:
        """Get trained model by ID."""
        return db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    
    @staticmethod
    def get_by_name_version(db: Session, name: str, version: str) -> Optional[TrainedModel]:
        """Get trained model by name and version."""
        return db.query(TrainedModel).filter(
            and_(TrainedModel.name == name, TrainedModel.version == version)
        ).first()
    
    @staticmethod
    def get_all(
        db: Session, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True,
        model_type: Optional[ModelType] = None
    ) -> List[TrainedModel]:
        """Get all trained models with optional filtering."""
        query = db.query(TrainedModel)
        
        if active_only:
            query = query.filter(TrainedModel.is_active == True)
        
        if model_type:
            query = query.filter(TrainedModel.model_type == model_type)
        
        return query.order_by(desc(TrainedModel.created_at)).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_best_model(db: Session, metric: str = "validation_accuracy") -> Optional[TrainedModel]:
        """Get the best performing model based on specified metric."""
        metric_column = getattr(TrainedModel, metric, None)
        if not metric_column:
            raise ValueError(f"Invalid metric: {metric}")
        
        return db.query(TrainedModel).filter(
            and_(
                TrainedModel.is_active == True,
                metric_column.isnot(None)
            )
        ).order_by(desc(metric_column)).first()
    
    @staticmethod
    def update(db: Session, model_id: int, update_data: Dict[str, Any]) -> Optional[TrainedModel]:
        """Update trained model record."""
        try:
            model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
            if not model:
                return None
            
            for field, value in update_data.items():
                if hasattr(model, field):
                    setattr(model, field, value)
            
            db.commit()
            db.refresh(model)
            
            logger.info(f"Updated trained model record ID: {model_id}")
            return model
            
        except Exception as e:
            logger.error(f"Error updating trained model {model_id}: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def delete(db: Session, model_id: int) -> bool:
        """Delete trained model record."""
        try:
            model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
            if not model:
                return False
            
            db.delete(model)
            db.commit()
            
            logger.info(f"Deleted trained model record ID: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting trained model {model_id}: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def deactivate_old_models(db: Session, keep_latest_n: int = 5) -> int:
        """Deactivate old models, keeping only the latest N."""
        try:
            # Get all models ordered by creation date
            all_models = db.query(TrainedModel).order_by(
                desc(TrainedModel.created_at)
            ).all()
            
            deactivated_count = 0
            for i, model in enumerate(all_models):
                if i >= keep_latest_n and model.is_active:
                    model.is_active = False
                    deactivated_count += 1
            
            db.commit()
            logger.info(f"Deactivated {deactivated_count} old models")
            return deactivated_count
            
        except Exception as e:
            logger.error(f"Error deactivating old models: {e}")
            db.rollback()
            raise


# ============================================================================
# TRAINING SESSION CRUD OPERATIONS
# ============================================================================

class TrainingSessionCRUD:
    """CRUD operations for training sessions."""
    
    @staticmethod
    def create(db: Session, session_data: Dict[str, Any]) -> TrainingSession:
        """Create a new training session."""
        try:
            if 'session_id' not in session_data:
                session_data['session_id'] = str(uuid.uuid4())
            
            session = TrainingSession(**session_data)
            session.started_at = datetime.utcnow()
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            logger.info(f"Created training session with ID: {session.id}")
            return session
            
        except Exception as e:
            logger.error(f"Error creating training session: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def get_by_id(db: Session, session_id: int) -> Optional[TrainingSession]:
        """Get training session by ID."""
        return db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    
    @staticmethod
    def get_by_session_id(db: Session, session_id: str) -> Optional[TrainingSession]:
        """Get training session by session ID."""
        return db.query(TrainingSession).filter(TrainingSession.session_id == session_id).first()
    
    @staticmethod
    def get_active_sessions(db: Session) -> List[TrainingSession]:
        """Get all active training sessions."""
        return db.query(TrainingSession).filter(
            TrainingSession.status == TrainingStatus.IN_PROGRESS
        ).all()
    
    @staticmethod
    def get_recent_sessions(db: Session, limit: int = 10) -> List[TrainingSession]:
        """Get recent training sessions."""
        return db.query(TrainingSession).order_by(
            desc(TrainingSession.started_at)
        ).limit(limit).all()
    
    @staticmethod
    def update_progress(
        db: Session, 
        session_id: str, 
        current_epoch: int,
        current_loss: float,
        current_accuracy: float,
        progress_percentage: float
    ) -> Optional[TrainingSession]:
        """Update training session progress."""
        try:
            session = db.query(TrainingSession).filter(
                TrainingSession.session_id == session_id
            ).first()
            
            if not session:
                return None
            
            session.current_epoch = current_epoch
            session.current_loss = current_loss
            session.current_accuracy = current_accuracy
            session.progress_percentage = progress_percentage
            
            db.commit()
            db.refresh(session)
            
            return session
            
        except Exception as e:
            logger.error(f"Error updating training session progress: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def complete_session(
        db: Session, 
        session_id: str, 
        final_metrics: Dict[str, Any],
        training_history: Dict[str, Any],
        model_id: Optional[int] = None
    ) -> Optional[TrainingSession]:
        """Mark training session as completed."""
        try:
            session = db.query(TrainingSession).filter(
                TrainingSession.session_id == session_id
            ).first()
            
            if not session:
                return None
            
            session.status = TrainingStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            session.final_metrics = final_metrics
            session.training_history = training_history
            session.progress_percentage = 100.0
            
            if model_id:
                session.model_id = model_id
            
            db.commit()
            db.refresh(session)
            
            logger.info(f"Completed training session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Error completing training session: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def fail_session(db: Session, session_id: str, error_message: str) -> Optional[TrainingSession]:
        """Mark training session as failed."""
        try:
            session = db.query(TrainingSession).filter(
                TrainingSession.session_id == session_id
            ).first()
            
            if not session:
                return None
            
            session.status = TrainingStatus.FAILED
            session.completed_at = datetime.utcnow()
            session.error_message = error_message
            
            db.commit()
            db.refresh(session)
            
            logger.error(f"Failed training session {session_id}: {error_message}")
            return session
            
        except Exception as e:
            logger.error(f"Error failing training session: {e}")
            db.rollback()
            raise


# ============================================================================
# DATASET IMPORT CRUD OPERATIONS
# ============================================================================

class DatasetImportCRUD:
    """CRUD operations for dataset imports."""
    
    @staticmethod
    def create(db: Session, import_data: Dict[str, Any]) -> DatasetImport:
        """Create a new dataset import record."""
        try:
            if 'import_id' not in import_data:
                import_data['import_id'] = str(uuid.uuid4())
            
            import_record = DatasetImport(**import_data)
            import_record.imported_at = datetime.utcnow()
            
            db.add(import_record)
            db.commit()
            db.refresh(import_record)
            
            logger.info(f"Created dataset import record with ID: {import_record.id}")
            return import_record
            
        except Exception as e:
            logger.error(f"Error creating dataset import: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def get_by_id(db: Session, import_id: int) -> Optional[DatasetImport]:
        """Get dataset import by ID."""
        return db.query(DatasetImport).filter(DatasetImport.id == import_id).first()
    
    @staticmethod
    def get_by_import_id(db: Session, import_id: str) -> Optional[DatasetImport]:
        """Get dataset import by import ID."""
        return db.query(DatasetImport).filter(DatasetImport.import_id == import_id).first()
    
    @staticmethod
    def get_recent_imports(db: Session, limit: int = 10) -> List[DatasetImport]:
        """Get recent dataset imports."""
        return db.query(DatasetImport).order_by(
            desc(DatasetImport.imported_at)
        ).limit(limit).all()
    
    @staticmethod
    def update_progress(
        db: Session, 
        import_id: str, 
        processed_files: int,
        failed_files: int = 0
    ) -> Optional[DatasetImport]:
        """Update dataset import progress."""
        try:
            import_record = db.query(DatasetImport).filter(
                DatasetImport.import_id == import_id
            ).first()
            
            if not import_record:
                return None
            
            import_record.processed_files = processed_files
            import_record.failed_files = failed_files
            
            if import_record.total_files:
                success_rate = (processed_files / import_record.total_files) * 100
                import_record.success_rate = success_rate
            
            db.commit()
            db.refresh(import_record)
            
            return import_record
            
        except Exception as e:
            logger.error(f"Error updating dataset import progress: {e}")
            db.rollback()
            raise


# ============================================================================
# COMPOSITE OPERATIONS
# ============================================================================

class CompositeOperations:
    """Complex operations involving multiple models."""
    
    @staticmethod
    def get_dataset_statistics(db: Session) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        try:
            # Basic counts
            total_strokes = db.query(func.count(StrokeData.id)).scalar()
            validated_strokes = db.query(func.count(StrokeData.id)).filter(
                StrokeData.is_validated == True
            ).scalar()
            
            # Label distribution
            label_dist = StrokeDataCRUD.get_label_distribution(db)
            
            # Quality statistics
            quality_stats = StrokeDataCRUD.get_quality_statistics(db)
            
            # Recent activity
            recent_imports = DatasetImportCRUD.get_recent_imports(db, limit=5)
            
            # Model counts
            total_models = db.query(func.count(TrainedModel.id)).scalar()
            active_models = db.query(func.count(TrainedModel.id)).filter(
                TrainedModel.is_active == True
            ).scalar()
            
            return {
                "stroke_statistics": {
                    "total_strokes": total_strokes,
                    "validated_strokes": validated_strokes,
                    "validation_rate": (validated_strokes / total_strokes * 100) if total_strokes > 0 else 0,
                    "label_distribution": label_dist,
                    "quality_statistics": quality_stats
                },
                "model_statistics": {
                    "total_models": total_models,
                    "active_models": active_models
                },
                "recent_imports": [imp.to_dict() for imp in recent_imports]
            }
            
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {e}")
            raise
    
    @staticmethod
    def prepare_training_dataset(
        db: Session,
        labels: Optional[List[str]] = None,
        min_quality: float = 0.7,
        train_split: float = 0.8,
        val_split: float = 0.15
    ) -> Dict[str, List[StrokeData]]:
        """Prepare training, validation, and test datasets."""
        try:
            # Get all suitable stroke data
            stroke_data = StrokeDataCRUD.get_training_data(
                db, labels=labels, min_quality=min_quality
            )
            
            if not stroke_data:
                return {"train": [], "validation": [], "test": []}
            
            # Group by label for stratified split
            label_groups = {}
            for stroke in stroke_data:
                if stroke.label not in label_groups:
                    label_groups[stroke.label] = []
                label_groups[stroke.label].append(stroke)
            
            train_data = []
            val_data = []
            test_data = []
            
            # Stratified split for each label
            for label, strokes in label_groups.items():
                n_strokes = len(strokes)
                n_train = int(n_strokes * train_split)
                n_val = int(n_strokes * val_split)
                
                # Shuffle strokes
                import random
                random.shuffle(strokes)
                
                train_data.extend(strokes[:n_train])
                val_data.extend(strokes[n_train:n_train + n_val])
                test_data.extend(strokes[n_train + n_val:])
            
            return {
                "train": train_data,
                "validation": val_data,
                "test": test_data
            }
            
        except Exception as e:
            logger.error(f"Error preparing training dataset: {e}")
            raise


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Create instances for easy import
stroke_crud = StrokeDataCRUD()
model_crud = TrainedModelCRUD()
session_crud = TrainingSessionCRUD()
import_crud = DatasetImportCRUD()
composite_ops = CompositeOperations()
