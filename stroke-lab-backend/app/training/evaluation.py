"""
Model evaluation and saving utilities for stroke classification models

Provides comprehensive evaluation metrics, model serialization,
and deployment utilities for trained stroke classification models.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import joblib

from app.training.model import create_model, count_parameters, get_model_size_mb
from app.training.dataset import StrokeFeatureDataset, create_data_loaders
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis.
    
    Provides detailed metrics, visualizations, and model interpretability tools.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_dataset(
        self,
        data_loader,
        dataset_name: str = "test",
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of the dataset (for logging)
            class_names: List of class names
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} dataset...")
        
        # Collect predictions and ground truth
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_features = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_features.extend(inputs.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        targets = np.array(all_targets)
        features = np.array(all_features)
        
        # Calculate metrics
        metrics = self._calculate_metrics(targets, predictions, probabilities, class_names)
        
        # Add dataset info
        metrics["dataset_info"] = {
            "name": dataset_name,
            "num_samples": len(targets),
            "num_classes": len(np.unique(targets)),
            "class_distribution": {
                str(i): int(np.sum(targets == i)) for i in np.unique(targets)
            }
        }
        
        # Store data for visualization
        metrics["evaluation_data"] = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "targets": targets.tolist(),
            "features": features.tolist() if features.size < 10000 else []  # Limit size
        }
        
        logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        num_classes = len(np.unique(targets))
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # Classification report
        if class_names:
            target_names = class_names
        else:
            target_names = [f"Class_{i}" for i in range(num_classes)]
        
        class_report = classification_report(
            targets, predictions, target_names=target_names, output_dict=True, zero_division=0
        )
        
        # Multi-class ROC AUC (if applicable)
        roc_auc = None
        if num_classes <= 50:  # Only calculate for reasonable number of classes
            try:
                if num_classes == 2:
                    roc_auc = roc_auc_score(targets, probabilities[:, 1])
                else:
                    roc_auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Top-k accuracy
        top_3_accuracy = self._calculate_top_k_accuracy(targets, probabilities, k=3)
        top_5_accuracy = self._calculate_top_k_accuracy(targets, probabilities, k=5)
        
        # Confidence statistics
        confidence_stats = self._calculate_confidence_statistics(probabilities, predictions == targets)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "top_3_accuracy": float(top_3_accuracy),
            "top_5_accuracy": float(top_5_accuracy),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "per_class_metrics": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "f1_score": f1.tolist(),
                "support": support.tolist(),
                "class_names": target_names
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "confidence_statistics": confidence_stats
        }
        
        return metrics
    
    def _calculate_top_k_accuracy(self, targets: np.ndarray, probabilities: np.ndarray, k: int) -> float:
        """Calculate top-k accuracy."""
        if k >= probabilities.shape[1]:
            return 1.0  # If k >= num_classes, top-k accuracy is always 1
        
        top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
        correct = np.any(top_k_predictions == targets.reshape(-1, 1), axis=1)
        return np.mean(correct)
    
    def _calculate_confidence_statistics(self, probabilities: np.ndarray, correct_mask: np.ndarray) -> Dict[str, float]:
        """Calculate confidence-related statistics."""
        max_probs = np.max(probabilities, axis=1)
        
        # Overall confidence stats
        mean_confidence = np.mean(max_probs)
        std_confidence = np.std(max_probs)
        
        # Confidence for correct vs incorrect predictions
        correct_confidence = np.mean(max_probs[correct_mask]) if np.any(correct_mask) else 0.0
        incorrect_confidence = np.mean(max_probs[~correct_mask]) if np.any(~correct_mask) else 0.0
        
        # Calibration metrics (simplified)
        confidence_bins = np.linspace(0, 1, 11)
        calibration_error = 0.0
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i + 1])
            if np.any(bin_mask):
                bin_accuracy = np.mean(correct_mask[bin_mask])
                bin_confidence = np.mean(max_probs[bin_mask])
                bin_weight = np.sum(bin_mask) / len(max_probs)
                calibration_error += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return {
            "mean_confidence": float(mean_confidence),
            "std_confidence": float(std_confidence),
            "correct_predictions_confidence": float(correct_confidence),
            "incorrect_predictions_confidence": float(incorrect_confidence),
            "expected_calibration_error": float(calibration_error)
        }
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None,
        normalize: bool = False
    ) -> plt.Figure:
        """Plot confusion matrix."""
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {save_path}")
        
        return plt.gcf()
    
    def plot_roc_curves(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Plot ROC curves for multi-class classification."""
        num_classes = probabilities.shape[1]
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(min(num_classes, 10)):  # Limit to 10 classes for readability
            # Convert to binary classification problem
            binary_targets = (targets == i).astype(int)
            class_probs = probabilities[:, i]
            
            fpr, tpr, _ = roc_curve(binary_targets, class_probs)
            auc = roc_auc_score(binary_targets, class_probs)
            
            class_name = class_names[i] if class_names else f"Class {i}"
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved: {save_path}")
        
        return plt.gcf()
    
    def analyze_feature_importance(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze feature importance using various methods."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import mutual_info_classif
        except ImportError:
            logger.warning("Scikit-learn not available for feature importance analysis")
            return {}
        
        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, targets)
        rf_importance = rf.feature_importances_
        
        # Mutual information
        mi_scores = mutual_info_classif(features, targets, random_state=42)
        
        # Statistical analysis
        feature_stats = {
            "means": np.mean(features, axis=0),
            "stds": np.std(features, axis=0),
            "mins": np.min(features, axis=0),
            "maxs": np.max(features, axis=0)
        }
        
        importance_analysis = {
            "random_forest_importance": rf_importance.tolist(),
            "mutual_information_scores": mi_scores.tolist(),
            "feature_statistics": {k: v.tolist() for k, v in feature_stats.items()},
            "feature_names": feature_names or [f"feature_{i}" for i in range(features.shape[1])]
        }
        
        return importance_analysis
    
    def visualize_feature_embeddings(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        method: str = "tsne",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Visualize feature embeddings in 2D."""
        if method.lower() == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) // 4))
        elif method.lower() == "pca":
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensionality
        embeddings = reducer.fit_transform(features[:1000])  # Limit for performance
        targets_subset = targets[:1000]
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=targets_subset,
            cmap='tab10',
            alpha=0.7
        )
        plt.colorbar(scatter)
        plt.title(f'{method.upper()} Visualization of Feature Embeddings')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature embeddings plot saved: {save_path}")
        
        return plt.gcf()


class ModelManager:
    """
    Model management utilities for saving, loading, and deploying models.
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory for saving models
        """
        self.models_dir = models_dir or settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelManager initialized with directory: {self.models_dir}")
    
    def save_model_package(
        self,
        model: nn.Module,
        model_info: Dict[str, Any],
        training_history: List[Dict[str, Any]],
        evaluation_results: Dict[str, Any],
        model_name: str,
        version: str = "1.0"
    ) -> Path:
        """
        Save a complete model package with all metadata.
        
        Args:
            model: Trained PyTorch model
            model_info: Model configuration and architecture info
            training_history: Training metrics history
            evaluation_results: Evaluation results
            model_name: Name of the model
            version: Model version
            
        Returns:
            Path to saved model package
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"{model_name}_v{version}_{timestamp}"
        package_dir = self.models_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = package_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save model architecture info
        architecture_info = {
            "model_class": model.__class__.__name__,
            "model_type": model_info.get("type", "unknown"),
            "input_dim": model_info.get("input_dim", 0),
            "num_classes": model_info.get("num_classes", 0),
            "parameters": count_parameters(model),
            "size_mb": get_model_size_mb(model),
            "config": model_info.get("config", {})
        }
        
        # Create comprehensive metadata
        metadata = {
            "model_info": architecture_info,
            "training_info": {
                "history": training_history,
                "final_metrics": training_history[-1] if training_history else {},
                "training_completed": True
            },
            "evaluation_results": evaluation_results,
            "package_info": {
                "name": model_name,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "package_name": package_name,
                "framework": "pytorch",
                "python_version": "3.8+",
                "dependencies": [
                    "torch>=1.9.0",
                    "torchvision>=0.10.0",
                    "numpy>=1.20.0",
                    "scikit-learn>=1.0.0"
                ]
            },
            "deployment_info": {
                "model_file": "model.pth",
                "metadata_file": "metadata.json",
                "requirements_file": "requirements.txt",
                "inference_example": "inference_example.py"
            }
        }
        
        # Save metadata
        metadata_path = package_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save requirements
        requirements_path = package_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            for req in metadata["package_info"]["dependencies"]:
                f.write(f"{req}\n")
        
        # Save inference example
        self._create_inference_example(package_dir, model_info)
        
        # Save feature scaler if available
        if "feature_scaler" in model_info:
            scaler_path = package_dir / "feature_scaler.pkl"
            joblib.dump(model_info["feature_scaler"], scaler_path)
        
        # Save label encoder if available
        if "label_encoder" in model_info:
            encoder_path = package_dir / "label_encoder.pkl"
            joblib.dump(model_info["label_encoder"], encoder_path)
        
        logger.info(f"Model package saved: {package_dir}")
        return package_dir
    
    def _create_inference_example(self, package_dir: Path, model_info: Dict[str, Any]):
        """Create an inference example script."""
        example_code = f'''"""
Example inference script for the saved stroke classification model.
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

def load_model(package_dir):
    """Load the trained model and metadata."""
    # Load metadata
    with open(package_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Get model info
    model_info = metadata["model_info"]
    
    # Create model (you need to have the model architecture available)
    # This is a simplified example - adapt based on your model type
    from your_model_module import create_model  # Replace with actual import
    
    model = create_model(
        model_type=model_info["model_type"],
        input_dim=model_info["input_dim"],
        num_classes=model_info["num_classes"],
        model_config=model_info["config"]
    )
    
    # Load state dict
    model.load_state_dict(torch.load(package_dir / "model.pth", map_location="cpu"))
    model.eval()
    
    return model, metadata

def predict(model, features):
    """Make predictions on new data."""
    with torch.no_grad():
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Add batch dimension if needed
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # Get predictions
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(outputs, dim=1)
        
        return predicted_classes.numpy(), probabilities.numpy()

# Example usage
if __name__ == "__main__":
    # Load model
    package_dir = Path("path/to/model/package")
    model, metadata = load_model(package_dir)
    
    # Example features (replace with actual stroke features)
    example_features = np.random.randn({model_info.get("input_dim", 49)})
    
    # Make prediction
    predicted_class, probabilities = predict(model, example_features)
    
    print(f"Predicted class: {{predicted_class[0]}}")
    print(f"Confidence: {{probabilities[0][predicted_class[0]]:.3f}}")
'''
        
        example_path = package_dir / "inference_example.py"
        with open(example_path, 'w') as f:
            f.write(example_code)
    
    def load_model_package(self, package_path: Path) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a model package.
        
        Args:
            package_path: Path to model package directory
            
        Returns:
            Tuple of (model, metadata)
        """
        # Load metadata
        metadata_path = package_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Model package loaded: {package_path}")
        return metadata
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """List all saved model packages."""
        model_packages = []
        
        for package_dir in self.models_dir.iterdir():
            if package_dir.is_dir():
                metadata_path = package_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        model_packages.append({
                            "package_name": package_dir.name,
                            "model_name": metadata["package_info"]["name"],
                            "version": metadata["package_info"]["version"],
                            "created_at": metadata["package_info"]["created_at"],
                            "path": str(package_dir),
                            "size_mb": metadata["model_info"]["size_mb"],
                            "accuracy": metadata["evaluation_results"].get("accuracy", 0.0)
                        })
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {package_dir}: {e}")
        
        # Sort by creation date
        model_packages.sort(key=lambda x: x["created_at"], reverse=True)
        
        return model_packages
    
    def cleanup_old_models(self, keep_latest: int = 5) -> int:
        """Remove old model packages, keeping only the latest N."""
        model_packages = self.list_saved_models()
        
        if len(model_packages) <= keep_latest:
            return 0
        
        # Remove old packages
        removed_count = 0
        for package in model_packages[keep_latest:]:
            package_path = Path(package["path"])
            try:
                import shutil
                shutil.rmtree(package_path)
                removed_count += 1
                logger.info(f"Removed old model package: {package['package_name']}")
            except Exception as e:
                logger.error(f"Could not remove {package_path}: {e}")
        
        return removed_count


# Utility functions
def comprehensive_model_evaluation(
    model: nn.Module,
    test_loader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computation device
        class_names: List of class names
        save_dir: Directory to save evaluation plots
        
    Returns:
        Complete evaluation results
    """
    evaluator = ModelEvaluator(model, device)
    
    # Evaluate on test set
    results = evaluator.evaluate_dataset(test_loader, "test", class_names)
    
    # Generate plots if save directory provided
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        cm = np.array(results["confusion_matrix"])
        evaluator.plot_confusion_matrix(
            cm, class_names, save_dir / "confusion_matrix.png"
        )
        evaluator.plot_confusion_matrix(
            cm, class_names, save_dir / "confusion_matrix_normalized.png", normalize=True
        )
        
        # ROC curves (if reasonable number of classes)
        if len(class_names or []) <= 10:
            eval_data = results["evaluation_data"]
            targets = np.array(eval_data["targets"])
            probabilities = np.array(eval_data["probabilities"])
            
            evaluator.plot_roc_curves(
                targets, probabilities, class_names, save_dir / "roc_curves.png"
            )
        
        # Feature embeddings (if features available)
        eval_data = results["evaluation_data"]
        if eval_data["features"]:
            features = np.array(eval_data["features"])
            targets = np.array(eval_data["targets"])
            
            evaluator.visualize_feature_embeddings(
                features, targets, "tsne", save_dir / "feature_embeddings_tsne.png"
            )
            evaluator.visualize_feature_embeddings(
                features, targets, "pca", save_dir / "feature_embeddings_pca.png"
            )
    
    return results
