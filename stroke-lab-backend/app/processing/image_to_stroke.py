"""
Image-to-Stroke Conversion Pipeline

Advanced image processing pipeline that converts handwritten character images
into time-series stroke data using skeletonization and path tracing algorithms.
"""

import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime
from dataclasses import dataclass
from scipy import ndimage
from skimage import morphology, measure
from skimage.morphology import skeletonize, thin
import networkx as nx

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class Point:
    """Represents a point in the stroke."""
    x: float
    y: float
    time: float
    pressure: float = 1.0


@dataclass
class ProcessingMetrics:
    """Metrics collected during processing."""
    processing_time_ms: float
    original_size: Tuple[int, int]
    skeleton_points: int
    stroke_points: int
    branches_detected: int
    quality_score: float


class ImagePreprocessor:
    """Handles image preprocessing operations."""
    
    def __init__(self, target_size: int = 128):
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and validate image file."""
        try:
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path)
            else:
                # Assume it's already a PIL Image or numpy array
                image = image_path
            
            # Convert to grayscale if needed
            if isinstance(image, Image.Image):
                if image.mode != 'L':
                    image = image.convert('L')
                image_array = np.array(image)
            else:
                image_array = image
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            logger.debug(f"Loaded image with shape: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Comprehensive image preprocessing pipeline."""
        # Store original size
        original_shape = image.shape
        
        # 1. Resize while maintaining aspect ratio
        image = self._resize_with_padding(image, self.target_size)
        
        # 2. Normalize intensity
        image = self._normalize_intensity(image)
        
        # 3. Apply adaptive thresholding
        binary = self._adaptive_threshold(image)
        
        # 4. Clean up noise
        binary = self._denoise(binary)
        
        # 5. Ensure proper foreground/background
        binary = self._ensure_foreground(binary)
        
        logger.debug(f"Preprocessed image from {original_shape} to {binary.shape}")
        return binary
    
    def _resize_with_padding(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio using padding."""
        h, w = image.shape
        
        # Calculate scaling factor
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.ones((target_size, target_size), dtype=np.uint8) * 255
        
        # Center the resized image
        start_x = (target_size - new_w) // 2
        start_y = (target_size - new_h) // 2
        padded[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return padded
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity to full range."""
        min_val, max_val = np.min(image), np.max(image)
        if max_val > min_val:
            normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = image
        return normalized
    
    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to binarize the image."""
        # Try multiple thresholding methods and choose the best
        methods = [
            cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2),
            cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2),
        ]
        
        # Also try Otsu's method
        _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        methods.append(otsu)
        
        # Choose the method that gives the most reasonable foreground ratio
        best_binary = methods[0]
        best_ratio = np.sum(best_binary > 0) / (best_binary.shape[0] * best_binary.shape[1])
        
        for binary in methods[1:]:
            ratio = np.sum(binary > 0) / (binary.shape[0] * binary.shape[1])
            # Prefer ratios between 0.05 and 0.3 (5% to 30% foreground)
            if 0.05 <= ratio <= 0.3:
                if not (0.05 <= best_ratio <= 0.3) or abs(ratio - 0.15) < abs(best_ratio - 0.15):
                    best_binary = binary
                    best_ratio = ratio
        
        return best_binary
    
    def _denoise(self, binary: np.ndarray) -> np.ndarray:
        """Remove noise from binary image."""
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Keep only components with reasonable size
        min_size = max(10, binary.shape[0] * binary.shape[1] * 0.001)  # At least 0.1% of image
        
        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned[labels == i] = 255
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _ensure_foreground(self, binary: np.ndarray) -> np.ndarray:
        """Ensure the character is foreground (white on black background)."""
        # Count pixels near the border
        border_pixels = np.concatenate([
            binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
        ])
        
        border_white = np.sum(border_pixels == 255)
        border_black = np.sum(border_pixels == 0)
        
        # If more border pixels are white, invert the image
        if border_white > border_black:
            return 255 - binary
        
        return binary


class SkeletonExtractor:
    """Extracts skeleton from binary image using advanced algorithms."""
    
    def __init__(self):
        pass
    
    def extract_skeleton(self, binary_image: np.ndarray) -> np.ndarray:
        """Extract skeleton using Zhang-Suen algorithm."""
        # Convert to binary (0 and 1)
        binary = (binary_image > 127).astype(np.uint8)
        
        # Apply Zhang-Suen skeletonization
        skeleton = self._zhang_suen_skeleton(binary)
        
        # Clean up skeleton
        skeleton = self._clean_skeleton(skeleton)
        
        return skeleton
    
    def _zhang_suen_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """Implement Zhang-Suen skeletonization algorithm."""
        # Use scikit-image implementation which is based on Zhang-Suen
        skeleton = skeletonize(binary)
        return skeleton.astype(np.uint8) * 255
    
    def _clean_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Clean up skeleton by removing isolated pixels and short branches."""
        # Convert to binary
        skel_binary = skeleton > 127
        
        # Remove isolated pixels
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(skel_binary.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Remove very short branches (length < 5 pixels)
        cleaned = self._remove_short_branches(cleaned, min_length=5)
        
        return cleaned * 255
    
    def _remove_short_branches(self, skeleton: np.ndarray, min_length: int = 5) -> np.ndarray:
        """Remove short branches from skeleton."""
        # Find endpoints and junctions
        endpoints = self._find_endpoints(skeleton)
        junctions = self._find_junctions(skeleton)
        
        # For each endpoint, trace back and remove if branch is too short
        result = skeleton.copy()
        
        for y, x in endpoints:
            if not result[y, x]:  # Already removed
                continue
                
            # Trace from endpoint to junction or another endpoint
            path = self._trace_from_point(result, (x, y))
            
            if len(path) < min_length:
                # Remove this short branch
                for px, py in path:
                    result[py, px] = 0
        
        return result
    
    def _find_endpoints(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find endpoints in skeleton (points with only one neighbor)."""
        endpoints = []
        h, w = skeleton.shape
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x]:
                    # Count 8-connected neighbors
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
                    if neighbors == 1:
                        endpoints.append((y, x))
        
        return endpoints
    
    def _find_junctions(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Find junction points in skeleton (points with 3+ neighbors)."""
        junctions = []
        h, w = skeleton.shape
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x]:
                    # Count 8-connected neighbors
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
                    if neighbors >= 3:
                        junctions.append((y, x))
        
        return junctions
    
    def _trace_from_point(self, skeleton: np.ndarray, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Trace path from a starting point until junction or endpoint."""
        path = [start]
        current = start
        visited = {start}
        
        while True:
            x, y = current
            neighbors = []
            
            # Find unvisited neighbors
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < skeleton.shape[1] and 
                        0 <= ny < skeleton.shape[0] and
                        skeleton[ny, nx] and 
                        (nx, ny) not in visited):
                        neighbors.append((nx, ny))
            
            if len(neighbors) == 0:
                # Dead end
                break
            elif len(neighbors) == 1:
                # Continue tracing
                current = neighbors[0]
                path.append(current)
                visited.add(current)
            else:
                # Junction reached
                break
        
        return path


class PathTracer:
    """Traces paths through skeleton to generate ordered stroke points."""
    
    def __init__(self):
        pass
    
    def trace_skeleton(self, skeleton: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Trace skeleton to extract ordered paths."""
        # Convert skeleton to graph
        graph = self._skeleton_to_graph(skeleton)
        
        # Find connected components
        components = list(nx.connected_components(graph))
        
        paths = []
        for component in components:
            # Extract paths from each component
            component_paths = self._extract_paths_from_component(graph, component)
            paths.extend(component_paths)
        
        return paths
    
    def _skeleton_to_graph(self, skeleton: np.ndarray) -> nx.Graph:
        """Convert skeleton image to networkx graph."""
        graph = nx.Graph()
        h, w = skeleton.shape
        
        # Add nodes for skeleton pixels
        for y in range(h):
            for x in range(w):
                if skeleton[y, x]:
                    graph.add_node((x, y))
        
        # Add edges between adjacent skeleton pixels
        for y in range(h):
            for x in range(w):
                if skeleton[y, x]:
                    # Check 8-connected neighbors
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            
                            nx_pos, ny_pos = x + dx, y + dy
                            if (0 <= nx_pos < w and 
                                0 <= ny_pos < h and 
                                skeleton[ny_pos, nx_pos]):
                                graph.add_edge((x, y), (nx_pos, ny_pos))
        
        return graph
    
    def _extract_paths_from_component(self, graph: nx.Graph, component: set) -> List[List[Tuple[int, int]]]:
        """Extract paths from a connected component."""
        subgraph = graph.subgraph(component)
        
        # Find endpoints (degree 1) and junctions (degree > 2)
        endpoints = [node for node in subgraph.nodes() if subgraph.degree(node) == 1]
        junctions = [node for node in subgraph.nodes() if subgraph.degree(node) > 2]
        
        paths = []
        
        if len(endpoints) >= 2:
            # Trace paths between endpoints
            for i, start in enumerate(endpoints):
                for end in endpoints[i+1:]:
                    try:
                        path = nx.shortest_path(subgraph, start, end)
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        elif len(endpoints) == 1 and len(junctions) > 0:
            # Trace from endpoint to junctions
            start = endpoints[0]
            for junction in junctions:
                try:
                    path = nx.shortest_path(subgraph, start, junction)
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        else:
            # No clear endpoints, trace the entire component
            # Use DFS to get a path through all nodes
            if component:
                start_node = next(iter(component))
                path = list(nx.dfs_preorder_nodes(subgraph, start_node))
                paths.append(path)
        
        return paths


class StrokeGenerator:
    """Generates time-series stroke data from traced paths."""
    
    def __init__(self, base_velocity: float = 100.0):
        self.base_velocity = base_velocity  # pixels per second
    
    def generate_strokes(self, paths: List[List[Tuple[int, int]]]) -> List[List[Point]]:
        """Generate stroke data from traced paths."""
        strokes = []
        current_time = 0.0
        
        for path in paths:
            if len(path) < 2:
                continue
            
            stroke = self._path_to_stroke(path, current_time)
            strokes.append(stroke)
            
            # Update time for next stroke (add gap between strokes)
            if stroke:
                current_time = stroke[-1].time + 0.1  # 100ms gap
        
        return strokes
    
    def _path_to_stroke(self, path: List[Tuple[int, int]], start_time: float) -> List[Point]:
        """Convert a path to stroke points with timestamps."""
        if len(path) < 2:
            return []
        
        stroke = []
        current_time = start_time
        
        for i, (x, y) in enumerate(path):
            if i == 0:
                # First point
                dt = 0
            else:
                # Calculate distance to previous point
                prev_x, prev_y = path[i-1]
                distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                
                # Calculate time based on distance and velocity
                # Slow down at high curvature points
                curvature = self._calculate_curvature(path, i)
                velocity = self.base_velocity * (1.0 - 0.5 * curvature)  # Slow down at curves
                
                dt = distance / velocity if velocity > 0 else 0.01
            
            current_time += dt
            
            # Calculate pressure based on position in stroke
            pressure = self._calculate_pressure(i, len(path))
            
            stroke.append(Point(x=float(x), y=float(y), time=current_time, pressure=pressure))
        
        return stroke
    
    def _calculate_curvature(self, path: List[Tuple[int, int]], index: int) -> float:
        """Calculate normalized curvature at a point (0 = straight, 1 = sharp turn)."""
        if index == 0 or index == len(path) - 1:
            return 0.0
        
        # Get three consecutive points
        p1 = np.array(path[index - 1])
        p2 = np.array(path[index])
        p3 = np.array(path[index + 1])
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate angle between vectors
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norms == 0:
            return 0.0
        
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Normalize to 0-1 (0 = straight line, 1 = 180-degree turn)
        curvature = angle / np.pi
        
        return curvature
    
    def _calculate_pressure(self, index: int, total_points: int) -> float:
        """Calculate pressure based on position in stroke."""
        # Start and end with lower pressure, peak in middle
        t = index / (total_points - 1) if total_points > 1 else 0
        
        # Bell curve-like pressure profile
        pressure = 0.5 + 0.4 * np.exp(-4 * (t - 0.5)**2) + 0.1 * np.random.random()
        
        return np.clip(pressure, 0.1, 1.0)


class ImageToStrokeConverter:
    """Main class that orchestrates the image-to-stroke conversion pipeline."""
    
    def __init__(self, target_size: int = 128, base_velocity: float = 100.0):
        self.preprocessor = ImagePreprocessor(target_size)
        self.skeleton_extractor = SkeletonExtractor()
        self.path_tracer = PathTracer()
        self.stroke_generator = StrokeGenerator(base_velocity)
    
    def convert_image_to_strokes(
        self, 
        image_path: str, 
        label: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convert an image to stroke data."""
        start_time = datetime.now()
        
        try:
            # Load and preprocess image
            raw_image = self.preprocessor.load_image(image_path)
            original_size = raw_image.shape
            
            preprocessed = self.preprocessor.preprocess_image(raw_image)
            
            # Extract skeleton
            skeleton = self.skeleton_extractor.extract_skeleton(preprocessed)
            
            # Trace paths
            paths = self.path_tracer.trace_skeleton(skeleton)
            
            # Generate strokes
            strokes = self.stroke_generator.generate_strokes(paths)
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            metrics = ProcessingMetrics(
                processing_time_ms=processing_time,
                original_size=original_size,
                skeleton_points=np.sum(skeleton > 0),
                stroke_points=sum(len(stroke) for stroke in strokes),
                branches_detected=len(paths),
                quality_score=self._calculate_quality_score(strokes, skeleton)
            )
            
            # Convert to output format
            result = {
                "success": True,
                "label": label,
                "strokes": [
                    {
                        "id": f"stroke_{i}",
                        "points": [
                            {
                                "x": p.x,
                                "y": p.y, 
                                "time": p.time,
                                "pressure": p.pressure
                            } for p in stroke
                        ],
                        "completed": True,
                        "color": "#3B82F6"
                    } for i, stroke in enumerate(strokes)
                ],
                "metrics": {
                    "processing_time_ms": metrics.processing_time_ms,
                    "original_size": metrics.original_size,
                    "skeleton_points": metrics.skeleton_points,
                    "stroke_points": metrics.stroke_points,
                    "branches_detected": metrics.branches_detected,
                    "quality_score": metrics.quality_score
                },
                "processing_metadata": {
                    "algorithm_version": "1.0",
                    "target_size": self.preprocessor.target_size,
                    "base_velocity": self.stroke_generator.base_velocity,
                    "timestamp": start_time.isoformat()
                }
            }
            
            logger.info(f"Successfully converted image {image_path} to {len(strokes)} strokes in {processing_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error converting image {image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": start_time.isoformat()
            }
    
    def _calculate_quality_score(self, strokes: List[List[Point]], skeleton: np.ndarray) -> float:
        """Calculate quality score for the conversion (0-1)."""
        if not strokes:
            return 0.0
        
        # Factors affecting quality:
        # 1. Number of stroke points vs skeleton points
        total_stroke_points = sum(len(stroke) for stroke in strokes)
        skeleton_points = np.sum(skeleton > 0)
        
        point_ratio = min(1.0, total_stroke_points / max(1, skeleton_points))
        
        # 2. Number of strokes (prefer fewer, more continuous strokes)
        stroke_penalty = min(1.0, 5.0 / max(1, len(strokes)))
        
        # 3. Average stroke length (prefer longer strokes)
        avg_stroke_length = total_stroke_points / len(strokes) if strokes else 0
        length_score = min(1.0, avg_stroke_length / 20.0)  # Normalize by expected length
        
        # Combine factors
        quality_score = (point_ratio * 0.4 + stroke_penalty * 0.3 + length_score * 0.3)
        
        return np.clip(quality_score, 0.0, 1.0)
    
    def convert_dataset(
        self, 
        dataset_path: str, 
        output_path: str,
        file_pattern: str = "**/*.png"
    ) -> Dict[str, Any]:
        """Convert an entire dataset of images to strokes."""
        dataset_dir = Path(dataset_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = list(dataset_dir.glob(file_pattern))
        
        results = {
            "total_files": len(image_files),
            "processed_files": 0,
            "failed_files": 0,
            "output_files": [],
            "errors": []
        }
        
        for image_file in image_files:
            try:
                # Extract label from filename or directory structure
                label = self._extract_label_from_path(image_file)
                
                # Convert image
                result = self.convert_image_to_strokes(str(image_file), label)
                
                if result["success"]:
                    # Save result
                    output_file = output_dir / f"{image_file.stem}_strokes.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    results["output_files"].append(str(output_file))
                    results["processed_files"] += 1
                else:
                    results["failed_files"] += 1
                    results["errors"].append({
                        "file": str(image_file),
                        "error": result.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                results["failed_files"] += 1
                results["errors"].append({
                    "file": str(image_file),
                    "error": str(e)
                })
        
        logger.info(f"Dataset conversion complete: {results['processed_files']}/{results['total_files']} files processed")
        return results

    async def convert_dataset_with_progress(
        self,
        dataset_path: str,
        output_path: str,
        import_id: str,
        db,
        sample_size: Optional[int] = None,
        labels_filter: Optional[List[str]] = None,
        min_quality: float = 0.0,
        file_pattern: str = "**/*.png"
    ) -> Dict[str, Any]:
        """Convert dataset with real-time progress updates."""
        from app.db.crud import import_crud
        
        dataset_dir = Path(dataset_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        all_image_files = list(dataset_dir.glob(file_pattern))
        
        # Apply sample size if specified
        if sample_size and sample_size < len(all_image_files):
            import random
            random.seed(42)  # For reproducible sampling
            image_files = random.sample(all_image_files, sample_size)
            logger.info(f"Sampling {sample_size} files from {len(all_image_files)} total files")
        else:
            image_files = all_image_files
        
        # Apply label filter if specified
        if labels_filter:
            filtered_files = []
            for image_file in image_files:
                label = self._extract_label_from_path(image_file)
                if label.upper() in [l.upper() for l in labels_filter]:
                    filtered_files.append(image_file)
            image_files = filtered_files
            logger.info(f"Filtered to {len(image_files)} files with labels: {labels_filter}")
        
        results = {
            "total_files": len(image_files),
            "processed_files": 0,
            "failed_files": 0,
            "output_files": [],
            "errors": [],
            "labels_processed": set()
        }
        
        # Update initial status
        import_record = import_crud.get_by_import_id(db, import_id)
        if import_record:
            import_record.total_files = len(image_files)
            import_record.processed_files = 0
            db.commit()
        
        for i, image_file in enumerate(image_files):
            try:
                # Extract label from filename or directory structure
                label = self._extract_label_from_path(image_file)
                
                # Convert image
                result = self.convert_image_to_strokes(str(image_file), label)
                
                if result["success"] and result.get("metrics", {}).get("quality_score", 0) >= min_quality:
                    # Save result
                    output_file = output_dir / f"{image_file.stem}_strokes.json"
                    with open(output_file, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    results["output_files"].append(str(output_file))
                    results["processed_files"] += 1
                    results["labels_processed"].add(label)
                    
                    # Save to database
                    stroke_data = {
                        "label": label,
                        "source_file": str(image_file),
                        "source_type": "imported",
                        "quality_score": float(result.get("metrics", {}).get("quality_score", 0)),
                        "stroke_data": result["strokes"],
                        "processing_metrics": {
                            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                            for k, v in result["metrics"].items()
                        }
                    }
                    
                    from app.db.crud import stroke_crud
                    stroke_crud.create(db, stroke_data)
                    
                else:
                    results["failed_files"] += 1
                    error_msg = result.get("error", "Quality below threshold") if not result["success"] else "Quality below threshold"
                    results["errors"].append({
                        "file": str(image_file),
                        "error": error_msg
                    })
                    
            except Exception as e:
                results["failed_files"] += 1
                results["errors"].append({
                    "file": str(image_file),
                    "error": str(e)
                })
            
            # Update progress every 10 files or on last file
            if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                progress_percentage = ((i + 1) / len(image_files)) * 100
                import_record = import_crud.get_by_import_id(db, import_id)
                if import_record:
                    import_record.processed_files = i + 1
                    import_record.progress_percentage = progress_percentage
                    import_record.current_file = str(image_file)
                    db.commit()
                
                logger.info(f"Progress: {i + 1}/{len(image_files)} ({progress_percentage:.1f}%) - Current: {image_file.name}")
        
        # Final update
        import_record = import_crud.get_by_import_id(db, import_id)
        if import_record:
            import_record.processed_files = results["processed_files"]
            import_record.failed_files = results["failed_files"]
            import_record.success_rate = (
                results["processed_files"] / results["total_files"] * 100
                if results["total_files"] > 0 else 0
            )
            import_record.processing_errors = results["errors"]
            import_record.labels_processed = list(results["labels_processed"])
            db.commit()
        
        logger.info(f"Dataset conversion complete: {results['processed_files']}/{results['total_files']} files processed")
        return results
    
    def _extract_label_from_path(self, file_path: Path) -> str:
        """Extract label from file path or filename."""
        # Try to extract from parent directory name
        parent_name = file_path.parent.name
        if len(parent_name) == 1 and parent_name.isalnum():
            return parent_name.upper()
        
        # Try to extract from filename
        filename = file_path.stem
        if len(filename) >= 1:
            # Look for single character in filename
            for char in filename:
                if char.isalnum():
                    return char.upper()
        
        # Default fallback
        return "UNKNOWN"


# Convenience functions
def convert_image_to_strokes(image_path: str, label: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to convert a single image."""
    converter = ImageToStrokeConverter()
    return converter.convert_image_to_strokes(image_path, label)


def convert_dataset(dataset_path: str, output_path: str) -> Dict[str, Any]:
    """Convenience function to convert a dataset."""
    converter = ImageToStrokeConverter()
    return converter.convert_dataset(dataset_path, output_path)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    converter = ImageToStrokeConverter()
    
    # Convert single image
    result = converter.convert_image_to_strokes("sample_A.png", "A")
    
    if result["success"]:
        print(f"Successfully converted image:")
        print(f"  - {len(result['strokes'])} strokes generated")
        print(f"  - {result['metrics']['stroke_points']} total points")
        print(f"  - Processing time: {result['metrics']['processing_time_ms']:.1f}ms")
        print(f"  - Quality score: {result['metrics']['quality_score']:.3f}")
    else:
        print(f"Conversion failed: {result['error']}")
