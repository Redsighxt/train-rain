// Mathematical Framework for 3D Stroke Signature Generation
// Based on advanced stroke analysis and invariant point detection

export interface Point {
  x: number;
  y: number;
  time: number;
  pressure?: number;
}

export interface Stroke {
  id: string;
  points: Point[];
  completed: boolean;
  color: string;
}

export interface InvariantPoint {
  id: string;
  index: number;
  strokeIndex: number;
  position: Point;
  stabilityScore: number;
  type: 'primary' | 'secondary' | 'tertiary';
  category: 'geometric' | 'topological' | 'statistical' | 'intersection';
  description: string;
  confidence: number;
}

export interface FeatureVector {
  // Positional features
  normalizedPositions: Point[];
  arcLengthParameterized: Point[];
  
  // Geometric features
  curvatureValues: number[];
  tangentAngles: number[];
  inflectionPoints: number[];
  
  // Temporal features
  velocityProfile: number[];
  accelerationProfile: number[];
  timingPattern: number[];
  
  // Directional features
  directionSequence: string[];
  angleChanges: number[];
  
  // Intersection features
  horizontalIntersections: number;
  verticalIntersections: number;
  diagonalIntersections: number;
  
  // Frequency domain features
  fourierMagnitudes: number[];
  fourierPhases: number[];
  spectralCentroid: number;
  
  // Topological features
  crossingNumber: number;
  windingNumber: number;
  
  // Statistical features
  complexityScore: number;
  regularityScore: number;
  symmetryScore: number;
  
  // Global features
  totalArcLength: number;
  boundingBoxRatio: number;
  centroid: Point;
}

export interface ThreeDSignature {
  points: Array<{
    x: number;
    y: number;
    z: number;
    originalIndex: number;
    strokeIndex: number;
    invarianceWeight: number;
  }>;
  invariantPoints: Array<{
    position: { x: number; y: number; z: number };
    type: string;
    stability: number;
  }>;
  signatureMetrics: {
    interClusterDistance: number;
    intraClusterVariance: number;
    separationRatio: number;
    stabilityIndex: number;
  };
}

/**
 * Normalize stroke points to unit square [0,1] x [0,1]
 */
export function normalizePoints(strokes: Stroke[]): Point[] {
  const allPoints = strokes.flatMap(stroke => stroke.points);
  if (allPoints.length === 0) return [];
  
  const xs = allPoints.map(p => p.x);
  const ys = allPoints.map(p => p.y);
  
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  
  const width = maxX - minX || 1;
  const height = maxY - minY || 1;
  
  return allPoints.map(p => ({
    ...p,
    x: (p.x - minX) / width,
    y: (p.y - minY) / height
  }));
}

/**
 * Calculate curvature at each point using discrete approximation
 */
export function calculateCurvature(points: Point[]): number[] {
  if (points.length < 3) return [];
  
  const curvatures: number[] = [];
  
  for (let i = 1; i < points.length - 1; i++) {
    const p1 = points[i - 1];
    const p2 = points[i];
    const p3 = points[i + 1];
    
    // Calculate vectors
    const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
    
    // Cross product for curvature
    const crossProduct = v1.x * v2.y - v1.y * v2.x;
    const v1Mag = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const v2Mag = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    
    const curvature = v1Mag > 0 && v2Mag > 0 ? crossProduct / (v1Mag * v2Mag) : 0;
    curvatures.push(curvature);
  }
  
  return curvatures;
}

/**
 * Calculate tangent angles along the stroke
 */
export function calculateTangentAngles(points: Point[]): number[] {
  if (points.length < 2) return [];
  
  const angles: number[] = [];
  
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    const angle = Math.atan2(dy, dx);
    angles.push(angle);
  }
  
  return angles;
}

/**
 * Calculate velocity profile
 */
export function calculateVelocity(points: Point[]): number[] {
  if (points.length < 2) return [];
  
  const velocities: number[] = [];
  
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    const dt = Math.max(points[i].time - points[i - 1].time, 1);
    
    const velocity = Math.sqrt(dx * dx + dy * dy) / dt;
    velocities.push(velocity);
  }
  
  return velocities;
}

/**
 * Simple FFT implementation for frequency analysis
 */
export function calculateFFT(signal: number[]): { magnitude: number[]; phase: number[] } {
  const n = Math.min(signal.length, 32); // Limit for performance
  const magnitudes: number[] = [];
  const phases: number[] = [];
  
  for (let k = 0; k < n / 2; k++) {
    let real = 0;
    let imag = 0;
    
    for (let j = 0; j < n; j++) {
      const angle = -2 * Math.PI * k * j / n;
      real += (signal[j] || 0) * Math.cos(angle);
      imag += (signal[j] || 0) * Math.sin(angle);
    }
    
    const magnitude = Math.sqrt(real * real + imag * imag) / n;
    const phase = Math.atan2(imag, real);
    
    magnitudes.push(magnitude);
    phases.push(phase);
  }
  
  return { magnitude: magnitudes, phase: phases };
}

/**
 * Extract comprehensive feature vector from strokes
 */
export function extractFeatureVector(strokes: Stroke[]): FeatureVector {
  const allPoints = normalizePoints(strokes);
  const curvatures = calculateCurvature(allPoints);
  const tangentAngles = calculateTangentAngles(allPoints);
  const velocities = calculateVelocity(allPoints);
  
  // Calculate arc length parameterization
  const arcLengths: number[] = [0];
  for (let i = 1; i < allPoints.length; i++) {
    const dx = allPoints[i].x - allPoints[i - 1].x;
    const dy = allPoints[i].y - allPoints[i - 1].y;
    const length = Math.sqrt(dx * dx + dy * dy);
    arcLengths.push(arcLengths[arcLengths.length - 1] + length);
  }
  
  const totalArcLength = arcLengths[arcLengths.length - 1] || 1;
  
  // Arc length parameterized points
  const arcLengthParameterized = allPoints.map((point, i) => ({
    ...point,
    t: arcLengths[i] / totalArcLength
  }));
  
  // Direction sequence
  const directionSequence = tangentAngles.map(angle => {
    const degrees = (angle * 180 / Math.PI + 360) % 360;
    if (degrees >= 315 || degrees < 45) return '→';
    else if (degrees >= 45 && degrees < 135) return '↓';
    else if (degrees >= 135 && degrees < 225) return '←';
    else return '↑';
  });
  
  // Acceleration
  const accelerations: number[] = [];
  for (let i = 1; i < velocities.length; i++) {
    const dv = velocities[i] - velocities[i - 1];
    const dt = Math.max(allPoints[i + 1].time - allPoints[i].time, 1);
    accelerations.push(dv / dt);
  }
  
  // Inflection points
  const inflectionPoints: number[] = [];
  for (let i = 1; i < curvatures.length - 1; i++) {
    if ((curvatures[i - 1] * curvatures[i + 1]) < 0) {
      inflectionPoints.push(i);
    }
  }
  
  // Intersection counts (simplified)
  let horizontalIntersections = 0;
  let verticalIntersections = 0;
  const referenceLines = [0.25, 0.5, 0.75];
  
  for (let i = 1; i < allPoints.length; i++) {
    const p1 = allPoints[i - 1];
    const p2 = allPoints[i];
    
    referenceLines.forEach(y => {
      if ((p1.y <= y && p2.y >= y) || (p1.y >= y && p2.y <= y)) {
        horizontalIntersections++;
      }
    });
    
    referenceLines.forEach(x => {
      if ((p1.x <= x && p2.x >= x) || (p1.x >= x && p2.x <= x)) {
        verticalIntersections++;
      }
    });
  }
  
  // FFT of tangent angles
  const fft = calculateFFT(tangentAngles);
  const spectralCentroid = fft.magnitude.length > 0 ?
    fft.magnitude.reduce((sum, mag, i) => sum + mag * i, 0) / fft.magnitude.reduce((sum, mag) => sum + mag, 1) : 0;
  
  // Complexity score
  const curvatureVariation = curvatures.length > 0 ?
    Math.sqrt(curvatures.reduce((sum, c) => sum + c * c, 0) / curvatures.length) : 0;
  
  const angleChanges = tangentAngles.slice(1).map((angle, i) => {
    const change = Math.abs(angle - tangentAngles[i]);
    return Math.min(change, 2 * Math.PI - change);
  });
  
  const complexityScore = Math.min(curvatureVariation * 10 + 
    (angleChanges.reduce((sum, change) => sum + change, 0) / Math.max(angleChanges.length, 1)), 10);
  
  // Regularity score (velocity consistency)
  const meanVelocity = velocities.reduce((sum, v) => sum + v, 0) / Math.max(velocities.length, 1);
  const velocityVariance = velocities.reduce((sum, v) => sum + Math.pow(v - meanVelocity, 2), 0) / Math.max(velocities.length, 1);
  const regularityScore = meanVelocity > 0 ? Math.max(0, 1 - Math.sqrt(velocityVariance) / meanVelocity) : 0;
  
  // Symmetry score (simplified horizontal symmetry)
  const centerX = 0.5;
  let symmetrySum = 0;
  allPoints.forEach(point => {
    const mirroredX = 2 * centerX - point.x;
    const minDistance = Math.min(...allPoints.map(p => 
      Math.sqrt((p.x - mirroredX) ** 2 + (p.y - point.y) ** 2)
    ));
    symmetrySum += Math.exp(-minDistance * 10);
  });
  const symmetryScore = allPoints.length > 0 ? symmetrySum / allPoints.length : 0;
  
  // Centroid
  const centroid = {
    x: allPoints.reduce((sum, p) => sum + p.x, 0) / Math.max(allPoints.length, 1),
    y: allPoints.reduce((sum, p) => sum + p.y, 0) / Math.max(allPoints.length, 1),
    time: 0
  };
  
  // Bounding box ratio
  const xs = allPoints.map(p => p.x);
  const ys = allPoints.map(p => p.y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  const boundingBoxRatio = height > 0 ? width / height : 1;
  
  return {
    normalizedPositions: allPoints,
    arcLengthParameterized,
    curvatureValues: curvatures,
    tangentAngles,
    inflectionPoints,
    velocityProfile: velocities,
    accelerationProfile: accelerations,
    timingPattern: allPoints.map(p => p.time),
    directionSequence,
    angleChanges,
    horizontalIntersections,
    verticalIntersections,
    diagonalIntersections: 0, // Simplified
    fourierMagnitudes: fft.magnitude,
    fourierPhases: fft.phase,
    spectralCentroid,
    crossingNumber: 0, // Simplified
    windingNumber: tangentAngles.reduce((sum, angle) => sum + angle, 0) / (2 * Math.PI),
    complexityScore,
    regularityScore,
    symmetryScore,
    totalArcLength,
    boundingBoxRatio,
    centroid
  };
}

/**
 * Generate 3D signature from feature vector using optimized coordinate mapping
 */
export function generate3DSignature(
  strokes: Stroke[], 
  invariantPoints: InvariantPoint[]
): ThreeDSignature {
  const featureVector = extractFeatureVector(strokes);
  const points: ThreeDSignature['points'] = [];
  
  strokes.forEach((stroke, strokeIndex) => {
    const normalizedPoints = normalizePoints([stroke]);
    const curvatures = calculateCurvature(normalizedPoints);
    const tangentAngles = calculateTangentAngles(normalizedPoints);
    const velocities = calculateVelocity(normalizedPoints);
    
    normalizedPoints.forEach((point, index) => {
      // Advanced 3D mapping function
      // X-axis: Geometric complexity + intersection patterns
      const curvature = curvatures[index - 1] || 0;
      const intersectionWeight = (featureVector.horizontalIntersections + featureVector.verticalIntersections) / 10;
      const x = Math.abs(curvature) * 5 + 
                featureVector.complexityScore * 0.5 + 
                intersectionWeight * 0.3 +
                point.x * 2;
      
      // Y-axis: Temporal patterns + positional distribution
      const velocity = velocities[index - 1] || 0;
      const arcPosition = index / Math.max(normalizedPoints.length - 1, 1);
      const y = point.y * 4 + 
                Math.min(velocity * 1000, 2) + 
                arcPosition * 2 +
                featureVector.regularityScore * 1;
      
      // Z-axis: Direction patterns + frequency components
      const tangentAngle = tangentAngles[index - 1] || 0;
      const angleComponent = (Math.sin(tangentAngle) + 1) * 2;
      const frequencyComponent = featureVector.spectralCentroid * 0.5;
      const z = angleComponent + 
                frequencyComponent +
                featureVector.symmetryScore * 2 +
                (point.x + point.y) * 0.5;
      
      // Calculate invariance weight based on proximity to invariant points
      let invarianceWeight = 0.1;
      invariantPoints.forEach(invPoint => {
        if (invPoint.strokeIndex === strokeIndex) {
          const distance = Math.abs(invPoint.index - index);
          if (distance < 3) {
            invarianceWeight = Math.max(invarianceWeight, invPoint.stabilityScore * (3 - distance) / 3);
          }
        }
      });
      
      points.push({
        x: x * 2, // Scale for better visualization
        y: y * 2,
        z: z * 2,
        originalIndex: index,
        strokeIndex,
        invarianceWeight
      });
    });
  });
  
  // Map invariant points to 3D space
  const invariant3DPoints = invariantPoints.map(invPoint => {
    const strokePoints = points.filter(p => p.strokeIndex === invPoint.strokeIndex);
    const point3D = strokePoints[invPoint.index];
    
    return {
      position: point3D ? { x: point3D.x, y: point3D.y, z: point3D.z } : { x: 0, y: 0, z: 0 },
      type: invPoint.type,
      stability: invPoint.stabilityScore
    };
  });
  
  // Calculate signature metrics
  const signatureMetrics = calculateSignatureMetrics(points, invariant3DPoints);
  
  return {
    points,
    invariantPoints: invariant3DPoints,
    signatureMetrics
  };
}

/**
 * Calculate quality metrics for the 3D signature
 */
function calculateSignatureMetrics(
  points: ThreeDSignature['points'],
  invariantPoints: Array<{ position: { x: number; y: number; z: number }; stability: number }>
) {
  if (points.length < 2) {
    return {
      interClusterDistance: 0,
      intraClusterVariance: 0,
      separationRatio: 0,
      stabilityIndex: 0
    };
  }
  
  // Calculate centroid
  const centroid = {
    x: points.reduce((sum, p) => sum + p.x, 0) / points.length,
    y: points.reduce((sum, p) => sum + p.y, 0) / points.length,
    z: points.reduce((sum, p) => sum + p.z, 0) / points.length
  };
  
  // Intra-cluster variance (spread of points)
  const intraClusterVariance = points.reduce((sum, p) => {
    const dx = p.x - centroid.x;
    const dy = p.y - centroid.y;
    const dz = p.z - centroid.z;
    return sum + Math.sqrt(dx * dx + dy * dy + dz * dz);
  }, 0) / points.length;
  
  // Stability index (average of invariant point stabilities)
  const stabilityIndex = invariantPoints.length > 0 ?
    invariantPoints.reduce((sum, p) => sum + p.stability, 0) / invariantPoints.length : 0;
  
  // Inter-cluster distance (simplified - would need multiple letter instances)
  const interClusterDistance = 10; // Placeholder - would be calculated from multiple letter samples
  
  // Separation ratio
  const separationRatio = intraClusterVariance > 0 ? interClusterDistance / intraClusterVariance : 0;
  
  return {
    interClusterDistance,
    intraClusterVariance,
    separationRatio,
    stabilityIndex
  };
}

/**
 * Compare two 3D signatures for similarity
 */
export function compareSignatures(sig1: ThreeDSignature, sig2: ThreeDSignature): number {
  // Simplified signature comparison based on key metrics
  const weightedDistance = 
    Math.abs(sig1.signatureMetrics.stabilityIndex - sig2.signatureMetrics.stabilityIndex) * 0.4 +
    Math.abs(sig1.signatureMetrics.separationRatio - sig2.signatureMetrics.separationRatio) * 0.3 +
    Math.abs(sig1.signatureMetrics.intraClusterVariance - sig2.signatureMetrics.intraClusterVariance) * 0.3;
  
  // Return similarity score (1 = identical, 0 = completely different)
  return Math.max(0, 1 - weightedDistance / 10);
}
