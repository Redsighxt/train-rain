import type { Point, ProcessedPoint, LandmarkPoint } from '@/store/researchStore';

/**
 * Advanced Curvature Analysis Module
 * Implements sophisticated curvature calculations, inflection point detection,
 * and landmark identification for handwriting analysis
 */

export interface CurvatureMetrics {
  pointwiseCurvature: number[];
  normalizedCurvature: number[];
  curvatureSignature: Array<{ s: number; kappa: number }>;
  inflectionPoints: number[];
  curvatureExtrema: { maxima: number[]; minima: number[] };
  totalCurvature: number;
  meanCurvature: number;
  curvatureVariance: number;
  curvatureEntropy: number;
  signatureComplexity: number;
}

export interface InflectionAnalysis {
  inflectionPoints: Array<{
    index: number;
    position: Point;
    curvatureChange: number;
    localComplexity: number;
    stability: number;
  }>;
  inflectionDensity: number;
  maxCurvatureChange: number;
  inflectionPattern: string;
}

/**
 * Calculate curvature using multiple methods for robustness
 */
export function calculateAdvancedCurvature(points: Point[]): {
  discrete: number[];
  continuous: number[];
  robust: number[];
} {
  if (points.length < 3) {
    return { discrete: [], continuous: [], robust: [] };
  }

  const discrete = calculateDiscreteCurvature(points);
  const continuous = calculateContinuousCurvature(points);
  const robust = calculateRobustCurvature(points);

  return { discrete, continuous, robust };
}

/**
 * Discrete curvature using finite differences
 */
function calculateDiscreteCurvature(points: Point[]): number[] {
  const curvatures: number[] = [];

  for (let i = 1; i < points.length - 1; i++) {
    const p1 = points[i - 1];
    const p2 = points[i];
    const p3 = points[i + 1];

    // Calculate first derivatives
    const dx1 = p2.x - p1.x;
    const dy1 = p2.y - p1.y;
    const dx2 = p3.x - p2.x;
    const dy2 = p3.y - p2.y;

    // Calculate second derivatives
    const ddx = dx2 - dx1;
    const ddy = dy2 - dy1;

    // Curvature formula: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    const numerator = dx1 * ddy - dy1 * ddx;
    const denominator = Math.pow(dx1 * dx1 + dy1 * dy1, 1.5);

    const curvature = denominator > 0 ? numerator / denominator : 0;
    curvatures.push(curvature);
  }

  return curvatures;
}

/**
 * Continuous curvature using spline fitting
 */
function calculateContinuousCurvature(points: Point[]): number[] {
  const curvatures: number[] = [];

  // Use a sliding window approach with polynomial fitting
  const windowSize = Math.min(5, points.length);

  for (let i = 2; i < points.length - 2; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(points.length, start + windowSize);
    const localPoints = points.slice(start, end);

    if (localPoints.length >= 3) {
      const localCurvature = calculateLocalPolynomialCurvature(localPoints, i - start);
      curvatures.push(localCurvature);
    } else {
      curvatures.push(0);
    }
  }

  return curvatures;
}

/**
 * Robust curvature using RANSAC-style outlier rejection
 */
function calculateRobustCurvature(points: Point[]): number[] {
  const discreteCurvatures = calculateDiscreteCurvature(points);
  const robustCurvatures: number[] = [];

  // Apply median filtering to remove outliers
  const windowSize = 3;

  for (let i = 0; i < discreteCurvatures.length; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(discreteCurvatures.length, start + windowSize);
    const window = discreteCurvatures.slice(start, end);

    window.sort((a, b) => a - b);
    const median = window[Math.floor(window.length / 2)];
    robustCurvatures.push(median);
  }

  return robustCurvatures;
}

/**
 * Local polynomial curvature calculation
 */
function calculateLocalPolynomialCurvature(points: Point[], centerIndex: number): number {
  if (points.length < 3 || centerIndex < 0 || centerIndex >= points.length) return 0;

  // Fit a second-degree polynomial and calculate curvature
  const n = points.length;
  let sumX = 0, sumY = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
  let sumXY = 0, sumX2Y = 0;

  // Normalize x values around the center point
  const centerX = points[centerIndex].x;

  for (let i = 0; i < n; i++) {
    const x = points[i].x - centerX;
    const y = points[i].y;
    const x2 = x * x;
    const x3 = x2 * x;
    const x4 = x2 * x2;

    sumX += x;
    sumY += y;
    sumX2 += x2;
    sumX3 += x3;
    sumX4 += x4;
    sumXY += x * y;
    sumX2Y += x2 * y;
  }

  // Solve for polynomial coefficients using least squares
  // y = a + bx + cx²
  const denominator = n * sumX2 * sumX4 - n * sumX3 * sumX3 - sumX * sumX * sumX4 + 
                     2 * sumX * sumX2 * sumX3 - sumX2 * sumX2 * sumX2;

  if (Math.abs(denominator) < 1e-10) return 0;

  const c = (n * sumX2Y * sumX2 - n * sumXY * sumX3 - sumX * sumX2Y * sumX + 
            sumX * sumXY * sumX2 + sumX2 * sumY * sumX3 - sumX2 * sumY * sumX2) / denominator;

  // Curvature at x=0 (center point) is 2c / (1 + b²)^(3/2)
  // For simplicity, approximate as 2c when b is small
  return 2 * c;
}

/**
 * Detect inflection points where curvature changes sign
 */
export function detectInflectionPoints(curvatures: number[]): InflectionAnalysis {
  const inflectionPoints: Array<{
    index: number;
    position: Point;
    curvatureChange: number;
    localComplexity: number;
    stability: number;
  }> = [];

  if (curvatures.length < 3) {
    return {
      inflectionPoints: [],
      inflectionDensity: 0,
      maxCurvatureChange: 0,
      inflectionPattern: 'none'
    };
  }

  let maxCurvatureChange = 0;

  for (let i = 1; i < curvatures.length - 1; i++) {
    const prev = curvatures[i - 1];
    const curr = curvatures[i];
    const next = curvatures[i + 1];

    // Check for sign change (inflection)
    const hasInflection = (prev * next < 0) || 
                         (Math.abs(curr) < 0.01 && Math.abs(prev - next) > 0.1);

    if (hasInflection) {
      const curvatureChange = Math.abs(next - prev);
      maxCurvatureChange = Math.max(maxCurvatureChange, curvatureChange);

      // Calculate local complexity in a window around the inflection
      const windowSize = 5;
      const start = Math.max(0, i - windowSize);
      const end = Math.min(curvatures.length, i + windowSize + 1);
      const localWindow = curvatures.slice(start, end);
      
      const localMean = localWindow.reduce((sum, c) => sum + c, 0) / localWindow.length;
      const localVariance = localWindow.reduce((sum, c) => sum + (c - localMean) ** 2, 0) / localWindow.length;
      const localComplexity = Math.sqrt(localVariance);

      // Stability based on how consistent this inflection is across multiple scales
      const stability = calculateInflectionStability(curvatures, i);

      inflectionPoints.push({
        index: i + 1, // Adjust for offset in curvature array
        position: { x: 0, y: 0, time: 0 }, // Will be filled in by caller
        curvatureChange,
        localComplexity,
        stability
      });
    }
  }

  const inflectionDensity = inflectionPoints.length / curvatures.length;
  const inflectionPattern = classifyInflectionPattern(inflectionPoints);

  return {
    inflectionPoints,
    inflectionDensity,
    maxCurvatureChange,
    inflectionPattern
  };
}

/**
 * Calculate stability of an inflection point
 */
function calculateInflectionStability(curvatures: number[], inflectionIndex: number): number {
  // Check if the inflection persists at different smoothing levels
  let stabilityScore = 0;
  const tests = [1, 3, 5]; // Different smoothing window sizes

  for (const windowSize of tests) {
    const smoothed = applySmoothingWindow(curvatures, windowSize);
    
    if (inflectionIndex < smoothed.length - 1) {
      const prev = smoothed[Math.max(0, inflectionIndex - 1)];
      const next = smoothed[Math.min(smoothed.length - 1, inflectionIndex + 1)];
      
      if (prev * next < 0) {
        stabilityScore += 1 / tests.length;
      }
    }
  }

  return stabilityScore;
}

/**
 * Apply smoothing window to curvature array
 */
function applySmoothingWindow(curvatures: number[], windowSize: number): number[] {
  const smoothed: number[] = [];
  const halfWindow = Math.floor(windowSize / 2);

  for (let i = 0; i < curvatures.length; i++) {
    let sum = 0;
    let count = 0;

    for (let j = -halfWindow; j <= halfWindow; j++) {
      const index = i + j;
      if (index >= 0 && index < curvatures.length) {
        sum += curvatures[index];
        count++;
      }
    }

    smoothed.push(count > 0 ? sum / count : 0);
  }

  return smoothed;
}

/**
 * Classify inflection pattern
 */
function classifyInflectionPattern(inflectionPoints: Array<{ curvatureChange: number }>): string {
  if (inflectionPoints.length === 0) return 'none';
  if (inflectionPoints.length === 1) return 'single';
  if (inflectionPoints.length === 2) return 'double';
  if (inflectionPoints.length <= 4) return 'multiple';
  
  // Check for regularity
  const changes = inflectionPoints.map(p => p.curvatureChange);
  const meanChange = changes.reduce((sum, c) => sum + c, 0) / changes.length;
  const variance = changes.reduce((sum, c) => sum + (c - meanChange) ** 2, 0) / changes.length;
  const coefficientOfVariation = Math.sqrt(variance) / meanChange;

  return coefficientOfVariation < 0.5 ? 'regular' : 'irregular';
}

/**
 * Find curvature extrema (maxima and minima)
 */
export function findCurvatureExtrema(curvatures: number[]): { maxima: number[]; minima: number[] } {
  const maxima: number[] = [];
  const minima: number[] = [];

  if (curvatures.length < 3) return { maxima, minima };

  for (let i = 1; i < curvatures.length - 1; i++) {
    const prev = curvatures[i - 1];
    const curr = curvatures[i];
    const next = curvatures[i + 1];

    // Local maximum
    if (curr > prev && curr > next && Math.abs(curr) > 0.01) {
      maxima.push(i + 1); // Adjust for offset
    }
    
    // Local minimum  
    if (curr < prev && curr < next && Math.abs(curr) > 0.01) {
      minima.push(i + 1); // Adjust for offset
    }
  }

  return { maxima, minima };
}

/**
 * Create normalized curvature signature
 */
export function createCurvatureSignature(
  points: Point[], 
  curvatures: number[]
): Array<{ s: number; kappa: number }> {
  if (points.length !== curvatures.length + 2) {
    throw new Error('Curvature array length mismatch');
  }

  const signature: Array<{ s: number; kappa: number }> = [];
  
  // Calculate arc length
  let totalLength = 0;
  const lengths = [0];
  
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    const segmentLength = Math.sqrt(dx * dx + dy * dy);
    totalLength += segmentLength;
    lengths.push(totalLength);
  }

  // Create signature points
  for (let i = 0; i < curvatures.length; i++) {
    const pointIndex = i + 1; // Curvature array is offset by 1
    const s = totalLength > 0 ? lengths[pointIndex] / totalLength : 0;
    const kappa = totalLength > 0 ? curvatures[i] * totalLength : curvatures[i];
    
    signature.push({ s, kappa });
  }

  return signature;
}

/**
 * Calculate curvature entropy (measure of complexity)
 */
export function calculateCurvatureEntropy(curvatures: number[]): number {
  if (curvatures.length === 0) return 0;

  // Discretize curvatures into bins
  const numBins = 20;
  const maxCurvature = Math.max(...curvatures.map(Math.abs));
  
  if (maxCurvature === 0) return 0;

  const binSize = (2 * maxCurvature) / numBins;
  const bins = new Array(numBins).fill(0);

  // Populate bins
  for (const curvature of curvatures) {
    const binIndex = Math.min(numBins - 1, 
      Math.floor((curvature + maxCurvature) / binSize)
    );
    bins[binIndex]++;
  }

  // Calculate entropy
  const total = curvatures.length;
  let entropy = 0;

  for (const count of bins) {
    if (count > 0) {
      const probability = count / total;
      entropy -= probability * Math.log2(probability);
    }
  }

  return entropy;
}

/**
 * Main function to compute comprehensive curvature metrics
 */
export function computeCurvatureMetrics(points: Point[]): CurvatureMetrics {
  if (points.length < 3) {
    return {
      pointwiseCurvature: [],
      normalizedCurvature: [],
      curvatureSignature: [],
      inflectionPoints: [],
      curvatureExtrema: { maxima: [], minima: [] },
      totalCurvature: 0,
      meanCurvature: 0,
      curvatureVariance: 0,
      curvatureEntropy: 0,
      signatureComplexity: 0
    };
  }

  // Calculate multiple curvature estimates and use the robust one
  const curvatureData = calculateAdvancedCurvature(points);
  const curvatures = curvatureData.robust;

  // Normalize curvatures
  const maxAbsCurvature = Math.max(...curvatures.map(Math.abs));
  const normalizedCurvature = maxAbsCurvature > 0 ? 
    curvatures.map(k => k / maxAbsCurvature) : curvatures;

  // Create signature
  const curvatureSignature = createCurvatureSignature(points, curvatures);

  // Find special points
  const inflectionAnalysis = detectInflectionPoints(curvatures);
  const extrema = findCurvatureExtrema(curvatures);

  // Calculate statistics
  const totalCurvature = curvatures.reduce((sum, k) => sum + Math.abs(k), 0);
  const meanCurvature = curvatures.length > 0 ? 
    curvatures.reduce((sum, k) => sum + k, 0) / curvatures.length : 0;
  
  const curvatureVariance = curvatures.length > 0 ?
    curvatures.reduce((sum, k) => sum + (k - meanCurvature) ** 2, 0) / curvatures.length : 0;

  const curvatureEntropy = calculateCurvatureEntropy(curvatures);

  // Signature complexity based on variation and inflections
  const signatureComplexity = Math.sqrt(curvatureVariance) + 
                             inflectionAnalysis.inflectionDensity * 5 +
                             (extrema.maxima.length + extrema.minima.length) * 0.1;

  return {
    pointwiseCurvature: curvatures,
    normalizedCurvature,
    curvatureSignature,
    inflectionPoints: inflectionAnalysis.inflectionPoints.map(p => p.index),
    curvatureExtrema: extrema,
    totalCurvature,
    meanCurvature,
    curvatureVariance,
    curvatureEntropy,
    signatureComplexity
  };
}

/**
 * Extract landmark points based on curvature analysis
 */
export function extractCurvatureLandmarks(
  points: Point[], 
  curvatureMetrics: CurvatureMetrics
): LandmarkPoint[] {
  const landmarks: LandmarkPoint[] = [];

  // Add inflection points as landmarks
  curvatureMetrics.inflectionPoints.forEach((index, i) => {
    if (index < points.length) {
      landmarks.push({
        id: `inflection-${i}`,
        type: 'inflection',
        position: {
          ...points[index],
          s_L: index / (points.length - 1),
          kappa_L: curvatureMetrics.normalizedCurvature[index - 1] || 0,
          velocity: 0,
          acceleration: 0,
          tangentAngle: 0
        },
        magnitude: Math.abs(curvatureMetrics.pointwiseCurvature[index - 1] || 0),
        stability: 0.8, // Default stability for inflection points
      });
    }
  });

  // Add curvature maxima as landmarks
  curvatureMetrics.curvatureExtrema.maxima.forEach((index, i) => {
    if (index < points.length) {
      landmarks.push({
        id: `curvature-max-${i}`,
        type: 'peak',
        position: {
          ...points[index],
          s_L: index / (points.length - 1),
          kappa_L: curvatureMetrics.normalizedCurvature[index - 1] || 0,
          velocity: 0,
          acceleration: 0,
          tangentAngle: 0
        },
        magnitude: Math.abs(curvatureMetrics.pointwiseCurvature[index - 1] || 0),
        stability: 0.9, // High stability for curvature maxima
      });
    }
  });

  // Add endpoints as landmarks
  landmarks.push({
    id: 'start-point',
    type: 'endpoint',
    position: {
      ...points[0],
      s_L: 0,
      kappa_L: 0,
      velocity: 0,
      acceleration: 0,
      tangentAngle: 0
    },
    magnitude: 1.0,
    stability: 1.0, // Highest stability for endpoints
  });

  landmarks.push({
    id: 'end-point',
    type: 'endpoint',
    position: {
      ...points[points.length - 1],
      s_L: 1,
      kappa_L: 0,
      velocity: 0,
      acceleration: 0,
      tangentAngle: 0
    },
    magnitude: 1.0,
    stability: 1.0, // Highest stability for endpoints
  });

  return landmarks;
}
