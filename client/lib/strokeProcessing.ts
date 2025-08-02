import type { Point, ProcessedPoint } from '@/store/researchStore';

/**
 * Advanced Stroke Processing Algorithms
 * Implements smooth interpolation, normalization, and preprocessing
 */

export interface SmoothingOptions {
  type: 'none' | 'gaussian' | 'movingAverage' | 'catmullRom';
  param: number;
}

export interface ProcessingOptions {
  smoothing: SmoothingOptions;
  targetPoints: number;
  preserveEndpoints: boolean;
}

/**
 * Catmull-Rom spline interpolation for smooth stroke rendering
 */
export function catmullRomSpline(p0: Point, p1: Point, p2: Point, p3: Point, t: number): Point {
  const t2 = t * t;
  const t3 = t2 * t;
  
  // Catmull-Rom matrix coefficients
  const x = 0.5 * (
    (2 * p1.x) +
    (-p0.x + p2.x) * t +
    (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
    (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3
  );
  
  const y = 0.5 * (
    (2 * p1.y) +
    (-p0.y + p2.y) * t +
    (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
    (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3
  );
  
  // Interpolate time as well
  const time = p1.time + (p2.time - p1.time) * t;
  
  return { x, y, time };
}

/**
 * Create smooth interpolated stroke using Catmull-Rom splines
 */
export function createSmoothStroke(points: Point[], targetPoints: number): Point[] {
  if (points.length < 4) return points;
  
  const smoothPoints: Point[] = [];
  const segments = points.length - 1;
  const pointsPerSegment = Math.max(1, Math.floor(targetPoints / segments));
  
  for (let i = 0; i < segments; i++) {
    // Get control points for Catmull-Rom
    const p0 = i === 0 ? points[0] : points[i - 1];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = i === segments - 1 ? points[i + 1] : points[i + 2];
    
    // Generate interpolated points along this segment
    for (let j = 0; j < pointsPerSegment; j++) {
      const t = j / pointsPerSegment;
      const interpolated = catmullRomSpline(p0, p1, p2, p3, t);
      smoothPoints.push(interpolated);
    }
  }
  
  // Always include the last point
  smoothPoints.push(points[points.length - 1]);
  
  return smoothPoints;
}

/**
 * Gaussian smoothing filter
 */
export function gaussianSmooth(points: Point[], sigma: number): Point[] {
  if (sigma <= 0 || points.length < 3) return points;
  
  const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
  const kernel: number[] = [];
  let kernelSum = 0;
  
  // Generate Gaussian kernel
  for (let i = 0; i < kernelSize; i++) {
    const x = i - Math.floor(kernelSize / 2);
    const value = Math.exp(-(x * x) / (2 * sigma * sigma));
    kernel.push(value);
    kernelSum += value;
  }
  
  // Normalize kernel
  for (let i = 0; i < kernelSize; i++) {
    kernel[i] /= kernelSum;
  }
  
  const smoothed: Point[] = [];
  const halfKernel = Math.floor(kernelSize / 2);
  
  for (let i = 0; i < points.length; i++) {
    let weightedX = 0;
    let weightedY = 0;
    let weightSum = 0;
    
    for (let j = 0; j < kernelSize; j++) {
      const index = i + j - halfKernel;
      if (index >= 0 && index < points.length) {
        const weight = kernel[j];
        weightedX += points[index].x * weight;
        weightedY += points[index].y * weight;
        weightSum += weight;
      }
    }
    
    if (weightSum > 0) {
      smoothed.push({
        x: weightedX / weightSum,
        y: weightedY / weightSum,
        time: points[i].time,
        pressure: points[i].pressure
      });
    } else {
      smoothed.push(points[i]);
    }
  }
  
  return smoothed;
}

/**
 * Moving average smoothing
 */
export function movingAverageSmooth(points: Point[], windowSize: number): Point[] {
  if (windowSize <= 1 || points.length < windowSize) return points;
  
  const smoothed: Point[] = [];
  const halfWindow = Math.floor(windowSize / 2);
  
  for (let i = 0; i < points.length; i++) {
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    
    const start = Math.max(0, i - halfWindow);
    const end = Math.min(points.length - 1, i + halfWindow);
    
    for (let j = start; j <= end; j++) {
      sumX += points[j].x;
      sumY += points[j].y;
      count++;
    }
    
    smoothed.push({
      x: sumX / count,
      y: sumY / count,
      time: points[i].time,
      pressure: points[i].pressure
    });
  }
  
  return smoothed;
}

/**
 * Calculate arc length and normalize to 0-1
 */
export function calculateArcLength(points: Point[]): { lengths: number[]; totalLength: number } {
  const lengths: number[] = [0];
  let totalLength = 0;
  
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    const segmentLength = Math.sqrt(dx * dx + dy * dy);
    totalLength += segmentLength;
    lengths.push(totalLength);
  }
  
  return { lengths, totalLength };
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
    const dt = Math.max(points[i].time - points[i - 1].time, 1); // Avoid division by zero
    
    const velocity = Math.sqrt(dx * dx + dy * dy) / dt;
    velocities.push(velocity);
  }
  
  return velocities;
}

/**
 * Calculate acceleration profile
 */
export function calculateAcceleration(velocities: number[], points: Point[]): number[] {
  if (velocities.length < 2) return [];
  
  const accelerations: number[] = [];
  
  for (let i = 1; i < velocities.length; i++) {
    const dv = velocities[i] - velocities[i - 1];
    const dt = Math.max(points[i + 1].time - points[i].time, 1);
    
    const acceleration = dv / dt;
    accelerations.push(acceleration);
  }
  
  return accelerations;
}

/**
 * Calculate curvature at each point
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
    
    // Calculate curvature using cross product
    const crossProduct = v1.x * v2.y - v1.y * v2.x;
    const v1Mag = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const v2Mag = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    
    const curvature = v1Mag > 0 && v2Mag > 0 ? crossProduct / (v1Mag * v2Mag) : 0;
    curvatures.push(curvature);
  }
  
  return curvatures;
}

/**
 * Calculate tangent angles
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
 * Resample stroke to target number of points with arc-length parameterization
 */
export function resampleStroke(points: Point[], targetPoints: number): Point[] {
  if (points.length <= targetPoints) return points;
  
  const { lengths, totalLength } = calculateArcLength(points);
  const resampled: Point[] = [];
  
  // Always include first point
  resampled.push(points[0]);
  
  for (let i = 1; i < targetPoints - 1; i++) {
    const targetLength = (i / (targetPoints - 1)) * totalLength;
    
    // Find the segment containing this length
    let segmentIndex = 0;
    for (let j = 1; j < lengths.length; j++) {
      if (lengths[j] >= targetLength) {
        segmentIndex = j - 1;
        break;
      }
    }
    
    if (segmentIndex < points.length - 1) {
      // Interpolate within the segment
      const segmentStart = lengths[segmentIndex];
      const segmentEnd = lengths[segmentIndex + 1];
      const segmentLength = segmentEnd - segmentStart;
      
      if (segmentLength > 0) {
        const t = (targetLength - segmentStart) / segmentLength;
        const p1 = points[segmentIndex];
        const p2 = points[segmentIndex + 1];
        
        resampled.push({
          x: p1.x + t * (p2.x - p1.x),
          y: p1.y + t * (p2.y - p1.y),
          time: p1.time + t * (p2.time - p1.time),
          pressure: p1.pressure ? p1.pressure + t * ((p2.pressure || 1) - p1.pressure) : undefined
        });
      } else {
        resampled.push(points[segmentIndex]);
      }
    }
  }
  
  // Always include last point
  resampled.push(points[points.length - 1]);
  
  return resampled;
}

/**
 * Main processing function to create ProcessedPoint array
 */
export function processStroke(rawPoints: Point[], options: ProcessingOptions): ProcessedPoint[] {
  if (rawPoints.length < 2) return [];
  
  let points = [...rawPoints];
  
  // Apply smoothing
  switch (options.smoothing.type) {
    case 'gaussian':
      points = gaussianSmooth(points, options.smoothing.param);
      break;
    case 'movingAverage':
      points = movingAverageSmooth(points, Math.round(options.smoothing.param));
      break;
    case 'catmullRom':
      points = createSmoothStroke(points, Math.max(points.length, options.targetPoints));
      break;
  }
  
  // Resample to target number of points
  if (points.length !== options.targetPoints) {
    points = resampleStroke(points, options.targetPoints);
  }
  
  // Calculate derived metrics
  const { lengths, totalLength } = calculateArcLength(points);
  const velocities = calculateVelocity(points);
  const accelerations = calculateAcceleration(velocities, points);
  const curvatures = calculateCurvature(points);
  const tangentAngles = calculateTangentAngles(points);
  
  // Create processed points
  const processed: ProcessedPoint[] = points.map((point, index) => {
    const s_L = totalLength > 0 ? lengths[index] / totalLength : 0;
    const kappa_L = totalLength > 0 && curvatures[index - 1] !== undefined ? 
      curvatures[index - 1] * totalLength : 0;
    
    return {
      ...point,
      s_L,
      kappa_L,
      velocity: velocities[index - 1] || 0,
      acceleration: accelerations[index - 2] || 0,
      tangentAngle: tangentAngles[index - 1] || 0,
    };
  });
  
  return processed;
}

/**
 * Normalize stroke to unit square [0,1] x [0,1]
 */
export function normalizeStroke(points: Point[]): Point[] {
  if (points.length === 0) return [];
  
  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  
  const width = maxX - minX || 1;
  const height = maxY - minY || 1;
  
  return points.map(p => ({
    ...p,
    x: (p.x - minX) / width,
    y: (p.y - minY) / height
  }));
}
