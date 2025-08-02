import type { Point } from '@/store/researchStore';

/**
 * Knot Theory and Advanced Geometric Analysis Module
 * Implements writhe calculation, winding numbers, and advanced geometric invariants
 */

export interface KnotMetrics {
  writhe: number;
  windingNumber: number;
  totalTurning: number;
  linkingNumber: number;
  selfIntersections: IntersectionPoint[];
  crossingPattern: CrossingInfo[];
  geometricComplexity: number;
  chirality: number; // Handedness measure
}

export interface IntersectionPoint {
  index1: number;
  index2: number;
  point: { x: number; y: number };
  crossingType: 'positive' | 'negative';
  magnitude: number;
}

export interface CrossingInfo {
  position: { x: number; y: number };
  sign: number; // +1 for right-handed, -1 for left-handed
  angle: number; // Crossing angle
  indices: [number, number, number, number]; // Four points involved
}

/**
 * Calculate the writhe (self-linking) of a closed curve
 */
export function calculateWrithe(points: Point[]): number {
  if (points.length < 4) return 0;
  
  let writhe = 0;
  const n = points.length;
  
  // For each pair of non-adjacent segments
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 2; j < n - 1; j++) {
      // Skip adjacent segments
      if (Math.abs(i - j) <= 1 || (i === 0 && j === n - 2)) continue;
      
      const p1 = points[i];
      const p2 = points[i + 1];
      const q1 = points[j];
      const q2 = points[j + 1];
      
      // Calculate the crossing contribution
      const crossing = calculateCrossingContribution(p1, p2, q1, q2);
      writhe += crossing;
    }
  }
  
  return writhe / (4 * Math.PI);
}

/**
 * Calculate crossing contribution between two line segments
 */
function calculateCrossingContribution(
  p1: Point, p2: Point, q1: Point, q2: Point
): number {
  // Vector from p1 to p2
  const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
  // Vector from q1 to q2
  const v2 = { x: q2.x - q1.x, y: q2.y - q1.y };
  
  // Vectors from p1 to q1 and q2
  const r1 = { x: q1.x - p1.x, y: q1.y - p1.y };
  const r2 = { x: q2.x - p1.x, y: q2.y - p1.y };
  
  // Cross products
  const cross1 = v1.x * r1.y - v1.y * r1.x;
  const cross2 = v1.x * r2.y - v1.y * r2.x;
  
  // Check if the segments actually cross
  if (cross1 * cross2 >= 0) return 0; // No crossing
  
  // Calculate the crossing angle and sign
  const dotProduct = v1.x * v2.x + v1.y * v2.y;
  const crossProduct = v1.x * v2.y - v1.y * v2.x;
  
  const angle = Math.atan2(crossProduct, dotProduct);
  
  // Return signed contribution
  return crossProduct > 0 ? angle : -angle;
}

/**
 * Find all self-intersections in the stroke
 */
export function findSelfIntersections(points: Point[]): IntersectionPoint[] {
  const intersections: IntersectionPoint[] = [];
  const n = points.length;
  
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 2; j < n - 1; j++) {
      // Skip adjacent segments and wrap-around
      if (Math.abs(i - j) <= 1 || (i === 0 && j === n - 2)) continue;
      
      const p1 = points[i];
      const p2 = points[i + 1];
      const q1 = points[j];
      const q2 = points[j + 1];
      
      const intersection = findLineIntersection(p1, p2, q1, q2);
      if (intersection) {
        const crossingType = determineCrossingType(p1, p2, q1, q2);
        intersections.push({
          index1: i,
          index2: j,
          point: intersection,
          crossingType,
          magnitude: 1.0
        });
      }
    }
  }
  
  return intersections;
}

/**
 * Find intersection point between two line segments
 */
function findLineIntersection(
  p1: Point, p2: Point, q1: Point, q2: Point
): { x: number; y: number } | null {
  const d1 = { x: p2.x - p1.x, y: p2.y - p1.y };
  const d2 = { x: q2.x - q1.x, y: q2.y - q1.y };
  
  const denominator = d1.x * d2.y - d1.y * d2.x;
  if (Math.abs(denominator) < 1e-10) return null; // Parallel lines
  
  const t = ((q1.x - p1.x) * d2.y - (q1.y - p1.y) * d2.x) / denominator;
  const s = ((q1.x - p1.x) * d1.y - (q1.y - p1.y) * d1.x) / denominator;
  
  // Check if intersection is within both segments
  if (t >= 0 && t <= 1 && s >= 0 && s <= 1) {
    return {
      x: p1.x + t * d1.x,
      y: p1.y + t * d1.y
    };
  }
  
  return null;
}

/**
 * Determine if a crossing is positive or negative
 */
function determineCrossingType(
  p1: Point, p2: Point, q1: Point, q2: Point
): 'positive' | 'negative' {
  const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
  const v2 = { x: q2.x - q1.x, y: q2.y - q1.y };
  
  const crossProduct = v1.x * v2.y - v1.y * v2.x;
  return crossProduct > 0 ? 'positive' : 'negative';
}

/**
 * Calculate winding number of the curve
 */
export function calculateWindingNumber(points: Point[]): number {
  if (points.length < 3) return 0;
  
  let totalAngle = 0;
  
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    
    const angle1 = Math.atan2(prev.y, prev.x);
    const angle2 = Math.atan2(curr.y, curr.x);
    
    let deltaAngle = angle2 - angle1;
    
    // Normalize to [-π, π]
    while (deltaAngle > Math.PI) deltaAngle -= 2 * Math.PI;
    while (deltaAngle < -Math.PI) deltaAngle += 2 * Math.PI;
    
    totalAngle += deltaAngle;
  }
  
  return totalAngle / (2 * Math.PI);
}

/**
 * Calculate total turning (integral of curvature)
 */
export function calculateTotalTurning(points: Point[]): number {
  if (points.length < 3) return 0;
  
  let totalTurning = 0;
  
  for (let i = 1; i < points.length - 1; i++) {
    const p1 = points[i - 1];
    const p2 = points[i];
    const p3 = points[i + 1];
    
    // Calculate turning angle at point p2
    const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
    const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
    
    const angle1 = Math.atan2(v1.y, v1.x);
    const angle2 = Math.atan2(v2.y, v2.x);
    
    let turningAngle = angle2 - angle1;
    
    // Normalize to [-π, π]
    while (turningAngle > Math.PI) turningAngle -= 2 * Math.PI;
    while (turningAngle < -Math.PI) turningAngle += 2 * Math.PI;
    
    totalTurning += Math.abs(turningAngle);
  }
  
  return totalTurning;
}

/**
 * Analyze crossing patterns in the stroke
 */
export function analyzeCrossingPattern(points: Point[]): CrossingInfo[] {
  const intersections = findSelfIntersections(points);
  const crossings: CrossingInfo[] = [];
  
  for (const intersection of intersections) {
    const i = intersection.index1;
    const j = intersection.index2;
    
    const p1 = points[i];
    const p2 = points[i + 1];
    const q1 = points[j];
    const q2 = points[j + 1];
    
    // Calculate crossing angle
    const v1 = { x: p2.x - p1.x, y: p2.y - p1.y };
    const v2 = { x: q2.x - q1.x, y: q2.y - q1.y };
    
    const dotProduct = v1.x * v2.x + v1.y * v2.y;
    const v1Mag = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
    const v2Mag = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
    
    const cosAngle = dotProduct / (v1Mag * v2Mag);
    const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
    
    const sign = intersection.crossingType === 'positive' ? 1 : -1;
    
    crossings.push({
      position: intersection.point,
      sign,
      angle,
      indices: [i, i + 1, j, j + 1]
    });
  }
  
  return crossings;
}

/**
 * Calculate geometric complexity based on curvature distribution
 */
export function calculateGeometricComplexity(points: Point[]): number {
  if (points.length < 3) return 0;
  
  const curvatures: number[] = [];
  
  for (let i = 1; i < points.length - 1; i++) {
    const p1 = points[i - 1];
    const p2 = points[i];
    const p3 = points[i + 1];
    
    // Calculate curvature using the circumcircle formula
    const a = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
    const b = Math.sqrt((p2.x - p3.x) ** 2 + (p2.y - p3.y) ** 2);
    const c = Math.sqrt((p3.x - p1.x) ** 2 + (p3.y - p1.y) ** 2);
    
    // Area using cross product
    const area = Math.abs((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)) / 2;
    
    const curvature = area > 0 ? (4 * area) / (a * b * c) : 0;
    curvatures.push(curvature);
  }
  
  if (curvatures.length === 0) return 0;
  
  // Calculate complexity as entropy of curvature distribution
  const maxCurvature = Math.max(...curvatures);
  if (maxCurvature === 0) return 0;
  
  // Normalize curvatures and calculate entropy
  const bins = 10;
  const histogram = new Array(bins).fill(0);
  
  for (const curvature of curvatures) {
    const binIndex = Math.min(bins - 1, Math.floor((curvature / maxCurvature) * bins));
    histogram[binIndex]++;
  }
  
  // Normalize histogram
  const total = curvatures.length;
  let entropy = 0;
  
  for (const count of histogram) {
    if (count > 0) {
      const probability = count / total;
      entropy -= probability * Math.log2(probability);
    }
  }
  
  return entropy;
}

/**
 * Calculate chirality (handedness) of the stroke
 */
export function calculateChirality(points: Point[]): number {
  if (points.length < 4) return 0;
  
  let chiralitySum = 0;
  let count = 0;
  
  for (let i = 1; i < points.length - 2; i++) {
    const p1 = points[i - 1];
    const p2 = points[i];
    const p3 = points[i + 1];
    const p4 = points[i + 2];
    
    // Calculate the dihedral angle between consecutive triangles
    const n1 = crossProduct(subtract(p2, p1), subtract(p3, p1));
    const n2 = crossProduct(subtract(p3, p2), subtract(p4, p2));
    
    const dotProd = n1.x * n2.x + n1.y * n2.y;
    const n1Mag = Math.sqrt(n1.x * n1.x + n1.y * n1.y);
    const n2Mag = Math.sqrt(n2.x * n2.x + n2.y * n2.y);
    
    if (n1Mag > 0 && n2Mag > 0) {
      const cosTheta = dotProd / (n1Mag * n2Mag);
      const theta = Math.acos(Math.max(-1, Math.min(1, cosTheta)));
      
      // Determine sign based on orientation
      const cross = n1.x * n2.y - n1.y * n2.x;
      const signedTheta = cross > 0 ? theta : -theta;
      
      chiralitySum += signedTheta;
      count++;
    }
  }
  
  return count > 0 ? chiralitySum / count : 0;
}

/**
 * Helper functions for vector operations
 */
function subtract(p1: Point, p2: Point): { x: number; y: number } {
  return { x: p1.x - p2.x, y: p1.y - p2.y };
}

function crossProduct(v1: { x: number; y: number }, v2: { x: number; y: number }): { x: number; y: number } {
  // For 2D vectors, we extend to 3D and take the cross product
  return { 
    x: 0, // v1.y * 0 - 0 * v2.y
    y: 0  // 0 * v2.x - v1.x * 0
  };
}

/**
 * Main function to compute all knot theory metrics
 */
export function computeKnotMetrics(points: Point[]): KnotMetrics {
  if (points.length < 3) {
    return {
      writhe: 0,
      windingNumber: 0,
      totalTurning: 0,
      linkingNumber: 0,
      selfIntersections: [],
      crossingPattern: [],
      geometricComplexity: 0,
      chirality: 0
    };
  }
  
  const writhe = calculateWrithe(points);
  const windingNumber = calculateWindingNumber(points);
  const totalTurning = calculateTotalTurning(points);
  const selfIntersections = findSelfIntersections(points);
  const crossingPattern = analyzeCrossingPattern(points);
  const geometricComplexity = calculateGeometricComplexity(points);
  const chirality = calculateChirality(points);
  
  // Linking number (for closed curves)
  const linkingNumber = (writhe + windingNumber) / 2;
  
  return {
    writhe,
    windingNumber,
    totalTurning,
    linkingNumber,
    selfIntersections,
    crossingPattern,
    geometricComplexity,
    chirality
  };
}

/**
 * Compare knot metrics for similarity
 */
export function compareKnotMetrics(metrics1: KnotMetrics, metrics2: KnotMetrics): number {
  // Weighted similarity calculation
  const writheSimil = 1 - Math.abs(metrics1.writhe - metrics2.writhe) / 10;
  const windingSimil = 1 - Math.abs(metrics1.windingNumber - metrics2.windingNumber) / 5;
  const turningSimil = 1 - Math.abs(metrics1.totalTurning - metrics2.totalTurning) / (2 * Math.PI);
  const complexitySimil = 1 - Math.abs(metrics1.geometricComplexity - metrics2.geometricComplexity) / 10;
  const chiralitySimil = 1 - Math.abs(metrics1.chirality - metrics2.chirality) / Math.PI;
  
  // Intersection count similarity
  const intersectionSimil = 1 - Math.abs(
    metrics1.selfIntersections.length - metrics2.selfIntersections.length
  ) / 10;
  
  // Weighted average
  return Math.max(0, Math.min(1, 
    0.2 * writheSimil +
    0.15 * windingSimil +
    0.2 * turningSimil +
    0.15 * complexitySimil +
    0.15 * chiralitySimil +
    0.15 * intersectionSimil
  ));
}

/**
 * Extract features for machine learning
 */
export function extractKnotFeatures(metrics: KnotMetrics): number[] {
  const features: number[] = [];
  
  // Basic topological invariants
  features.push(metrics.writhe);
  features.push(metrics.windingNumber);
  features.push(metrics.totalTurning);
  features.push(metrics.linkingNumber);
  
  // Geometric measures
  features.push(metrics.geometricComplexity);
  features.push(metrics.chirality);
  
  // Intersection statistics
  features.push(metrics.selfIntersections.length);
  features.push(metrics.crossingPattern.length);
  
  // Crossing pattern statistics
  if (metrics.crossingPattern.length > 0) {
    const positiveCount = metrics.crossingPattern.filter(c => c.sign > 0).length;
    const negativeCount = metrics.crossingPattern.filter(c => c.sign < 0).length;
    const avgAngle = metrics.crossingPattern.reduce((sum, c) => sum + c.angle, 0) / metrics.crossingPattern.length;
    
    features.push(positiveCount);
    features.push(negativeCount);
    features.push(avgAngle);
    features.push(positiveCount - negativeCount); // Crossing balance
  } else {
    features.push(0, 0, 0, 0);
  }
  
  return features;
}
