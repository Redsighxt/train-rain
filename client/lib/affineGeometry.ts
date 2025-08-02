import type { Point, ProcessedPoint } from '@/store/researchStore';

/**
 * Affine Differential Geometry Module
 * Implements affine-invariant curve analysis for handwriting recognition
 * Based on affine arc length and affine curvature computations
 */

export interface AffineMetrics {
  affineArcLength: number[];
  affineCurvature: number[];
  affineSignature: Array<{ sigma: number; kappa_a: number }>;
  totalAffineLength: number;
}

/**
 * Calculate affine arc length element
 * Formula: dσ = |x'y'' - x''y'|^(1/3) dt
 */
export function calculateAffineArcLengthElement(
  x1: number, y1: number,
  x2: number, y2: number, 
  x3: number, y3: number,
  dt: number
): number {
  // Calculate first derivatives (velocity)
  const xPrime = (x2 - x1) / dt;
  const yPrime = (y2 - y1) / dt;
  
  // Calculate second derivatives (acceleration)
  const xDoublePrime = (x3 - 2 * x2 + x1) / (dt * dt);
  const yDoublePrime = (y3 - 2 * y2 + y1) / (dt * dt);
  
  // Calculate the determinant |x'y'' - x''y'|
  const determinant = Math.abs(xPrime * yDoublePrime - xDoublePrime * yPrime);
  
  // Affine arc length element: determinant^(1/3) * dt
  return Math.pow(determinant, 1/3) * dt;
}

/**
 * Calculate affine arc length parameterization for entire stroke
 */
export function calculateAffineArcLength(points: Point[]): number[] {
  if (points.length < 3) return [];
  
  const affineArcLengths: number[] = [0, 0]; // First two points have zero affine arc length
  let cumulativeLength = 0;
  
  for (let i = 2; i < points.length; i++) {
    const p1 = points[i - 2];
    const p2 = points[i - 1];
    const p3 = points[i];
    
    // Use time differences for dt
    const dt1 = Math.max(p2.time - p1.time, 1);
    const dt2 = Math.max(p3.time - p2.time, 1);
    const avgDt = (dt1 + dt2) / 2;
    
    const dSigma = calculateAffineArcLengthElement(
      p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, avgDt
    );
    
    cumulativeLength += dSigma;
    affineArcLengths.push(cumulativeLength);
  }
  
  return affineArcLengths;
}

/**
 * Calculate affine curvature
 * Formula: κₐ = d²x/dσ² * dy/dσ - d²y/dσ² * dx/dσ
 * where σ is the affine arc length parameter
 */
export function calculateAffineCurvature(points: Point[], affineArcLengths: number[]): number[] {
  if (points.length < 5 || affineArcLengths.length < 5) return [];
  
  const affineCurvatures: number[] = [];
  
  for (let i = 2; i < points.length - 2; i++) {
    const sigma1 = affineArcLengths[i - 1];
    const sigma2 = affineArcLengths[i];
    const sigma3 = affineArcLengths[i + 1];
    
    if (sigma3 - sigma1 < 1e-10) {
      affineCurvatures.push(0);
      continue;
    }
    
    // Calculate first derivatives with respect to affine arc length
    const dSigma1 = sigma2 - sigma1;
    const dSigma2 = sigma3 - sigma2;
    
    if (dSigma1 < 1e-10 || dSigma2 < 1e-10) {
      affineCurvatures.push(0);
      continue;
    }
    
    const dxdSigma = (points[i + 1].x - points[i - 1].x) / (sigma3 - sigma1);
    const dydSigma = (points[i + 1].y - points[i - 1].y) / (sigma3 - sigma1);
    
    // Calculate second derivatives
    const d2xdSigma2 = (
      (points[i + 1].x - points[i].x) / dSigma2 - 
      (points[i].x - points[i - 1].x) / dSigma1
    ) / ((dSigma1 + dSigma2) / 2);
    
    const d2ydSigma2 = (
      (points[i + 1].y - points[i].y) / dSigma2 - 
      (points[i].y - points[i - 1].y) / dSigma1
    ) / ((dSigma1 + dSigma2) / 2);
    
    // Affine curvature formula
    const affineCurvature = d2xdSigma2 * dydSigma - d2ydSigma2 * dxdSigma;
    affineCurvatures.push(affineCurvature);
  }
  
  return affineCurvatures;
}

/**
 * Normalize affine arc length to [0, 1] range
 */
export function normalizeAffineArcLength(affineArcLengths: number[]): number[] {
  if (affineArcLengths.length === 0) return [];
  
  const maxLength = Math.max(...affineArcLengths);
  if (maxLength === 0) return affineArcLengths.map(() => 0);
  
  return affineArcLengths.map(length => length / maxLength);
}

/**
 * Create affine signature: (σ, κₐ) pairs
 */
export function createAffineSignature(
  normalizedAffineArcLengths: number[], 
  affineCurvatures: number[]
): Array<{ sigma: number; kappa_a: number }> {
  const signature: Array<{ sigma: number; kappa_a: number }> = [];
  
  // Align arrays - affine curvature has fewer points
  const offset = Math.floor((normalizedAffineArcLengths.length - affineCurvatures.length) / 2);
  
  for (let i = 0; i < affineCurvatures.length; i++) {
    const sigmaIndex = i + offset;
    if (sigmaIndex >= 0 && sigmaIndex < normalizedAffineArcLengths.length) {
      signature.push({
        sigma: normalizedAffineArcLengths[sigmaIndex],
        kappa_a: affineCurvatures[i]
      });
    }
  }
  
  return signature;
}

/**
 * Calculate affine invariant features for comparison
 */
export function calculateAffineInvariants(points: Point[]): {
  maxAffineCurvature: number;
  meanAbsAffineCurvature: number;
  affineCurvatureVariance: number;
  affineComplexity: number;
  affineSymmetry: number;
} {
  const affineArcLengths = calculateAffineArcLength(points);
  const affineCurvatures = calculateAffineCurvature(points, affineArcLengths);
  
  if (affineCurvatures.length === 0) {
    return {
      maxAffineCurvature: 0,
      meanAbsAffineCurvature: 0,
      affineCurvatureVariance: 0,
      affineComplexity: 0,
      affineSymmetry: 0
    };
  }
  
  const absAffineCurvatures = affineCurvatures.map(Math.abs);
  const maxAffineCurvature = Math.max(...absAffineCurvatures);
  const meanAbsAffineCurvature = absAffineCurvatures.reduce((sum, c) => sum + c, 0) / absAffineCurvatures.length;
  
  // Calculate variance
  const variance = absAffineCurvatures.reduce(
    (sum, c) => sum + Math.pow(c - meanAbsAffineCurvature, 2), 0
  ) / absAffineCurvatures.length;
  
  // Affine complexity - measure of curvature variation
  const affineComplexity = Math.sqrt(variance) / (meanAbsAffineCurvature + 1e-10);
  
  // Affine symmetry - measure of symmetry in affine space
  const midPoint = Math.floor(affineCurvatures.length / 2);
  let symmetrySum = 0;
  let symmetryCount = 0;
  
  for (let i = 0; i < midPoint; i++) {
    const leftIndex = i;
    const rightIndex = affineCurvatures.length - 1 - i;
    if (rightIndex > leftIndex) {
      const diff = Math.abs(affineCurvatures[leftIndex] - affineCurvatures[rightIndex]);
      symmetrySum += Math.exp(-diff * 10); // Exponential decay based on difference
      symmetryCount++;
    }
  }
  
  const affineSymmetry = symmetryCount > 0 ? symmetrySum / symmetryCount : 0;
  
  return {
    maxAffineCurvature,
    meanAbsAffineCurvature,
    affineCurvatureVariance: variance,
    affineComplexity,
    affineSymmetry
  };
}

/**
 * Main function to compute all affine metrics
 */
export function computeAffineMetrics(points: Point[]): AffineMetrics {
  if (points.length < 3) {
    return {
      affineArcLength: [],
      affineCurvature: [],
      affineSignature: [],
      totalAffineLength: 0
    };
  }
  
  const affineArcLengths = calculateAffineArcLength(points);
  const totalAffineLength = affineArcLengths.length > 0 ? 
    Math.max(...affineArcLengths) : 0;
  
  const normalizedAffineArcLengths = normalizeAffineArcLength(affineArcLengths);
  const affineCurvatures = calculateAffineCurvature(points, affineArcLengths);
  const affineSignature = createAffineSignature(normalizedAffineArcLengths, affineCurvatures);
  
  return {
    affineArcLength: normalizedAffineArcLengths,
    affineCurvature: affineCurvatures,
    affineSignature,
    totalAffineLength
  };
}

/**
 * Compare two affine signatures for similarity
 */
export function compareAffineSignatures(
  sig1: Array<{ sigma: number; kappa_a: number }>,
  sig2: Array<{ sigma: number; kappa_a: number }>
): number {
  if (sig1.length === 0 || sig2.length === 0) return 0;
  
  // Resample both signatures to same length for comparison
  const targetLength = Math.min(sig1.length, sig2.length, 50);
  
  const resample = (sig: Array<{ sigma: number; kappa_a: number }>) => {
    const resampled: Array<{ sigma: number; kappa_a: number }> = [];
    for (let i = 0; i < targetLength; i++) {
      const t = i / (targetLength - 1);
      const index = Math.floor(t * (sig.length - 1));
      const nextIndex = Math.min(index + 1, sig.length - 1);
      const localT = (t * (sig.length - 1)) - index;
      
      const sigma = sig[index].sigma + localT * (sig[nextIndex].sigma - sig[index].sigma);
      const kappa_a = sig[index].kappa_a + localT * (sig[nextIndex].kappa_a - sig[index].kappa_a);
      
      resampled.push({ sigma, kappa_a });
    }
    return resampled;
  };
  
  const resampled1 = resample(sig1);
  const resampled2 = resample(sig2);
  
  // Calculate similarity using normalized cross-correlation
  let similarity = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < targetLength; i++) {
    similarity += resampled1[i].kappa_a * resampled2[i].kappa_a;
    norm1 += resampled1[i].kappa_a * resampled1[i].kappa_a;
    norm2 += resampled2[i].kappa_a * resampled2[i].kappa_a;
  }
  
  const normalizer = Math.sqrt(norm1 * norm2);
  return normalizer > 0 ? Math.abs(similarity / normalizer) : 0;
}
