import type { Point } from '@/store/researchStore';

/**
 * Path Signature Calculation Module
 * Implements iterated integrals and path signatures for sequential data analysis
 * Based on rough path theory and provides reparametrization-invariant features
 */

export interface PathSignatureData {
  level1: number[];          // First-order terms (displacement)
  level2: number[];          // Second-order terms (areas)
  level3: number[];          // Third-order terms (higher moments)
  level4: number[];          // Fourth-order terms
  logSignature: number[];    // Log signature for compact representation
  totalVariation: number;    // p-variation of the path
  signatureNorm: number;     // L2 norm of signature
}

/**
 * Calculate first-order signature terms (total displacement)
 */
export function calculateLevel1Signature(path: Point[]): number[] {
  if (path.length < 2) return [0, 0];
  
  const start = path[0];
  const end = path[path.length - 1];
  
  return [
    end.x - start.x,  // Total x displacement
    end.y - start.y   // Total y displacement
  ];
}

/**
 * Calculate second-order signature terms (signed areas)
 */
export function calculateLevel2Signature(path: Point[]): number[] {
  if (path.length < 2) return [0, 0, 0, 0];
  
  let S_xx = 0; // ∫x dX (moment of x with respect to x-increments)
  let S_xy = 0; // ∫x dY (signed area under x with respect to y)
  let S_yx = 0; // ∫y dX (signed area under y with respect to x)
  let S_yy = 0; // ∫y dY (moment of y with respect to y-increments)
  
  for (let i = 1; i < path.length; i++) {
    const prev = path[i - 1];
    const curr = path[i];
    
    const dx = curr.x - prev.x;
    const dy = curr.y - prev.y;
    
    // Trapezoidal rule for integration
    S_xx += (prev.x + curr.x) * 0.5 * dx;
    S_xy += (prev.x + curr.x) * 0.5 * dy;
    S_yx += (prev.y + curr.y) * 0.5 * dx;
    S_yy += (prev.y + curr.y) * 0.5 * dy;
  }
  
  return [S_xx, S_xy, S_yx, S_yy];
}

/**
 * Calculate third-order signature terms (triple integrals)
 */
export function calculateLevel3Signature(path: Point[]): number[] {
  if (path.length < 3) return new Array(8).fill(0);
  
  // Third-order terms: S_xyz where each index can be x or y
  let S_xxx = 0, S_xxy = 0, S_xyx = 0, S_xyy = 0;
  let S_yxx = 0, S_yxy = 0, S_yyx = 0, S_yyy = 0;
  
  for (let i = 2; i < path.length; i++) {
    const p0 = path[i - 2];
    const p1 = path[i - 1];
    const p2 = path[i];
    
    const dx1 = p1.x - p0.x;
    const dy1 = p1.y - p0.y;
    const dx2 = p2.x - p1.x;
    const dy2 = p2.y - p1.y;
    
    // Simplified third-order calculation (using midpoint rule)
    const x_mid = (p0.x + p1.x + p2.x) / 3;
    const y_mid = (p0.y + p1.y + p2.y) / 3;
    
    S_xxx += x_mid * dx1 * dx2;
    S_xxy += x_mid * dx1 * dy2;
    S_xyx += x_mid * dy1 * dx2;
    S_xyy += x_mid * dy1 * dy2;
    S_yxx += y_mid * dx1 * dx2;
    S_yxy += y_mid * dx1 * dy2;
    S_yyx += y_mid * dy1 * dx2;
    S_yyy += y_mid * dy1 * dy2;
  }
  
  return [S_xxx, S_xxy, S_xyx, S_xyy, S_yxx, S_yxy, S_yyx, S_yyy];
}

/**
 * Calculate fourth-order signature terms (simplified)
 */
export function calculateLevel4Signature(path: Point[]): number[] {
  if (path.length < 4) return new Array(16).fill(0);
  
  // For computational efficiency, we'll compute a subset of the most important terms
  const terms: number[] = new Array(16).fill(0);
  
  for (let i = 3; i < path.length; i++) {
    const p0 = path[i - 3];
    const p1 = path[i - 2];
    const p2 = path[i - 1];
    const p3 = path[i];
    
    const dx1 = p1.x - p0.x;
    const dy1 = p1.y - p0.y;
    const dx2 = p2.x - p1.x;
    const dy2 = p2.y - p1.y;
    const dx3 = p3.x - p2.x;
    const dy3 = p3.y - p2.y;
    
    // Calculate a few representative fourth-order terms
    const x_avg = (p0.x + p1.x + p2.x + p3.x) / 4;
    const y_avg = (p0.y + p1.y + p2.y + p3.y) / 4;
    
    terms[0] += x_avg * dx1 * dx2 * dx3;  // S_xxxx
    terms[1] += x_avg * dx1 * dx2 * dy3;  // S_xxxy
    terms[2] += x_avg * dx1 * dy2 * dx3;  // S_xxyx
    terms[3] += x_avg * dy1 * dx2 * dx3;  // S_xyxx
    terms[4] += y_avg * dx1 * dx2 * dx3;  // S_yxxx
    terms[5] += x_avg * dy1 * dy2 * dy3;  // S_xyyy
    terms[6] += y_avg * dx1 * dy2 * dy3;  // S_yxyy
    terms[7] += y_avg * dy1 * dx2 * dy3;  // S_yyxy
    terms[8] += y_avg * dy1 * dy2 * dx3;  // S_yyyx
    terms[9] += y_avg * dy1 * dy2 * dy3;  // S_yyyy
    
    // Additional cross terms
    terms[10] += x_avg * x_avg * dx1 * dy1;
    terms[11] += y_avg * y_avg * dx1 * dy1;
    terms[12] += x_avg * y_avg * dx1 * dx2;
    terms[13] += x_avg * y_avg * dy1 * dy2;
    terms[14] += Math.abs(dx1 * dy2 - dy1 * dx2); // Local area element
    terms[15] += Math.sqrt(dx1*dx1 + dy1*dy1) * Math.sqrt(dx2*dx2 + dy2*dy2); // Speed correlation
  }
  
  return terms;
}

/**
 * Calculate log signature using the BCH (Baker-Campbell-Hausdorff) formula
 * This provides a more compact representation
 */
export function calculateLogSignature(
  level1: number[], 
  level2: number[], 
  level3: number[]
): number[] {
  // Simplified log signature calculation
  // In practice, this would use the full BCH formula
  
  const logSig: number[] = [];
  
  // First-order terms remain the same
  logSig.push(...level1);
  
  // Second-order terms with corrections
  for (let i = 0; i < level2.length; i++) {
    let correction = 0;
    
    // Apply BCH corrections (simplified)
    if (i === 1) { // S_xy term
      correction = -0.5 * level1[0] * level1[1]; // -0.5 * [X,Y]
    } else if (i === 2) { // S_yx term
      correction = 0.5 * level1[0] * level1[1]; // 0.5 * [X,Y]
    }
    
    logSig.push(level2[i] + correction);
  }
  
  // Third-order terms with higher-order corrections
  for (let i = 0; i < Math.min(level3.length, 8); i++) {
    let correction = 0;
    
    // Apply third-order BCH corrections (very simplified)
    if (i < 4) {
      correction = (1/12) * level1[0] * level2[i % 2];
    } else {
      correction = (1/12) * level1[1] * level2[(i - 4) % 2];
    }
    
    logSig.push(level3[i] + correction);
  }
  
  return logSig;
}

/**
 * Calculate p-variation of the path (roughness measure)
 */
export function calculateTotalVariation(path: Point[], p: number = 2): number {
  if (path.length < 2) return 0;
  
  let variation = 0;
  
  for (let i = 1; i < path.length; i++) {
    const dx = path[i].x - path[i - 1].x;
    const dy = path[i].y - path[i - 1].y;
    const increment = Math.sqrt(dx * dx + dy * dy);
    
    variation += Math.pow(increment, p);
  }
  
  return Math.pow(variation, 1 / p);
}

/**
 * Calculate the L2 norm of the signature
 */
export function calculateSignatureNorm(signature: number[]): number {
  return Math.sqrt(signature.reduce((sum, x) => sum + x * x, 0));
}

/**
 * Normalize path to remove translation and scale
 */
export function normalizePath(path: Point[]): Point[] {
  if (path.length === 0) return [];
  
  // Remove translation (center at origin)
  const meanX = path.reduce((sum, p) => sum + p.x, 0) / path.length;
  const meanY = path.reduce((sum, p) => sum + p.y, 0) / path.length;
  
  const centered = path.map(p => ({
    ...p,
    x: p.x - meanX,
    y: p.y - meanY
  }));
  
  // Remove scale (normalize to unit total variation)
  const totalVar = calculateTotalVariation(centered, 1);
  if (totalVar === 0) return centered;
  
  return centered.map(p => ({
    ...p,
    x: p.x / totalVar,
    y: p.y / totalVar
  }));
}

/**
 * Main function to compute complete path signature
 */
export function computePathSignature(path: Point[]): PathSignatureData {
  if (path.length < 2) {
    return {
      level1: [0, 0],
      level2: [0, 0, 0, 0],
      level3: new Array(8).fill(0),
      level4: new Array(16).fill(0),
      logSignature: new Array(14).fill(0),
      totalVariation: 0,
      signatureNorm: 0
    };
  }
  
  // Normalize the path first
  const normalizedPath = normalizePath(path);
  
  // Calculate signature levels
  const level1 = calculateLevel1Signature(normalizedPath);
  const level2 = calculateLevel2Signature(normalizedPath);
  const level3 = calculateLevel3Signature(normalizedPath);
  const level4 = calculateLevel4Signature(normalizedPath);
  
  // Calculate log signature
  const logSignature = calculateLogSignature(level1, level2, level3);
  
  // Calculate additional metrics
  const totalVariation = calculateTotalVariation(normalizedPath);
  const fullSignature = [...level1, ...level2, ...level3];
  const signatureNorm = calculateSignatureNorm(fullSignature);
  
  return {
    level1,
    level2,
    level3,
    level4,
    logSignature,
    totalVariation,
    signatureNorm
  };
}

/**
 * Compare two path signatures for similarity
 */
export function comparePathSignatures(sig1: PathSignatureData, sig2: PathSignatureData): number {
  // Use log signatures for comparison as they're more compact
  const logSig1 = sig1.logSignature;
  const logSig2 = sig2.logSignature;
  
  if (logSig1.length === 0 || logSig2.length === 0) return 0;
  
  // Calculate normalized dot product
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  const minLength = Math.min(logSig1.length, logSig2.length);
  
  for (let i = 0; i < minLength; i++) {
    dotProduct += logSig1[i] * logSig2[i];
    norm1 += logSig1[i] * logSig1[i];
    norm2 += logSig2[i] * logSig2[i];
  }
  
  const normalizer = Math.sqrt(norm1 * norm2);
  return normalizer > 0 ? Math.abs(dotProduct / normalizer) : 0;
}

/**
 * Extract key features from path signature for machine learning
 */
export function extractSignatureFeatures(signature: PathSignatureData): number[] {
  const features: number[] = [];
  
  // Level 1 features (displacement)
  features.push(Math.sqrt(signature.level1[0]**2 + signature.level1[1]**2)); // Total displacement magnitude
  features.push(Math.atan2(signature.level1[1], signature.level1[0])); // Displacement angle
  
  // Level 2 features (areas)
  features.push(signature.level2[1] - signature.level2[2]); // Net signed area (xy - yx)
  features.push(signature.level2[0] + signature.level2[3]); // Trace (xx + yy)
  
  // Level 3 features (selected)
  features.push(signature.level3[0]); // xxx term
  features.push(signature.level3[7]); // yyy term
  features.push(signature.level3[1] + signature.level3[6]); // Cross terms
  
  // Global features
  features.push(signature.totalVariation);
  features.push(signature.signatureNorm);
  features.push(Math.log(1 + signature.signatureNorm)); // Log-scaled norm
  
  // Ratios and normalized features
  if (signature.totalVariation > 0) {
    features.push(signature.signatureNorm / signature.totalVariation);
  } else {
    features.push(0);
  }
  
  return features;
}
