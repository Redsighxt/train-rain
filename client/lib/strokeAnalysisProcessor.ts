import type { Point, ProcessedPoint, InvariantMetrics, LandmarkPoint, Stroke } from '@/store/researchStore';
import { processStroke, normalizeStroke } from './strokeProcessing';
import { computeAffineMetrics } from './affineGeometry';
import { computePathSignature } from './pathSignature';
import { computeTopologicalMetrics } from './topologicalAnalysis';
import { computeKnotMetrics } from './knotTheory';
import { computeCurvatureMetrics, extractCurvatureLandmarks } from './curvatureAnalysis';
import { computeSpectralMetrics } from './spectralAnalysis';
import { create3DSignatureUMAP, extractComprehensiveFeatures } from './umapMapping';

/**
 * Comprehensive Stroke Analysis Processor
 * Integrates all advanced mathematical analyses with performance optimization
 */

export interface AnalysisConfig {
  enableAffineAnalysis: boolean;
  enableTopologicalAnalysis: boolean;
  enablePathSignature: boolean;
  enableSpectralAnalysis: boolean;
  enableKnotTheory: boolean;
  targetProcessingTime: number; // Target time in ms
  adaptiveQuality: boolean; // Reduce quality if needed for performance
  cacheResults: boolean;
  maxPoints: number;
}

export interface ProcessingResult {
  processed: ProcessedPoint[];
  landmarks: LandmarkPoint[];
  invariants: InvariantMetrics;
  processingTime: number;
  qualityScore: number;
  cacheHit: boolean;
}

export interface PerformanceMetrics {
  preprocessingTime: number;
  affineTime: number;
  topologicalTime: number;
  pathSignatureTime: number;
  spectralTime: number;
  knotTheoryTime: number;
  curvatureTime: number;
  umapTime: number;
  totalTime: number;
  pointsProcessed: number;
  algorithmsUsed: string[];
}

// Results cache for performance optimization
const analysisCache = new Map<string, ProcessingResult>();

// Performance monitoring
let performanceHistory: PerformanceMetrics[] = [];

/**
 * Default analysis configuration optimized for real-time performance
 */
export const defaultAnalysisConfig: AnalysisConfig = {
  enableAffineAnalysis: true,
  enableTopologicalAnalysis: true,
  enablePathSignature: true,
  enableSpectralAnalysis: true,
  enableKnotTheory: true,
  targetProcessingTime: 100, // 100ms target
  adaptiveQuality: true,
  cacheResults: true,
  maxPoints: 100
};

/**
 * Generate cache key for stroke analysis
 */
function generateCacheKey(stroke: Stroke, config: AnalysisConfig): string {
  const pointsHash = stroke.points
    .map(p => `${p.x.toFixed(2)},${p.y.toFixed(2)}`)
    .join('|');
  
  const configHash = Object.entries(config)
    .map(([key, value]) => `${key}:${value}`)
    .join('|');
  
  return `${pointsHash}::${configHash}`;
}

/**
 * Adaptive quality reduction based on performance constraints
 */
function adaptConfigForPerformance(
  config: AnalysisConfig, 
  pointCount: number,
  recentPerformance: PerformanceMetrics[]
): AnalysisConfig {
  if (!config.adaptiveQuality) return config;
  
  const avgTime = recentPerformance.length > 0 ?
    recentPerformance.reduce((sum, p) => sum + p.totalTime, 0) / recentPerformance.length : 0;
  
  const adaptedConfig = { ...config };
  
  // If recent performance is slow, reduce quality
  if (avgTime > config.targetProcessingTime * 1.5) {
    // Reduce point count for complex analyses
    adaptedConfig.maxPoints = Math.min(config.maxPoints, 50);
    
    // Disable expensive analyses if needed
    if (avgTime > config.targetProcessingTime * 2) {
      adaptedConfig.enableTopologicalAnalysis = false;
      adaptedConfig.enableKnotTheory = false;
    }
    
    if (avgTime > config.targetProcessingTime * 3) {
      adaptedConfig.enablePathSignature = false;
      adaptedConfig.enableSpectralAnalysis = false;
    }
  }
  
  // If stroke is very complex, reduce scope
  if (pointCount > 200) {
    adaptedConfig.enableTopologicalAnalysis = false;
    adaptedConfig.enableKnotTheory = false;
    adaptedConfig.maxPoints = Math.min(adaptedConfig.maxPoints, 75);
  }
  
  return adaptedConfig;
}

/**
 * High-performance timer for precise measurement
 */
function timer() {
  const start = performance.now();
  return () => performance.now() - start;
}

/**
 * Process stroke with full mathematical analysis
 */
export async function analyzeStroke(
  stroke: Stroke,
  config: AnalysisConfig = defaultAnalysisConfig
): Promise<ProcessingResult> {
  const globalTimer = timer();
  
  // Check cache first
  if (config.cacheResults) {
    const cacheKey = generateCacheKey(stroke, config);
    const cached = analysisCache.get(cacheKey);
    if (cached) {
      return { ...cached, cacheHit: true };
    }
  }
  
  // Adaptive configuration based on recent performance
  const recentPerformance = performanceHistory.slice(-5);
  const adaptedConfig = adaptConfigForPerformance(config, stroke.points.length, recentPerformance);
  
  const metrics: PerformanceMetrics = {
    preprocessingTime: 0,
    affineTime: 0,
    topologicalTime: 0,
    pathSignatureTime: 0,
    spectralTime: 0,
    knotTheoryTime: 0,
    curvatureTime: 0,
    umapTime: 0,
    totalTime: 0,
    pointsProcessed: stroke.points.length,
    algorithmsUsed: []
  };
  
  // 1. Preprocessing and normalization
  let preprocessTimer = timer();
  
  let normalizedPoints = normalizeStroke(stroke.points);
  
  // Reduce points if needed for performance
  if (normalizedPoints.length > adaptedConfig.maxPoints) {
    const step = Math.ceil(normalizedPoints.length / adaptedConfig.maxPoints);
    normalizedPoints = normalizedPoints.filter((_, i) => i % step === 0);
  }
  
  const processed = processStroke(normalizedPoints, {
    smoothing: { type: 'catmullRom', param: 0.5 },
    targetPoints: Math.min(normalizedPoints.length, adaptedConfig.maxPoints),
    preserveEndpoints: true
  });
  
  metrics.preprocessingTime = preprocessTimer();
  metrics.algorithmsUsed.push('preprocessing');
  
  // Initialize invariants object
  const invariants: InvariantMetrics = {
    arcLength: 0,
    totalTurning: 0,
    windingNumber: 0,
    writhe: 0,
    bettiNumbers: { b0: 1, b1: 0 },
    persistenceDiagram: [],
    complexityScore: 0,
    regularityScore: 0,
    symmetryScore: 0,
    stabilityIndex: 0,
    pathSignature: {
      level1: [0, 0],
      level2: [0, 0, 0, 0],
      level3: new Array(8).fill(0),
      logSignature: new Array(14).fill(0)
    },
    affineSignature: [],
    fourierCoefficients: { magnitude: [], phase: [] },
    waveletCoefficients: [],
    spectralCentroid: 0
  };
  
  let landmarks: LandmarkPoint[] = [];
  
  // 2. Curvature Analysis (always enabled - fundamental)
  let curvatureTimer = timer();
  const curvatureMetrics = computeCurvatureMetrics(normalizedPoints);
  landmarks.push(...extractCurvatureLandmarks(normalizedPoints, curvatureMetrics));
  
  invariants.complexityScore = curvatureMetrics.signatureComplexity;
  invariants.regularityScore = 1 - Math.min(1, curvatureMetrics.curvatureVariance);
  
  metrics.curvatureTime = curvatureTimer();
  metrics.algorithmsUsed.push('curvature');
  
  // 3. Affine Differential Geometry
  if (adaptedConfig.enableAffineAnalysis) {
    let affineTimer = timer();
    const affineMetrics = computeAffineMetrics(normalizedPoints);
    invariants.affineSignature = affineMetrics.affineSignature;
    invariants.arcLength = affineMetrics.totalAffineLength;
    metrics.affineTime = affineTimer();
    metrics.algorithmsUsed.push('affine');
  }
  
  // 4. Path Signature Analysis
  if (adaptedConfig.enablePathSignature) {
    let pathTimer = timer();
    const pathSignature = computePathSignature(normalizedPoints);
    invariants.pathSignature = pathSignature;
    metrics.pathSignatureTime = pathTimer();
    metrics.algorithmsUsed.push('pathSignature');
  }
  
  // 5. Topological Data Analysis
  if (adaptedConfig.enableTopologicalAnalysis) {
    let topoTimer = timer();
    const topoMetrics = computeTopologicalMetrics(normalizedPoints);
    invariants.bettiNumbers = topoMetrics.bettiNumbers;
    invariants.persistenceDiagram = topoMetrics.persistenceDiagram;
    metrics.topologicalTime = topoTimer();
    metrics.algorithmsUsed.push('topological');
  }
  
  // 6. Knot Theory Analysis
  if (adaptedConfig.enableKnotTheory) {
    let knotTimer = timer();
    const knotMetrics = computeKnotMetrics(normalizedPoints);
    invariants.writhe = knotMetrics.writhe;
    invariants.windingNumber = knotMetrics.windingNumber;
    invariants.totalTurning = knotMetrics.totalTurning;
    metrics.knotTheoryTime = knotTimer();
    metrics.algorithmsUsed.push('knotTheory');
  }
  
  // 7. Spectral Analysis
  if (adaptedConfig.enableSpectralAnalysis && processed.length > 0) {
    let spectralTimer = timer();
    const tangentAngles = processed.map(p => p.tangentAngle);
    const spectralMetrics = computeSpectralMetrics(tangentAngles);
    invariants.fourierCoefficients = {
      magnitude: spectralMetrics.fourierMagnitudes,
      phase: spectralMetrics.fourierPhases
    };
    invariants.spectralCentroid = spectralMetrics.spectralCentroid;
    invariants.waveletCoefficients = spectralMetrics.waveletCoefficients.approximation;
    metrics.spectralTime = spectralTimer();
    metrics.algorithmsUsed.push('spectral');
  }
  
  // 8. Calculate overall stability and symmetry
  invariants.stabilityIndex = landmarks.length > 0 ?
    landmarks.reduce((sum, l) => sum + l.stability, 0) / landmarks.length : 0;
  
  // Simple symmetry calculation
  invariants.symmetryScore = calculateSimpleSymmetry(normalizedPoints);
  
  // Record performance metrics
  metrics.totalTime = globalTimer();
  performanceHistory.push(metrics);
  
  // Keep only recent performance history
  if (performanceHistory.length > 20) {
    performanceHistory = performanceHistory.slice(-20);
  }
  
  // Calculate quality score
  const qualityScore = calculateQualityScore(adaptedConfig, metrics);
  
  const result: ProcessingResult = {
    processed,
    landmarks,
    invariants,
    processingTime: metrics.totalTime,
    qualityScore,
    cacheHit: false
  };
  
  // Cache result if enabled
  if (config.cacheResults) {
    const cacheKey = generateCacheKey(stroke, config);
    analysisCache.set(cacheKey, result);
    
    // Limit cache size
    if (analysisCache.size > 100) {
      const firstKey = analysisCache.keys().next().value;
      analysisCache.delete(firstKey);
    }
  }
  
  return result;
}

/**
 * Calculate simple symmetry score
 */
function calculateSimpleSymmetry(points: Point[]): number {
  if (points.length < 4) return 0;
  
  const centerX = 0.5; // Normalized coordinates
  let symmetrySum = 0;
  let count = 0;
  
  for (const point of points) {
    const mirroredX = 2 * centerX - point.x;
    const minDistance = Math.min(...points.map(p => 
      Math.sqrt((p.x - mirroredX) ** 2 + (p.y - point.y) ** 2)
    ));
    symmetrySum += Math.exp(-minDistance * 10);
    count++;
  }
  
  return count > 0 ? symmetrySum / count : 0;
}

/**
 * Calculate analysis quality score
 */
function calculateQualityScore(config: AnalysisConfig, metrics: PerformanceMetrics): number {
  let score = 0;
  
  // Performance component (higher is better if under target)
  const performanceScore = metrics.totalTime <= config.targetProcessingTime ? 1 :
    Math.max(0, 1 - (metrics.totalTime - config.targetProcessingTime) / config.targetProcessingTime);
  
  // Completeness component (more algorithms = higher quality)
  const maxAlgorithms = 7; // Total number of algorithm types
  const completenessScore = metrics.algorithmsUsed.length / maxAlgorithms;
  
  // Points processed component
  const pointsScore = Math.min(1, metrics.pointsProcessed / config.maxPoints);
  
  score = 0.4 * performanceScore + 0.4 * completenessScore + 0.2 * pointsScore;
  
  return Math.max(0, Math.min(1, score));
}

/**
 * Get performance statistics
 */
export function getPerformanceStats(): {
  averageTime: number;
  medianTime: number;
  successRate: number;
  recentTrend: 'improving' | 'stable' | 'degrading';
} {
  if (performanceHistory.length === 0) {
    return {
      averageTime: 0,
      medianTime: 0,
      successRate: 0,
      recentTrend: 'stable'
    };
  }
  
  const times = performanceHistory.map(p => p.totalTime);
  const averageTime = times.reduce((sum, t) => sum + t, 0) / times.length;
  
  const sortedTimes = [...times].sort((a, b) => a - b);
  const medianTime = sortedTimes[Math.floor(sortedTimes.length / 2)];
  
  const successRate = performanceHistory.filter(p => p.totalTime <= 150).length / performanceHistory.length;
  
  // Determine trend (compare recent half with earlier half)
  let recentTrend: 'improving' | 'stable' | 'degrading' = 'stable';
  if (performanceHistory.length >= 6) {
    const half = Math.floor(performanceHistory.length / 2);
    const recentAvg = performanceHistory.slice(half).reduce((sum, p) => sum + p.totalTime, 0) / (performanceHistory.length - half);
    const earlierAvg = performanceHistory.slice(0, half).reduce((sum, p) => sum + p.totalTime, 0) / half;
    
    if (recentAvg < earlierAvg * 0.9) recentTrend = 'improving';
    else if (recentAvg > earlierAvg * 1.1) recentTrend = 'degrading';
  }
  
  return {
    averageTime,
    medianTime,
    successRate,
    recentTrend
  };
}

/**
 * Clear performance history and cache
 */
export function clearPerformanceData(): void {
  performanceHistory = [];
  analysisCache.clear();
}

/**
 * Batch analysis for multiple strokes (useful for training data)
 */
export async function analyzeBatch(
  strokes: Stroke[],
  config: AnalysisConfig = defaultAnalysisConfig,
  progressCallback?: (progress: number) => void
): Promise<ProcessingResult[]> {
  const results: ProcessingResult[] = [];
  
  for (let i = 0; i < strokes.length; i++) {
    try {
      const result = await analyzeStroke(strokes[i], config);
      results.push(result);
      
      if (progressCallback) {
        progressCallback((i + 1) / strokes.length);
      }
      
      // Yield control to prevent blocking UI
      if (i % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 1));
      }
    } catch (error) {
      console.error(`Error analyzing stroke ${i}:`, error);
      // Add empty result to maintain array consistency
      results.push({
        processed: [],
        landmarks: [],
        invariants: invariants,
        processingTime: 0,
        qualityScore: 0,
        cacheHit: false
      });
    }
  }
  
  return results;
}

/**
 * Real-time analysis with adaptive quality
 */
export async function analyzeRealTime(
  stroke: Stroke,
  maxTime: number = 50
): Promise<ProcessingResult> {
  const quickConfig: AnalysisConfig = {
    enableAffineAnalysis: true,
    enableTopologicalAnalysis: false, // Disable for speed
    enablePathSignature: true,
    enableSpectralAnalysis: false, // Disable for speed
    enableKnotTheory: false, // Disable for speed
    targetProcessingTime: maxTime,
    adaptiveQuality: true,
    cacheResults: true,
    maxPoints: 50 // Reduced for speed
  };
  
  return analyzeStroke(stroke, quickConfig);
}
