import type { Point, InvariantMetrics, ThreeDSignature } from '@/store/researchStore';

/**
 * UMAP (Uniform Manifold Approximation and Projection) Implementation
 * Advanced dimensionality reduction for creating optimal 3D stroke signatures
 */

export interface UMAPConfig {
  nNeighbors: number;
  minDist: number;
  nComponents: number;
  metric: 'euclidean' | 'cosine' | 'manhattan';
  learningRate: number;
  nEpochs: number;
  spread: number;
  randomSeed?: number;
}

export interface NeighborGraph {
  indices: number[][];
  distances: number[][];
  weights: number[][];
}

export interface EmbeddingResult {
  coordinates: number[][];
  stress: number;
  trustworthiness: number;
  continuity: number;
}

/**
 * Default UMAP configuration for stroke analysis
 */
export const defaultUMAPConfig: UMAPConfig = {
  nNeighbors: 15,
  minDist: 0.1,
  nComponents: 3,
  metric: 'euclidean',
  learningRate: 1.0,
  nEpochs: 200,
  spread: 1.0,
  randomSeed: 42
};

/**
 * Calculate distance between two feature vectors
 */
export function calculateDistance(
  vector1: number[], 
  vector2: number[], 
  metric: string = 'euclidean'
): number {
  if (vector1.length !== vector2.length) {
    throw new Error('Vectors must have the same length');
  }

  switch (metric) {
    case 'euclidean': {
      let sum = 0;
      for (let i = 0; i < vector1.length; i++) {
        const diff = vector1[i] - vector2[i];
        sum += diff * diff;
      }
      return Math.sqrt(sum);
    }
    
    case 'cosine': {
      let dot = 0;
      let norm1 = 0;
      let norm2 = 0;
      
      for (let i = 0; i < vector1.length; i++) {
        dot += vector1[i] * vector2[i];
        norm1 += vector1[i] * vector1[i];
        norm2 += vector2[i] * vector2[i];
      }
      
      const normProduct = Math.sqrt(norm1) * Math.sqrt(norm2);
      return normProduct > 0 ? 1 - (dot / normProduct) : 1;
    }
    
    case 'manhattan': {
      let sum = 0;
      for (let i = 0; i < vector1.length; i++) {
        sum += Math.abs(vector1[i] - vector2[i]);
      }
      return sum;
    }
    
    default:
      throw new Error(`Unknown metric: ${metric}`);
  }
}

/**
 * Build k-nearest neighbor graph
 */
export function buildKNNGraph(
  data: number[][], 
  config: UMAPConfig
): NeighborGraph {
  const n = data.length;
  const indices: number[][] = [];
  const distances: number[][] = [];
  
  for (let i = 0; i < n; i++) {
    const neighborData: Array<{ index: number; distance: number }> = [];
    
    // Calculate distances to all other points
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const distance = calculateDistance(data[i], data[j], config.metric);
        neighborData.push({ index: j, distance });
      }
    }
    
    // Sort by distance and take k nearest neighbors
    neighborData.sort((a, b) => a.distance - b.distance);
    const kNearest = neighborData.slice(0, config.nNeighbors);
    
    indices.push(kNearest.map(n => n.index));
    distances.push(kNearest.map(n => n.distance));
  }
  
  // Calculate fuzzy set weights
  const weights = calculateFuzzySetWeights(distances, config);
  
  return { indices, distances, weights };
}

/**
 * Calculate fuzzy set membership weights
 */
function calculateFuzzySetWeights(
  distances: number[][], 
  config: UMAPConfig
): number[][] {
  const weights: number[][] = [];
  
  for (const pointDistances of distances) {
    const pointWeights: number[] = [];
    
    // Find local connectivity (distance to nearest neighbor)
    const rho = pointDistances.length > 0 ? pointDistances[0] : 0;
    
    // Calculate sigma using binary search for smooth k-nearest neighbor
    const targetLog2 = Math.log2(config.nNeighbors);
    let sigma = 1.0;
    
    // Binary search for optimal sigma
    let low = 1e-10;
    let high = 1000.0;
    
    for (let iter = 0; iter < 64; iter++) {
      sigma = (low + high) / 2;
      
      let sumWeights = 0;
      for (const dist of pointDistances) {
        const adjustedDist = Math.max(0, dist - rho);
        sumWeights += Math.exp(-adjustedDist / sigma);
      }
      
      const log2Sum = Math.log2(sumWeights);
      
      if (Math.abs(log2Sum - targetLog2) < 1e-5) break;
      
      if (log2Sum > targetLog2) {
        high = sigma;
      } else {
        low = sigma;
      }
    }
    
    // Calculate final weights
    for (const dist of pointDistances) {
      const adjustedDist = Math.max(0, dist - rho);
      const weight = Math.exp(-adjustedDist / sigma);
      pointWeights.push(weight);
    }
    
    weights.push(pointWeights);
  }
  
  return weights;
}

/**
 * Symmetrize the neighbor graph
 */
function symmetrizeGraph(graph: NeighborGraph): NeighborGraph {
  const n = graph.indices.length;
  const symmetricIndices: number[][] = Array(n).fill(null).map(() => []);
  const symmetricDistances: number[][] = Array(n).fill(null).map(() => []);
  const symmetricWeights: number[][] = Array(n).fill(null).map(() => []);
  
  // Build adjacency map
  const adjacency = new Map<string, { distance: number; weight: number }>();
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < graph.indices[i].length; j++) {
      const neighbor = graph.indices[i][j];
      const distance = graph.distances[i][j];
      const weight = graph.weights[i][j];
      
      const key1 = `${i}-${neighbor}`;
      const key2 = `${neighbor}-${i}`;
      
      if (!adjacency.has(key1)) {
        adjacency.set(key1, { distance, weight });
      }
      
      if (!adjacency.has(key2)) {
        adjacency.set(key2, { distance, weight });
      }
    }
  }
  
  // Build symmetric graph
  for (let i = 0; i < n; i++) {
    const neighbors = new Set<number>();
    
    // Add all neighbors from original graph
    for (const neighbor of graph.indices[i]) {
      neighbors.add(neighbor);
    }
    
    // Add reverse neighbors
    for (let j = 0; j < n; j++) {
      if (graph.indices[j].includes(i)) {
        neighbors.add(j);
      }
    }
    
    const sortedNeighbors = Array.from(neighbors).sort((a, b) => a - b);
    
    for (const neighbor of sortedNeighbors) {
      const key = `${i}-${neighbor}`;
      const entry = adjacency.get(key);
      
      if (entry) {
        symmetricIndices[i].push(neighbor);
        symmetricDistances[i].push(entry.distance);
        symmetricWeights[i].push(entry.weight);
      }
    }
  }
  
  return {
    indices: symmetricIndices,
    distances: symmetricDistances,
    weights: symmetricWeights
  };
}

/**
 * Initialize low-dimensional embedding
 */
function initializeEmbedding(n: number, nComponents: number, seed?: number): number[][] {
  // Use seeded random number generator for reproducibility
  let randomSeed = seed || 42;
  const random = () => {
    randomSeed = (randomSeed * 9301 + 49297) % 233280;
    return randomSeed / 233280;
  };
  
  const embedding: number[][] = [];
  
  for (let i = 0; i < n; i++) {
    const point: number[] = [];
    for (let j = 0; j < nComponents; j++) {
      // Initialize with small random values
      point.push((random() - 0.5) * 20);
    }
    embedding.push(point);
  }
  
  return embedding;
}

/**
 * Calculate attractive and repulsive forces for UMAP optimization
 */
function calculateForces(
  embedding: number[][],
  graph: NeighborGraph,
  config: UMAPConfig
): number[][] {
  const n = embedding.length;
  const nComponents = embedding[0].length;
  const forces: number[][] = Array(n).fill(null).map(() => new Array(nComponents).fill(0));
  
  // Attractive forces (from high-dimensional neighbors)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < graph.indices[i].length; j++) {
      const neighbor = graph.indices[i][j];
      const weight = graph.weights[i][j];
      
      const distance = calculateDistance(embedding[i], embedding[neighbor], 'euclidean');
      const attraction = weight * Math.max(0, 1 - distance / config.spread);
      
      for (let d = 0; d < nComponents; d++) {
        const direction = embedding[neighbor][d] - embedding[i][d];
        forces[i][d] += attraction * direction * config.learningRate;
      }
    }
  }
  
  // Repulsive forces (sample random points)
  const nNegativeSamples = 5;
  
  for (let i = 0; i < n; i++) {
    for (let sample = 0; sample < nNegativeSamples; sample++) {
      const j = Math.floor(Math.random() * n);
      if (i === j) continue;
      
      const distance = calculateDistance(embedding[i], embedding[j], 'euclidean');
      const repulsion = 1 / (1 + distance * distance / config.spread);
      
      for (let d = 0; d < nComponents; d++) {
        const direction = embedding[i][d] - embedding[j][d];
        forces[i][d] += repulsion * direction * config.learningRate * 0.5;
      }
    }
  }
  
  return forces;
}

/**
 * Run UMAP optimization
 */
export function runUMAP(data: number[][], config: UMAPConfig): EmbeddingResult {
  if (data.length === 0) {
    return {
      coordinates: [],
      stress: 0,
      trustworthiness: 0,
      continuity: 0
    };
  }
  
  // Build neighbor graph
  const graph = buildKNNGraph(data, config);
  const symmetricGraph = symmetrizeGraph(graph);
  
  // Initialize embedding
  let embedding = initializeEmbedding(data.length, config.nComponents, config.randomSeed);
  
  // Optimization loop
  for (let epoch = 0; epoch < config.nEpochs; epoch++) {
    const forces = calculateForces(embedding, symmetricGraph, config);
    
    // Update positions
    for (let i = 0; i < embedding.length; i++) {
      for (let j = 0; j < config.nComponents; j++) {
        embedding[i][j] += forces[i][j];
      }
    }
    
    // Adaptive learning rate
    if (epoch > config.nEpochs / 2) {
      config.learningRate *= 0.99;
    }
  }
  
  // Calculate quality metrics
  const stress = calculateStress(data, embedding, config.metric);
  const trustworthiness = calculateTrustworthiness(data, embedding, config);
  const continuity = calculateContinuity(data, embedding, config);
  
  return {
    coordinates: embedding,
    stress,
    trustworthiness,
    continuity
  };
}

/**
 * Calculate stress (quality metric)
 */
function calculateStress(
  originalData: number[][],
  embedding: number[][],
  metric: string
): number {
  const n = originalData.length;
  let stress = 0;
  let totalDistance = 0;
  
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const originalDist = calculateDistance(originalData[i], originalData[j], metric);
      const embeddedDist = calculateDistance(embedding[i], embedding[j], 'euclidean');
      
      stress += Math.pow(originalDist - embeddedDist, 2);
      totalDistance += originalDist * originalDist;
    }
  }
  
  return totalDistance > 0 ? Math.sqrt(stress / totalDistance) : 0;
}

/**
 * Calculate trustworthiness (preservation of neighborhoods)
 */
function calculateTrustworthiness(
  originalData: number[][],
  embedding: number[][],
  config: UMAPConfig
): number {
  const n = originalData.length;
  const k = Math.min(config.nNeighbors, n - 1);
  
  let trustworthiness = 0;
  
  for (let i = 0; i < n; i++) {
    // Find k nearest neighbors in embedding
    const embeddingDistances: Array<{ index: number; distance: number }> = [];
    
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const distance = calculateDistance(embedding[i], embedding[j], 'euclidean');
        embeddingDistances.push({ index: j, distance });
      }
    }
    
    embeddingDistances.sort((a, b) => a.distance - b.distance);
    const embeddingNeighbors = embeddingDistances.slice(0, k).map(d => d.index);
    
    // Find k nearest neighbors in original space
    const originalDistances: Array<{ index: number; distance: number }> = [];
    
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const distance = calculateDistance(originalData[i], originalData[j], config.metric);
        originalDistances.push({ index: j, distance });
      }
    }
    
    originalDistances.sort((a, b) => a.distance - b.distance);
    const originalNeighbors = new Set(originalDistances.slice(0, k).map(d => d.index));
    
    // Count how many embedding neighbors are also original neighbors
    let matches = 0;
    for (const neighbor of embeddingNeighbors) {
      if (originalNeighbors.has(neighbor)) {
        matches++;
      }
    }
    
    trustworthiness += matches / k;
  }
  
  return trustworthiness / n;
}

/**
 * Calculate continuity (no gaps in embedding)
 */
function calculateContinuity(
  originalData: number[][],
  embedding: number[][],
  config: UMAPConfig
): number {
  const n = originalData.length;
  const k = Math.min(config.nNeighbors, n - 1);
  
  let continuity = 0;
  
  for (let i = 0; i < n; i++) {
    // Find k nearest neighbors in original space
    const originalDistances: Array<{ index: number; distance: number }> = [];
    
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const distance = calculateDistance(originalData[i], originalData[j], config.metric);
        originalDistances.push({ index: j, distance });
      }
    }
    
    originalDistances.sort((a, b) => a.distance - b.distance);
    const originalNeighbors = originalDistances.slice(0, k).map(d => d.index);
    
    // Find k nearest neighbors in embedding
    const embeddingDistances: Array<{ index: number; distance: number }> = [];
    
    for (let j = 0; j < n; j++) {
      if (i !== j) {
        const distance = calculateDistance(embedding[i], embedding[j], 'euclidean');
        embeddingDistances.push({ index: j, distance });
      }
    }
    
    embeddingDistances.sort((a, b) => a.distance - b.distance);
    const embeddingNeighbors = new Set(embeddingDistances.slice(0, k).map(d => d.index));
    
    // Count how many original neighbors are also embedding neighbors
    let matches = 0;
    for (const neighbor of originalNeighbors) {
      if (embeddingNeighbors.has(neighbor)) {
        matches++;
      }
    }
    
    continuity += matches / k;
  }
  
  return continuity / n;
}

/**
 * Extract comprehensive feature vector for UMAP input
 */
export function extractComprehensiveFeatures(invariants: InvariantMetrics): number[] {
  const features: number[] = [];
  
  // Geometric features
  features.push(invariants.arcLength);
  features.push(invariants.totalTurning);
  features.push(invariants.windingNumber);
  features.push(invariants.writhe);
  
  // Topological features
  features.push(invariants.bettiNumbers.b0);
  features.push(invariants.bettiNumbers.b1);
  features.push(invariants.bettiNumbers.b2);
  
  // Statistical features
  features.push(invariants.complexityScore);
  features.push(invariants.regularityScore);
  features.push(invariants.symmetryScore);
  features.push(invariants.stabilityIndex);
  
  // Path signature features (first 10 components)
  const pathSigFeatures = [
    ...invariants.pathSignature.level1,
    ...invariants.pathSignature.level2.slice(0, 4),
    ...invariants.pathSignature.logSignature.slice(0, 4)
  ];
  features.push(...pathSigFeatures);
  
  // Affine signature features (statistical measures)
  if (invariants.affineSignature.length > 0) {
    const kappaValues = invariants.affineSignature.map(s => s.kappa_a);
    const meanKappa = kappaValues.reduce((sum, k) => sum + k, 0) / kappaValues.length;
    const varKappa = kappaValues.reduce((sum, k) => sum + (k - meanKappa) ** 2, 0) / kappaValues.length;
    const maxKappa = Math.max(...kappaValues.map(Math.abs));
    
    features.push(meanKappa, Math.sqrt(varKappa), maxKappa);
  } else {
    features.push(0, 0, 0);
  }
  
  // Fourier features (first 8 coefficients)
  features.push(...invariants.fourierCoefficients.magnitude.slice(0, 8));
  
  // Ensure fixed length
  while (features.length < 50) {
    features.push(0);
  }
  
  return features.slice(0, 50);
}

/**
 * Create optimized 3D signature using UMAP
 */
export function create3DSignatureUMAP(
  allInvariants: InvariantMetrics[],
  targetIndex: number,
  config: Partial<UMAPConfig> = {}
): ThreeDSignature {
  const fullConfig = { ...defaultUMAPConfig, ...config };
  
  if (allInvariants.length === 0) {
    return {
      coordinates: [],
      axisMapping: { x: 'umap_0', y: 'umap_1', z: 'umap_2' },
      qualityMetrics: {
        separationRatio: 0,
        intraClusterVariance: 0,
        stabilityIndex: 0
      }
    };
  }
  
  // Extract feature vectors for all strokes
  const featureVectors = allInvariants.map(extractComprehensiveFeatures);
  
  // Run UMAP
  const umapResult = runUMAP(featureVectors, fullConfig);
  
  // Extract coordinates for target stroke
  const targetCoordinates = umapResult.coordinates[targetIndex] || [0, 0, 0];
  
  // Create signature points (for visualization, we'll interpolate along the stroke)
  const numPoints = 100; // Standard number of points
  const coordinates = [];
  
  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1);
    
    // Use UMAP coordinates as base and add local variation
    const invarianceWeight = allInvariants[targetIndex]?.stabilityIndex || 0.5;
    
    coordinates.push({
      x: targetCoordinates[0] + Math.sin(t * Math.PI * 2) * 0.1,
      y: targetCoordinates[1] + Math.cos(t * Math.PI * 2) * 0.1,
      z: targetCoordinates[2] + t * 0.2,
      invarianceWeight
    });
  }
  
  // Calculate quality metrics
  const qualityMetrics = {
    separationRatio: umapResult.trustworthiness * 10,
    intraClusterVariance: umapResult.stress,
    stabilityIndex: umapResult.continuity
  };
  
  return {
    coordinates,
    axisMapping: { 
      x: 'UMAP Component 1', 
      y: 'UMAP Component 2', 
      z: 'UMAP Component 3' 
    },
    qualityMetrics
  };
}
