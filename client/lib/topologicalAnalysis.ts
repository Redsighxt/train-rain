import type { Point } from '@/store/researchStore';

/**
 * Topological Data Analysis Module
 * Implements persistent homology, Betti numbers, and topological invariants
 * for handwriting stroke analysis
 */

export interface PersistenceInterval {
  birth: number;
  death: number;
  dimension: number; // 0 for components, 1 for loops, 2 for voids
  persistence: number; // death - birth
}

export interface TopologicalMetrics {
  bettiNumbers: { b0: number; b1: number; b2: number };
  persistenceDiagram: PersistenceInterval[];
  totalPersistence: number;
  maxPersistence: number;
  topologicalComplexity: number;
  loopSignature: number[];
  eulerCharacteristic: number;
}

export interface SimplexComplex {
  vertices: Point[];
  edges: Array<[number, number]>;
  triangles: Array<[number, number, number]>;
  distances: number[][];
}

/**
 * Build distance matrix between all points
 */
export function buildDistanceMatrix(points: Point[]): number[][] {
  const n = points.length;
  const distances: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      if (i === j) {
        distances[i][j] = 0;
      } else {
        const dx = points[i].x - points[j].x;
        const dy = points[i].y - points[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        distances[i][j] = dist;
        distances[j][i] = dist;
      }
    }
  }
  
  return distances;
}

/**
 * Build Vietoris-Rips complex for given epsilon
 */
export function buildVietorisRipsComplex(points: Point[], epsilon: number): SimplexComplex {
  const n = points.length;
  const distances = buildDistanceMatrix(points);
  
  // Find edges (pairs of points within epsilon distance)
  const edges: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (distances[i][j] <= epsilon) {
        edges.push([i, j]);
      }
    }
  }
  
  // Find triangles (triplets where all pairwise distances are <= epsilon)
  const triangles: Array<[number, number, number]> = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        if (distances[i][j] <= epsilon && 
            distances[i][k] <= epsilon && 
            distances[j][k] <= epsilon) {
          triangles.push([i, j, k]);
        }
      }
    }
  }
  
  return {
    vertices: points,
    edges,
    triangles,
    distances
  };
}

/**
 * Union-Find data structure for tracking connected components
 */
class UnionFind {
  private parent: number[];
  private rank: number[];
  public componentCount: number;
  
  constructor(n: number) {
    this.parent = Array(n).fill(0).map((_, i) => i);
    this.rank = Array(n).fill(0);
    this.componentCount = n;
  }
  
  find(x: number): number {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]);
    }
    return this.parent[x];
  }
  
  union(x: number, y: number): boolean {
    const rootX = this.find(x);
    const rootY = this.find(y);
    
    if (rootX === rootY) return false;
    
    if (this.rank[rootX] < this.rank[rootY]) {
      this.parent[rootX] = rootY;
    } else if (this.rank[rootX] > this.rank[rootY]) {
      this.parent[rootY] = rootX;
    } else {
      this.parent[rootY] = rootX;
      this.rank[rootX]++;
    }
    
    this.componentCount--;
    return true;
  }
}

/**
 * Calculate persistent homology using filtration
 */
export function calculatePersistentHomology(points: Point[], maxEpsilon: number, steps: number): PersistenceInterval[] {
  if (points.length < 2) return [];
  
  const distances = buildDistanceMatrix(points);
  const n = points.length;
  
  // Get all unique distances and sort them
  const uniqueDistances: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (distances[i][j] <= maxEpsilon) {
        uniqueDistances.push(distances[i][j]);
      }
    }
  }
  
  uniqueDistances.sort((a, b) => a - b);
  const filtrationValues = [0, ...uniqueDistances.slice(0, steps)];
  
  const intervals: PersistenceInterval[] = [];
  
  // Track 0-dimensional homology (connected components)
  const unionFind = new UnionFind(n);
  const componentBirths = Array(n).fill(0).map((_, i) => ({ id: i, birth: 0 }));
  
  // Process each filtration step
  for (let step = 1; step < filtrationValues.length; step++) {
    const epsilon = filtrationValues[step];
    const complex = buildVietorisRipsComplex(points, epsilon);
    
    // Process edges for 0-dimensional homology
    for (const [i, j] of complex.edges) {
      if (distances[i][j] <= epsilon) {
        const wasUnited = unionFind.union(i, j);
        if (wasUnited) {
          // A component died - record the persistence interval
          const olderRoot = Math.min(unionFind.find(i), unionFind.find(j));
          const birth = componentBirths.find(c => unionFind.find(c.id) === olderRoot)?.birth || 0;
          
          intervals.push({
            birth,
            death: epsilon,
            dimension: 0,
            persistence: epsilon - birth
          });
        }
      }
    }
  }
  
  // Add infinite intervals for remaining components
  const finalComponents = new Set();
  for (let i = 0; i < n; i++) {
    finalComponents.add(unionFind.find(i));
  }
  
  finalComponents.forEach(root => {
    const birth = componentBirths.find(c => unionFind.find(c.id) === root)?.birth || 0;
    intervals.push({
      birth,
      death: Infinity,
      dimension: 0,
      persistence: Infinity
    });
  });
  
  // Simple 1-dimensional homology (loops) detection
  intervals.push(...detect1DHomology(points, filtrationValues));
  
  return intervals.filter(interval => interval.persistence > 1e-10);
}

/**
 * Detect 1-dimensional homology (loops) using a simplified approach
 */
function detect1DHomology(points: Point[], filtrationValues: number[]): PersistenceInterval[] {
  const intervals: PersistenceInterval[] = [];
  const n = points.length;
  
  if (n < 3) return intervals;
  
  // Look for closed loops in the stroke
  for (let step = 1; step < filtrationValues.length; step++) {
    const epsilon = filtrationValues[step];
    
    // Check for loops by finding cycles in the graph
    const adjacencyList: number[][] = Array(n).fill(null).map(() => []);
    const distances = buildDistanceMatrix(points);
    
    // Build adjacency list
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (distances[i][j] <= epsilon) {
          adjacencyList[i].push(j);
          adjacencyList[j].push(i);
        }
      }
    }
    
    // Detect cycles using DFS
    const visited = Array(n).fill(false);
    const parent = Array(n).fill(-1);
    
    for (let i = 0; i < n; i++) {
      if (!visited[i]) {
        if (hasCycleDFS(i, visited, parent, adjacencyList)) {
          intervals.push({
            birth: epsilon,
            death: filtrationValues[Math.min(step + 1, filtrationValues.length - 1)],
            dimension: 1,
            persistence: filtrationValues[Math.min(step + 1, filtrationValues.length - 1)] - epsilon
          });
        }
      }
    }
  }
  
  return intervals;
}

/**
 * DFS helper for cycle detection
 */
function hasCycleDFS(v: number, visited: boolean[], parent: number[], adj: number[][]): boolean {
  visited[v] = true;
  
  for (const neighbor of adj[v]) {
    if (!visited[neighbor]) {
      parent[neighbor] = v;
      if (hasCycleDFS(neighbor, visited, parent, adj)) {
        return true;
      }
    } else if (parent[v] !== neighbor) {
      return true;
    }
  }
  
  return false;
}

/**
 * Calculate Betti numbers from persistence diagram
 */
export function calculateBettiNumbers(intervals: PersistenceInterval[], epsilon: number): { b0: number; b1: number; b2: number } {
  let b0 = 0; // Connected components
  let b1 = 0; // Loops
  let b2 = 0; // Voids
  
  for (const interval of intervals) {
    if (interval.birth <= epsilon && (interval.death > epsilon || interval.death === Infinity)) {
      switch (interval.dimension) {
        case 0: b0++; break;
        case 1: b1++; break;
        case 2: b2++; break;
      }
    }
  }
  
  return { b0, b1, b2 };
}

/**
 * Calculate loop signature - characteristic lengths of detected loops
 */
export function calculateLoopSignature(points: Point[]): number[] {
  if (points.length < 4) return [];
  
  const signature: number[] = [];
  const n = points.length;
  
  // Look for potential loops by finding points that are close to much earlier points
  for (let i = 3; i < n; i++) {
    for (let j = 0; j < i - 2; j++) {
      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      // If we're close to a much earlier point, we might have a loop
      if (distance < 0.1) { // Threshold for loop closure
        const loopLength = i - j;
        const normalizedLength = loopLength / n;
        signature.push(normalizedLength);
      }
    }
  }
  
  return signature.sort((a, b) => b - a); // Sort by length, largest first
}

/**
 * Calculate topological complexity based on persistence
 */
export function calculateTopologicalComplexity(intervals: PersistenceInterval[]): number {
  if (intervals.length === 0) return 0;
  
  // Weight intervals by their persistence and dimension
  let complexity = 0;
  
  for (const interval of intervals) {
    if (interval.persistence !== Infinity && interval.persistence > 0) {
      const weight = interval.dimension === 0 ? 1 : interval.dimension === 1 ? 2 : 3;
      complexity += weight * Math.log(1 + interval.persistence);
    }
  }
  
  return complexity;
}

/**
 * Main function to compute all topological metrics
 */
export function computeTopologicalMetrics(points: Point[]): TopologicalMetrics {
  if (points.length < 2) {
    return {
      bettiNumbers: { b0: 1, b1: 0, b2: 0 },
      persistenceDiagram: [],
      totalPersistence: 0,
      maxPersistence: 0,
      topologicalComplexity: 0,
      loopSignature: [],
      eulerCharacteristic: 1
    };
  }
  
  // Normalize points to unit square for consistent analysis
  const normalized = normalizePoints(points);
  
  // Calculate persistence diagram
  const maxEpsilon = 0.5; // Half the diagonal of unit square
  const steps = 20;
  const persistenceDiagram = calculatePersistentHomology(normalized, maxEpsilon, steps);
  
  // Calculate Betti numbers at a representative scale
  const representativeEpsilon = 0.1;
  const bettiNumbers = calculateBettiNumbers(persistenceDiagram, representativeEpsilon);
  
  // Calculate various metrics
  const validIntervals = persistenceDiagram.filter(i => i.persistence !== Infinity && i.persistence > 0);
  const totalPersistence = validIntervals.reduce((sum, i) => sum + i.persistence, 0);
  const maxPersistence = validIntervals.length > 0 ? Math.max(...validIntervals.map(i => i.persistence)) : 0;
  
  const topologicalComplexity = calculateTopologicalComplexity(persistenceDiagram);
  const loopSignature = calculateLoopSignature(normalized);
  
  // Euler characteristic: χ = b0 - b1 + b2
  const eulerCharacteristic = bettiNumbers.b0 - bettiNumbers.b1 + bettiNumbers.b2;
  
  return {
    bettiNumbers,
    persistenceDiagram,
    totalPersistence,
    maxPersistence,
    topologicalComplexity,
    loopSignature,
    eulerCharacteristic
  };
}

/**
 * Normalize points to unit square
 */
function normalizePoints(points: Point[]): Point[] {
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

/**
 * Compare topological signatures
 */
export function compareTopologicalSignatures(
  metrics1: TopologicalMetrics, 
  metrics2: TopologicalMetrics
): number {
  // Compare Betti numbers
  const bettiSimilarity = 1 - Math.abs(
    (metrics1.bettiNumbers.b0 - metrics2.bettiNumbers.b0) +
    (metrics1.bettiNumbers.b1 - metrics2.bettiNumbers.b1) * 2 + // Weight loops more heavily
    (metrics1.bettiNumbers.b2 - metrics2.bettiNumbers.b2) * 3
  ) / 10;
  
  // Compare topological complexity
  const complexitySimilarity = 1 - Math.abs(
    metrics1.topologicalComplexity - metrics2.topologicalComplexity
  ) / (Math.max(metrics1.topologicalComplexity, metrics2.topologicalComplexity) + 1);
  
  // Compare loop signatures
  const maxLoops = Math.max(metrics1.loopSignature.length, metrics2.loopSignature.length);
  let loopSimilarity = 1;
  
  if (maxLoops > 0) {
    let loopDifference = 0;
    for (let i = 0; i < maxLoops; i++) {
      const loop1 = metrics1.loopSignature[i] || 0;
      const loop2 = metrics2.loopSignature[i] || 0;
      loopDifference += Math.abs(loop1 - loop2);
    }
    loopSimilarity = 1 - loopDifference / maxLoops;
  }
  
  // Weighted combination
  return 0.4 * bettiSimilarity + 0.3 * complexitySimilarity + 0.3 * loopSimilarity;
}

/**
 * Extract topological features for machine learning
 */
export function extractTopologicalFeatures(metrics: TopologicalMetrics): number[] {
  const features: number[] = [];
  
  // Betti numbers
  features.push(metrics.bettiNumbers.b0);
  features.push(metrics.bettiNumbers.b1);
  features.push(metrics.bettiNumbers.b2);
  
  // Persistence metrics
  features.push(metrics.totalPersistence);
  features.push(metrics.maxPersistence);
  features.push(metrics.topologicalComplexity);
  
  // Euler characteristic
  features.push(metrics.eulerCharacteristic);
  
  // Loop signature features (pad or truncate to fixed size)
  const maxLoopFeatures = 5;
  for (let i = 0; i < maxLoopFeatures; i++) {
    features.push(metrics.loopSignature[i] || 0);
  }
  
  // Statistical features of persistence diagram
  const validIntervals = metrics.persistenceDiagram.filter(i => 
    i.persistence !== Infinity && i.persistence > 0
  );
  
  if (validIntervals.length > 0) {
    const persistences = validIntervals.map(i => i.persistence);
    const meanPersistence = persistences.reduce((sum, p) => sum + p, 0) / persistences.length;
    const variance = persistences.reduce((sum, p) => sum + (p - meanPersistence) ** 2, 0) / persistences.length;
    
    features.push(meanPersistence);
    features.push(Math.sqrt(variance));
    features.push(persistences.length);
  } else {
    features.push(0, 0, 0);
  }
  
  return features;
}
