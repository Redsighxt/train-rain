import type { Point } from '@/store/researchStore';

/**
 * Alpha Shapes Computation Module
 * Implements alpha shapes for advanced topological analysis of stroke patterns
 */

export interface AlphaShapeResult {
  vertices: Point[];
  edges: Array<[number, number]>;
  triangles: Array<[number, number, number]>;
  alpha: number;
  boundaryLength: number;
  area: number;
  holes: Array<Point[]>;
}

export interface AlphaComplex {
  simplicies: {
    vertices: Point[];
    edges: Array<{ indices: [number, number]; alpha: number }>;
    triangles: Array<{ indices: [number, number, number]; alpha: number }>;
  };
  criticalAlphas: number[];
}

/**
 * Compute Delaunay triangulation (simplified version)
 */
function computeDelaunayTriangulation(points: Point[]): Array<[number, number, number]> {
  const triangles: Array<[number, number, number]> = [];
  const n = points.length;
  
  if (n < 3) return triangles;
  
  // Simple triangulation using incremental algorithm
  for (let i = 0; i < n - 2; i++) {
    for (let j = i + 1; j < n - 1; j++) {
      for (let k = j + 1; k < n; k++) {
        // Check if triangle is valid (not degenerate)
        const p1 = points[i];
        const p2 = points[j];
        const p3 = points[k];
        
        const area = Math.abs(
          (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
        ) / 2;
        
        if (area > 1e-10) {
          // Check if any other point is inside the circumcircle
          const circumcenter = computeCircumcenter(p1, p2, p3);
          const circumradius = distance(p1, circumcenter);
          
          let isDelaunay = true;
          for (let l = 0; l < n; l++) {
            if (l !== i && l !== j && l !== k) {
              if (distance(points[l], circumcenter) < circumradius - 1e-10) {
                isDelaunay = false;
                break;
              }
            }
          }
          
          if (isDelaunay) {
            triangles.push([i, j, k]);
          }
        }
      }
    }
  }
  
  return triangles;
}

/**
 * Compute circumcenter of a triangle
 */
function computeCircumcenter(p1: Point, p2: Point, p3: Point): Point {
  const D = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));
  
  if (Math.abs(D) < 1e-10) {
    // Degenerate triangle, return centroid
    return {
      x: (p1.x + p2.x + p3.x) / 3,
      y: (p1.y + p2.y + p3.y) / 3,
      time: (p1.time + p2.time + p3.time) / 3
    };
  }
  
  const ux = (p1.x * p1.x + p1.y * p1.y) * (p2.y - p3.y) +
            (p2.x * p2.x + p2.y * p2.y) * (p3.y - p1.y) +
            (p3.x * p3.x + p3.y * p3.y) * (p1.y - p2.y);
  
  const uy = (p1.x * p1.x + p1.y * p1.y) * (p3.x - p2.x) +
            (p2.x * p2.x + p2.y * p2.y) * (p1.x - p3.x) +
            (p3.x * p3.x + p3.y * p3.y) * (p2.x - p1.x);
  
  return {
    x: ux / D,
    y: uy / D,
    time: 0
  };
}

/**
 * Compute circumradius of a triangle
 */
function computeCircumradius(p1: Point, p2: Point, p3: Point): number {
  const a = distance(p2, p3);
  const b = distance(p1, p3);
  const c = distance(p1, p2);
  
  const area = Math.abs(
    (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
  ) / 2;
  
  if (area < 1e-10) return Infinity;
  
  return (a * b * c) / (4 * area);
}

/**
 * Distance between two points
 */
function distance(p1: Point, p2: Point): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Compute alpha complex from Delaunay triangulation
 */
function computeAlphaComplex(points: Point[]): AlphaComplex {
  const triangles = computeDelaunayTriangulation(points);
  const n = points.length;
  
  // Initialize alpha complex
  const alphaComplex: AlphaComplex = {
    simplicies: {
      vertices: points,
      edges: [],
      triangles: []
    },
    criticalAlphas: []
  };
  
  const alphaValues = new Set<number>();
  
  // Process triangles
  for (const [i, j, k] of triangles) {
    const circumradius = computeCircumradius(points[i], points[j], points[k]);
    alphaComplex.simplicies.triangles.push({
      indices: [i, j, k],
      alpha: circumradius
    });
    alphaValues.add(circumradius);
  }
  
  // Process edges
  const edgeMap = new Map<string, { indices: [number, number]; alpha: number }>();
  
  // Add edges from triangles
  for (const triangle of alphaComplex.simplicies.triangles) {
    const [i, j, k] = triangle.indices;
    const edges = [[i, j], [j, k], [i, k]];
    
    for (const [a, b] of edges) {
      const key = a < b ? `${a}-${b}` : `${b}-${a}`;
      const edgeAlpha = Math.min(triangle.alpha, 
        distance(points[a], points[b]) / 2
      );
      
      if (!edgeMap.has(key) || edgeMap.get(key)!.alpha > edgeAlpha) {
        edgeMap.set(key, {
          indices: a < b ? [a, b] : [b, a],
          alpha: edgeAlpha
        });
        alphaValues.add(edgeAlpha);
      }
    }
  }
  
  alphaComplex.simplicies.edges = Array.from(edgeMap.values());
  
  // Sort critical alpha values
  alphaComplex.criticalAlphas = Array.from(alphaValues).sort((a, b) => a - b);
  
  return alphaComplex;
}

/**
 * Extract alpha shape for given alpha value
 */
export function computeAlphaShape(points: Point[], alpha: number): AlphaShapeResult {
  if (points.length < 3) {
    return {
      vertices: points,
      edges: [],
      triangles: [],
      alpha,
      boundaryLength: 0,
      area: 0,
      holes: []
    };
  }
  
  const alphaComplex = computeAlphaComplex(points);
  
  // Filter simplicies by alpha value
  const validTriangles = alphaComplex.simplicies.triangles
    .filter(triangle => triangle.alpha <= alpha)
    .map(triangle => triangle.indices);
  
  const validEdges = alphaComplex.simplicies.edges
    .filter(edge => edge.alpha <= alpha)
    .map(edge => edge.indices);
  
  // Find boundary edges (edges that belong to only one triangle)
  const edgeCount = new Map<string, number>();
  
  for (const [i, j, k] of validTriangles) {
    const edges = [
      [Math.min(i, j), Math.max(i, j)],
      [Math.min(j, k), Math.max(j, k)],
      [Math.min(i, k), Math.max(i, k)]
    ];
    
    for (const [a, b] of edges) {
      const key = `${a}-${b}`;
      edgeCount.set(key, (edgeCount.get(key) || 0) + 1);
    }
  }
  
  const boundaryEdges: Array<[number, number]> = [];
  for (const [key, count] of edgeCount.entries()) {
    if (count === 1) {
      const [a, b] = key.split('-').map(Number);
      boundaryEdges.push([a, b]);
    }
  }
  
  // Compute area
  let totalArea = 0;
  for (const [i, j, k] of validTriangles) {
    const p1 = points[i];
    const p2 = points[j];
    const p3 = points[k];
    
    const area = Math.abs(
      (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
    ) / 2;
    
    totalArea += area;
  }
  
  // Compute boundary length
  let boundaryLength = 0;
  for (const [i, j] of boundaryEdges) {
    boundaryLength += distance(points[i], points[j]);
  }
  
  // Detect holes (simplified)
  const holes = detectHoles(points, boundaryEdges);
  
  return {
    vertices: points,
    edges: boundaryEdges,
    triangles: validTriangles,
    alpha,
    boundaryLength,
    area: totalArea,
    holes
  };
}

/**
 * Detect holes in alpha shape (simplified implementation)
 */
function detectHoles(points: Point[], boundaryEdges: Array<[number, number]>): Array<Point[]> {
  // This is a simplified hole detection
  // A full implementation would use more sophisticated topological analysis
  
  const holes: Array<Point[]> = [];
  
  // Build adjacency list for boundary
  const adjacency = new Map<number, number[]>();
  
  for (const [i, j] of boundaryEdges) {
    if (!adjacency.has(i)) adjacency.set(i, []);
    if (!adjacency.has(j)) adjacency.set(j, []);
    adjacency.get(i)!.push(j);
    adjacency.get(j)!.push(i);
  }
  
  // Find cycles in the boundary (potential holes)
  const visited = new Set<number>();
  
  for (const [startVertex] of adjacency.entries()) {
    if (visited.has(startVertex)) continue;
    
    const cycle: number[] = [];
    const stack = [startVertex];
    
    while (stack.length > 0) {
      const vertex = stack.pop()!;
      if (visited.has(vertex)) continue;
      
      visited.add(vertex);
      cycle.push(vertex);
      
      const neighbors = adjacency.get(vertex) || [];
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          stack.push(neighbor);
        }
      }
    }
    
    // If cycle is small and forms a closed loop, it might be a hole
    if (cycle.length >= 3 && cycle.length <= points.length / 3) {
      const holePoints = cycle.map(index => points[index]);
      holes.push(holePoints);
    }
  }
  
  return holes;
}

/**
 * Find optimal alpha value using heuristics
 */
export function findOptimalAlpha(points: Point[]): number {
  if (points.length < 3) return 0;
  
  const alphaComplex = computeAlphaComplex(points);
  const alphas = alphaComplex.criticalAlphas;
  
  if (alphas.length === 0) return 0;
  
  // Use median alpha as a reasonable default
  const medianIndex = Math.floor(alphas.length / 2);
  return alphas[medianIndex];
}

/**
 * Compute alpha shape features for machine learning
 */
export function extractAlphaShapeFeatures(alphaShape: AlphaShapeResult): number[] {
  const features: number[] = [];
  
  // Basic shape features
  features.push(alphaShape.area);
  features.push(alphaShape.boundaryLength);
  features.push(alphaShape.vertices.length);
  features.push(alphaShape.edges.length);
  features.push(alphaShape.triangles.length);
  
  // Shape complexity
  const complexity = alphaShape.boundaryLength > 0 ? 
    alphaShape.area / (alphaShape.boundaryLength * alphaShape.boundaryLength) : 0;
  features.push(complexity);
  
  // Hole characteristics
  features.push(alphaShape.holes.length);
  const totalHoleArea = alphaShape.holes.reduce((sum, hole) => {
    // Simplified hole area calculation
    return sum + computePolygonArea(hole);
  }, 0);
  features.push(totalHoleArea);
  
  // Shape ratios
  features.push(alphaShape.area > 0 ? totalHoleArea / alphaShape.area : 0);
  
  // Alpha value
  features.push(alphaShape.alpha);
  
  return features;
}

/**
 * Compute polygon area using shoelace formula
 */
function computePolygonArea(polygon: Point[]): number {
  if (polygon.length < 3) return 0;
  
  let area = 0;
  const n = polygon.length;
  
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += polygon[i].x * polygon[j].y;
    area -= polygon[j].x * polygon[i].y;
  }
  
  return Math.abs(area) / 2;
}

/**
 * Compare alpha shapes for similarity
 */
export function compareAlphaShapes(shape1: AlphaShapeResult, shape2: AlphaShapeResult): number {
  const features1 = extractAlphaShapeFeatures(shape1);
  const features2 = extractAlphaShapeFeatures(shape2);
  
  // Normalize features
  const maxValues = features1.map((f1, i) => Math.max(Math.abs(f1), Math.abs(features2[i])));
  
  let similarity = 0;
  let validFeatures = 0;
  
  for (let i = 0; i < features1.length; i++) {
    if (maxValues[i] > 1e-10) {
      const normalizedDiff = Math.abs(features1[i] - features2[i]) / maxValues[i];
      similarity += Math.exp(-normalizedDiff);
      validFeatures++;
    }
  }
  
  return validFeatures > 0 ? similarity / validFeatures : 0;
}
