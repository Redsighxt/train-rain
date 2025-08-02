import React, { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Crosshair, Star, Diamond, Circle, Triangle } from 'lucide-react';

interface Point {
  x: number;
  y: number;
  time: number;
  pressure?: number;
}

interface Stroke {
  id: string;
  points: Point[];
  completed: boolean;
  color: string;
}

interface InvariantPoint {
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

interface InvariantPointsProps {
  strokeData: Stroke[];
  analysisData: any;
  onInvariantPointsUpdate: (points: InvariantPoint[]) => void;
}

const InvariantPoints: React.FC<InvariantPointsProps> = ({
  strokeData,
  analysisData,
  onInvariantPointsUpdate
}) => {
  const [invariantPoints, setInvariantPoints] = useState<InvariantPoint[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);

  // Normalize points to 0-1 range
  const normalizePoints = (points: Point[]) => {
    if (points.length === 0) return [];
    
    const allPoints = strokeData.flatMap(stroke => stroke.points);
    const xs = allPoints.map(p => p.x);
    const ys = allPoints.map(p => p.y);
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
  };

  // Calculate curvature at each point
  const calculateCurvature = (points: Point[]): number[] => {
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
      
      const curvature = v1Mag > 0 && v2Mag > 0 ? Math.abs(crossProduct) / (v1Mag * v2Mag) : 0;
      curvatures.push(curvature);
    }
    
    return curvatures;
  };

  // Detect geometric invariant points (high curvature, inflection points, extrema)
  const detectGeometricInvariants = (stroke: Stroke, strokeIndex: number): InvariantPoint[] => {
    const points = normalizePoints(stroke.points);
    const invariants: InvariantPoint[] = [];
    
    if (points.length < 3) return invariants;
    
    const curvatures = calculateCurvature(points);
    
    // Find high curvature points (corners, sharp turns)
    const curvatureThreshold = 0.5;
    curvatures.forEach((curvature, i) => {
      if (curvature > curvatureThreshold) {
        const actualIndex = i + 1; // Offset due to curvature calculation
        invariants.push({
          id: `geom-curve-${strokeIndex}-${actualIndex}`,
          index: actualIndex,
          strokeIndex,
          position: points[actualIndex],
          stabilityScore: Math.min(curvature, 1),
          type: curvature > 0.8 ? 'primary' : 'secondary',
          category: 'geometric',
          description: `High curvature point (${curvature.toFixed(2)})`,
          confidence: Math.min(curvature, 1)
        });
      }
    });

    // Find extrema points (highest/lowest X and Y)
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    
    const minXIndex = xs.indexOf(Math.min(...xs));
    const maxXIndex = xs.indexOf(Math.max(...xs));
    const minYIndex = ys.indexOf(Math.min(...ys));
    const maxYIndex = ys.indexOf(Math.max(...ys));
    
    [
      { index: minXIndex, type: 'Leftmost point' },
      { index: maxXIndex, type: 'Rightmost point' },
      { index: minYIndex, type: 'Topmost point' },
      { index: maxYIndex, type: 'Bottommost point' }
    ].forEach(({ index, type }) => {
      if (index >= 0 && index < points.length) {
        invariants.push({
          id: `geom-extrema-${strokeIndex}-${index}`,
          index,
          strokeIndex,
          position: points[index],
          stabilityScore: 0.8,
          type: 'secondary',
          category: 'geometric',
          description: type,
          confidence: 0.8
        });
      }
    });

    // Find inflection points (where curvature changes sign)
    for (let i = 1; i < curvatures.length - 1; i++) {
      const prev = curvatures[i - 1];
      const curr = curvatures[i];
      const next = curvatures[i + 1];
      
      // Check for significant curvature change
      if (Math.abs(curr - prev) > 0.3 && Math.abs(next - curr) > 0.3) {
        const actualIndex = i + 1;
        invariants.push({
          id: `geom-inflection-${strokeIndex}-${actualIndex}`,
          index: actualIndex,
          strokeIndex,
          position: points[actualIndex],
          stabilityScore: 0.6,
          type: 'tertiary',
          category: 'geometric',
          description: 'Curvature inflection point',
          confidence: 0.6
        });
      }
    }

    return invariants;
  };

  // Detect topological invariant points (stroke endpoints, intersections)
  const detectTopologicalInvariants = (stroke: Stroke, strokeIndex: number): InvariantPoint[] => {
    const points = normalizePoints(stroke.points);
    const invariants: InvariantPoint[] = [];
    
    if (points.length < 2) return invariants;

    // Stroke endpoints are highly invariant
    invariants.push({
      id: `topo-start-${strokeIndex}`,
      index: 0,
      strokeIndex,
      position: points[0],
      stabilityScore: 0.95,
      type: 'primary',
      category: 'topological',
      description: 'Stroke start point',
      confidence: 0.95
    });

    invariants.push({
      id: `topo-end-${strokeIndex}`,
      index: points.length - 1,
      strokeIndex,
      position: points[points.length - 1],
      stabilityScore: 0.95,
      type: 'primary',
      category: 'topological',
      description: 'Stroke end point',
      confidence: 0.95
    });

    // Find potential self-intersection points (simplified)
    for (let i = 0; i < points.length - 2; i++) {
      for (let j = i + 2; j < points.length - 1; j++) {
        const p1 = points[i];
        const p2 = points[i + 1];
        const p3 = points[j];
        const p4 = points[j + 1];
        
        // Check if line segments are close (simplified intersection test)
        const dist = distancePointToLineSegment(p1, p3, p4);
        const dist2 = distancePointToLineSegment(p2, p3, p4);
        
        if (dist < 0.05 && dist2 < 0.05) {
          const midIndex = Math.floor((i + j) / 2);
          invariants.push({
            id: `topo-intersection-${strokeIndex}-${midIndex}`,
            index: midIndex,
            strokeIndex,
            position: points[midIndex],
            stabilityScore: 0.7,
            type: 'secondary',
            category: 'topological',
            description: 'Potential self-intersection',
            confidence: 0.7
          });
        }
      }
    }

    return invariants;
  };

  // Helper function: distance from point to line segment
  const distancePointToLineSegment = (point: Point, lineStart: Point, lineEnd: Point): number => {
    const A = point.x - lineStart.x;
    const B = point.y - lineStart.y;
    const C = lineEnd.x - lineStart.x;
    const D = lineEnd.y - lineStart.y;

    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    
    if (lenSq === 0) return Math.sqrt(A * A + B * B);

    const param = Math.max(0, Math.min(1, dot / lenSq));
    const projX = lineStart.x + param * C;
    const projY = lineStart.y + param * D;

    const dx = point.x - projX;
    const dy = point.y - projY;
    
    return Math.sqrt(dx * dx + dy * dy);
  };

  // Detect statistical invariant points (based on stability across multiple instances)
  const detectStatisticalInvariants = (stroke: Stroke, strokeIndex: number): InvariantPoint[] => {
    const points = normalizePoints(stroke.points);
    const invariants: InvariantPoint[] = [];
    
    // For now, simulate statistical stability based on stroke characteristics
    // In a real implementation, this would use multiple instances of the same letter
    
    const quarterPoints = [
      Math.floor(points.length * 0.25),
      Math.floor(points.length * 0.5),
      Math.floor(points.length * 0.75)
    ];
    
    quarterPoints.forEach((index, i) => {
      if (index < points.length) {
        invariants.push({
          id: `stat-quarter-${strokeIndex}-${index}`,
          index,
          strokeIndex,
          position: points[index],
          stabilityScore: 0.6 - i * 0.1, // Decreasing stability
          type: i === 1 ? 'secondary' : 'tertiary',
          category: 'statistical',
          description: `${['Quarter', 'Mid', 'Three-quarter'][i]} point`,
          confidence: 0.6 - i * 0.1
        });
      }
    });

    return invariants;
  };

  // Detect intersection-based invariant points
  const detectIntersectionInvariants = (stroke: Stroke, strokeIndex: number): InvariantPoint[] => {
    const points = normalizePoints(stroke.points);
    const invariants: InvariantPoint[] = [];
    
    // Points that cross reference lines (grid intersections)
    const referenceLines = {
      horizontal: [0.25, 0.5, 0.75],
      vertical: [0.25, 0.5, 0.75]
    };

    for (let i = 1; i < points.length; i++) {
      const p1 = points[i - 1];
      const p2 = points[i];

      // Check horizontal line crossings
      referenceLines.horizontal.forEach((y, lineIndex) => {
        if ((p1.y <= y && p2.y >= y) || (p1.y >= y && p2.y <= y)) {
          // Find intersection point
          const t = (y - p1.y) / (p2.y - p1.y);
          const intersectionX = p1.x + t * (p2.x - p1.x);
          
          invariants.push({
            id: `intersect-h-${strokeIndex}-${i}-${lineIndex}`,
            index: i,
            strokeIndex,
            position: { x: intersectionX, y, time: p1.time },
            stabilityScore: 0.5,
            type: 'tertiary',
            category: 'intersection',
            description: `Horizontal line crossing at y=${y.toFixed(2)}`,
            confidence: 0.5
          });
        }
      });

      // Check vertical line crossings
      referenceLines.vertical.forEach((x, lineIndex) => {
        if ((p1.x <= x && p2.x >= x) || (p1.x >= x && p2.x <= x)) {
          // Find intersection point
          const t = (x - p1.x) / (p2.x - p1.x);
          const intersectionY = p1.y + t * (p2.y - p1.y);
          
          invariants.push({
            id: `intersect-v-${strokeIndex}-${i}-${lineIndex}`,
            index: i,
            strokeIndex,
            position: { x, y: intersectionY, time: p1.time },
            stabilityScore: 0.5,
            type: 'tertiary',
            category: 'intersection',
            description: `Vertical line crossing at x=${x.toFixed(2)}`,
            confidence: 0.5
          });
        }
      });
    }

    return invariants;
  };

  // Main detection function
  const detectInvariantPoints = async () => {
    if (strokeData.length === 0) {
      setInvariantPoints([]);
      onInvariantPointsUpdate([]);
      return;
    }

    setIsProcessing(true);
    
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 200));

    const allInvariants: InvariantPoint[] = [];

    strokeData.forEach((stroke, strokeIndex) => {
      // Detect different types of invariant points
      const geometric = detectGeometricInvariants(stroke, strokeIndex);
      const topological = detectTopologicalInvariants(stroke, strokeIndex);
      const statistical = detectStatisticalInvariants(stroke, strokeIndex);
      const intersection = detectIntersectionInvariants(stroke, strokeIndex);
      
      allInvariants.push(...geometric, ...topological, ...statistical, ...intersection);
    });

    // Remove duplicates and sort by stability score
    const uniqueInvariants = allInvariants
      .filter((point, index, array) => 
        array.findIndex(p => 
          p.strokeIndex === point.strokeIndex && 
          Math.abs(p.index - point.index) < 2
        ) === index
      )
      .sort((a, b) => b.stabilityScore - a.stabilityScore)
      .slice(0, 20); // Limit to top 20 invariant points

    setInvariantPoints(uniqueInvariants);
    onInvariantPointsUpdate(uniqueInvariants);
    setIsProcessing(false);
  };

  // Run detection when stroke data or analysis data changes
  useEffect(() => {
    detectInvariantPoints();
  }, [strokeData, analysisData]);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'primary': return <Star className="w-4 h-4" />;
      case 'secondary': return <Diamond className="w-4 h-4" />;
      case 'tertiary': return <Circle className="w-4 h-4" />;
      default: return <Crosshair className="w-4 h-4" />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'geometric': return 'bg-blue-500/20 text-blue-200 border-blue-400/30';
      case 'topological': return 'bg-green-500/20 text-green-200 border-green-400/30';
      case 'statistical': return 'bg-purple-500/20 text-purple-200 border-purple-400/30';
      case 'intersection': return 'bg-orange-500/20 text-orange-200 border-orange-400/30';
      default: return 'bg-gray-500/20 text-gray-200 border-gray-400/30';
    }
  };

  if (strokeData.length === 0) {
    return (
      <div className="text-center text-muted-foreground py-8">
        <Crosshair className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>Draw a letter to detect invariant points</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 text-white">
      {isProcessing && (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
          <span className="ml-3 text-blue-200">Detecting invariant points...</span>
        </div>
      )}

      {/* Summary */}
      <div className="grid grid-cols-3 gap-4">
        {['primary', 'secondary', 'tertiary'].map(type => {
          const count = invariantPoints.filter(p => p.type === type).length;
          return (
            <Card key={type} className="bg-white/5 border-white/10">
              <CardContent className="p-4 text-center">
                <div className="flex items-center justify-center mb-2">
                  {getTypeIcon(type)}
                </div>
                <div className="text-2xl font-bold">{count}</div>
                <div className="text-sm text-gray-300 capitalize">{type}</div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Category Distribution */}
      <div className="space-y-3">
        <h4 className="font-medium text-white">Categories</h4>
        <div className="grid grid-cols-2 gap-3">
          {['geometric', 'topological', 'statistical', 'intersection'].map(category => {
            const count = invariantPoints.filter(p => p.category === category).length;
            const percentage = invariantPoints.length > 0 ? (count / invariantPoints.length) * 100 : 0;
            
            return (
              <div key={category} className="bg-white/5 rounded-lg p-3 border border-white/10">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300 capitalize">{category}</span>
                  <span className="text-sm font-bold">{count}</span>
                </div>
                <Progress value={percentage} className="h-2" />
              </div>
            );
          })}
        </div>
      </div>

      {/* Invariant Points List */}
      <div className="space-y-3">
        <h4 className="font-medium text-white">Detected Points</h4>
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {invariantPoints.slice(0, 10).map((point) => (
            <div key={point.id} className="bg-white/5 rounded-lg p-3 border border-white/10">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-2">
                    {getTypeIcon(point.type)}
                    <Badge variant="outline" className={getCategoryColor(point.category)}>
                      {point.category}
                    </Badge>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-white">{point.description}</div>
                    <div className="text-xs text-gray-400">
                      Stroke {point.strokeIndex + 1}, Point {point.index} • 
                      Position: ({point.position.x.toFixed(3)}, {point.position.y.toFixed(3)})
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm font-bold text-blue-400">
                    {(point.stabilityScore * 100).toFixed(0)}%
                  </div>
                  <div className="text-xs text-gray-400">stability</div>
                </div>
              </div>
            </div>
          ))}
          
          {invariantPoints.length > 10 && (
            <div className="text-center text-gray-400 text-sm py-2">
              ... and {invariantPoints.length - 10} more points
            </div>
          )}
        </div>
      </div>

      {invariantPoints.length === 0 && !isProcessing && (
        <div className="text-center text-gray-400 py-8">
          <Crosshair className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No invariant points detected</p>
          <p className="text-sm mt-2">Try drawing a more complex letter</p>
        </div>
      )}
    </div>
  );
};

export default InvariantPoints;
