import React, { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent } from '@/components/ui/card';
import { TrendingUp, Zap, Target, Activity, BarChart3 } from 'lucide-react';

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

interface StrokeAnalysisProps {
  strokeData: Stroke[];
  onAnalysisUpdate: (analysis: any) => void;
}

interface AnalysisResult {
  totalPoints: number;
  totalStrokes: number;
  directionSequences: string[][];
  peakValleySequences: { x: string[]; y: string[] }[];
  intersectionCounts: {
    horizontal: number[];
    vertical: number[];
    positiveDiagonal: number[];
    negativeDiagonal: number[];
  };
  tangentAngles: number[][];
  curvatureValues: number[][];
  velocityProfiles: number[][];
  accelerationProfiles: number[][];
  fourierCoefficients: { magnitude: number[]; phase: number[] }[];
  complexityScore: number;
  regularityScore: number;
  symmetryScore: number;
  arcLengths: number[];
}

const StrokeAnalysis: React.FC<StrokeAnalysisProps> = ({ strokeData, onAnalysisUpdate }) => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Normalize points to 0-1 range
  const normalizePoints = (points: Point[]) => {
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
  };

  // Calculate direction sequence for a stroke
  const calculateDirectionSequence = (points: Point[]): string[] => {
    const directions: string[] = [];
    
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      
      const angle = Math.atan2(dy, dx);
      const degrees = (angle * 180 / Math.PI + 360) % 360;
      
      if (degrees >= 315 || degrees < 45) directions.push('→');
      else if (degrees >= 45 && degrees < 135) directions.push('↓');
      else if (degrees >= 135 && degrees < 225) directions.push('←');
      else directions.push('↑');
    }
    
    return directions;
  };

  // Calculate peak-valley sequence
  const calculatePeakValleySequence = (values: number[]): string[] => {
    const sequence: string[] = [];
    
    for (let i = 1; i < values.length - 1; i++) {
      const prev = values[i - 1];
      const curr = values[i];
      const next = values[i + 1];
      
      if (curr > prev && curr > next) {
        sequence.push('P'); // Peak
      } else if (curr < prev && curr < next) {
        sequence.push('V'); // Valley
      } else {
        sequence.push('-'); // Neither
      }
    }
    
    return sequence;
  };

  // Calculate intersection counts with reference lines
  const calculateIntersectionCounts = (points: Point[]) => {
    const counts = {
      horizontal: 0,
      vertical: 0,
      positiveDiagonal: 0,
      negativeDiagonal: 0
    };

    // Create reference lines at different positions
    const referenceLines = {
      horizontal: [0.25, 0.5, 0.75],
      vertical: [0.25, 0.5, 0.75],
      positiveDiagonal: [-0.5, 0, 0.5], // y = x + c
      negativeDiagonal: [0.5, 1, 1.5]   // y = -x + c
    };

    for (let i = 1; i < points.length; i++) {
      const p1 = points[i - 1];
      const p2 = points[i];

      // Check horizontal line intersections
      referenceLines.horizontal.forEach(y => {
        if ((p1.y <= y && p2.y >= y) || (p1.y >= y && p2.y <= y)) {
          counts.horizontal++;
        }
      });

      // Check vertical line intersections
      referenceLines.vertical.forEach(x => {
        if ((p1.x <= x && p2.x >= x) || (p1.x >= x && p2.x <= x)) {
          counts.vertical++;
        }
      });

      // Check diagonal intersections (simplified)
      referenceLines.positiveDiagonal.forEach(c => {
        const y1Expected = p1.x + c;
        const y2Expected = p2.x + c;
        if ((p1.y <= y1Expected && p2.y >= y2Expected) || (p1.y >= y1Expected && p2.y <= y2Expected)) {
          counts.positiveDiagonal++;
        }
      });

      referenceLines.negativeDiagonal.forEach(c => {
        const y1Expected = -p1.x + c;
        const y2Expected = -p2.x + c;
        if ((p1.y <= y1Expected && p2.y >= y2Expected) || (p1.y >= y1Expected && p2.y <= y2Expected)) {
          counts.negativeDiagonal++;
        }
      });
    }

    return counts;
  };

  // Calculate tangent angles
  const calculateTangentAngles = (points: Point[]): number[] => {
    const angles: number[] = [];
    
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      const angle = Math.atan2(dy, dx);
      angles.push(angle);
    }
    
    return angles;
  };

  // Calculate curvature
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
      
      const curvature = v1Mag > 0 && v2Mag > 0 ? crossProduct / (v1Mag * v2Mag) : 0;
      curvatures.push(curvature);
    }
    
    return curvatures;
  };

  // Calculate velocity profile
  const calculateVelocity = (points: Point[]): number[] => {
    const velocities: number[] = [];
    
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      const dt = Math.max(points[i].time - points[i - 1].time, 1);
      
      const velocity = Math.sqrt(dx * dx + dy * dy) / dt;
      velocities.push(velocity);
    }
    
    return velocities;
  };

  // Calculate acceleration profile
  const calculateAcceleration = (velocities: number[], points: Point[]): number[] => {
    const accelerations: number[] = [];
    
    for (let i = 1; i < velocities.length; i++) {
      const dv = velocities[i] - velocities[i - 1];
      const dt = Math.max(points[i + 1].time - points[i].time, 1);
      
      const acceleration = dv / dt;
      accelerations.push(acceleration);
    }
    
    return accelerations;
  };

  // Simple FFT for tangent angles (magnitude only)
  const calculateFourierCoefficients = (angles: number[]) => {
    const n = Math.min(angles.length, 32); // Limit for performance
    const magnitudes: number[] = [];
    const phases: number[] = [];
    
    for (let k = 0; k < n / 2; k++) {
      let real = 0;
      let imag = 0;
      
      for (let j = 0; j < n; j++) {
        const angle = -2 * Math.PI * k * j / n;
        real += angles[j] * Math.cos(angle);
        imag += angles[j] * Math.sin(angle);
      }
      
      const magnitude = Math.sqrt(real * real + imag * imag) / n;
      const phase = Math.atan2(imag, real);
      
      magnitudes.push(magnitude);
      phases.push(phase);
    }
    
    return { magnitude: magnitudes, phase: phases };
  };

  // Calculate complexity score
  const calculateComplexityScore = (stroke: Stroke): number => {
    const normalizedPoints = normalizePoints(stroke.points);
    const curvatures = calculateCurvature(normalizedPoints);
    const tangentAngles = calculateTangentAngles(normalizedPoints);
    
    // Complexity based on curvature variation and direction changes
    const curvatureVariation = curvatures.length > 0 ? 
      Math.sqrt(curvatures.reduce((sum, c) => sum + c * c, 0) / curvatures.length) : 0;
    
    const angleChanges = tangentAngles.length > 1 ?
      tangentAngles.slice(1).reduce((sum, angle, i) => {
        const change = Math.abs(angle - tangentAngles[i]);
        return sum + Math.min(change, 2 * Math.PI - change);
      }, 0) / (tangentAngles.length - 1) : 0;
    
    return Math.min(curvatureVariation * 10 + angleChanges, 10);
  };

  // Calculate regularity score
  const calculateRegularityScore = (stroke: Stroke): number => {
    const velocities = calculateVelocity(stroke.points);
    if (velocities.length === 0) return 0;
    
    const meanVelocity = velocities.reduce((sum, v) => sum + v, 0) / velocities.length;
    const velocityVariance = velocities.reduce((sum, v) => sum + Math.pow(v - meanVelocity, 2), 0) / velocities.length;
    const coefficientOfVariation = meanVelocity > 0 ? Math.sqrt(velocityVariance) / meanVelocity : 1;
    
    return Math.max(0, 1 - coefficientOfVariation);
  };

  // Calculate symmetry score (simplified)
  const calculateSymmetryScore = (strokes: Stroke[]): number => {
    if (strokes.length === 0) return 0;
    
    // Combine all points
    const allPoints = strokes.flatMap(stroke => normalizePoints(stroke.points));
    if (allPoints.length < 4) return 0;
    
    // Check horizontal symmetry around center
    const centerX = 0.5;
    let symmetrySum = 0;
    let count = 0;
    
    allPoints.forEach(point => {
      const mirroredX = 2 * centerX - point.x;
      const minDistance = Math.min(...allPoints.map(p => 
        Math.sqrt((p.x - mirroredX) ** 2 + (p.y - point.y) ** 2)
      ));
      symmetrySum += Math.exp(-minDistance * 10); // Exponential decay
      count++;
    });
    
    return count > 0 ? symmetrySum / count : 0;
  };

  // Calculate arc length
  const calculateArcLength = (points: Point[]): number => {
    let length = 0;
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      length += Math.sqrt(dx * dx + dy * dy);
    }
    return length;
  };

  // Main analysis function
  const analyzeStrokes = async (strokes: Stroke[]) => {
    if (strokes.length === 0) {
      setAnalysisResult(null);
      onAnalysisUpdate(null);
      return;
    }

    setIsAnalyzing(true);

    // Simulate processing delay for complex calculations
    await new Promise(resolve => setTimeout(resolve, 100));

    const results: AnalysisResult = {
      totalPoints: strokes.reduce((sum, stroke) => sum + stroke.points.length, 0),
      totalStrokes: strokes.length,
      directionSequences: [],
      peakValleySequences: { x: [], y: [] },
      intersectionCounts: {
        horizontal: [],
        vertical: [],
        positiveDiagonal: [],
        negativeDiagonal: []
      },
      tangentAngles: [],
      curvatureValues: [],
      velocityProfiles: [],
      accelerationProfiles: [],
      fourierCoefficients: [],
      complexityScore: 0,
      regularityScore: 0,
      symmetryScore: 0,
      arcLengths: []
    };

    // Analyze each stroke
    strokes.forEach(stroke => {
      const normalizedPoints = normalizePoints(stroke.points);
      
      // Direction sequences
      results.directionSequences.push(calculateDirectionSequence(normalizedPoints));
      
      // Peak-valley sequences
      const xValues = normalizedPoints.map(p => p.x);
      const yValues = normalizedPoints.map(p => p.y);
      results.peakValleySequences.x.push(calculatePeakValleySequence(xValues));
      results.peakValleySequences.y.push(calculatePeakValleySequence(yValues));
      
      // Intersection counts
      const intersections = calculateIntersectionCounts(normalizedPoints);
      results.intersectionCounts.horizontal.push(intersections.horizontal);
      results.intersectionCounts.vertical.push(intersections.vertical);
      results.intersectionCounts.positiveDiagonal.push(intersections.positiveDiagonal);
      results.intersectionCounts.negativeDiagonal.push(intersections.negativeDiagonal);
      
      // Tangent angles and curvature
      const tangentAngles = calculateTangentAngles(normalizedPoints);
      const curvatureValues = calculateCurvature(normalizedPoints);
      results.tangentAngles.push(tangentAngles);
      results.curvatureValues.push(curvatureValues);
      
      // Velocity and acceleration
      const velocities = calculateVelocity(normalizedPoints);
      const accelerations = calculateAcceleration(velocities, normalizedPoints);
      results.velocityProfiles.push(velocities);
      results.accelerationProfiles.push(accelerations);
      
      // Fourier coefficients
      if (tangentAngles.length > 0) {
        results.fourierCoefficients.push(calculateFourierCoefficients(tangentAngles));
      }
      
      // Arc length
      results.arcLengths.push(calculateArcLength(normalizedPoints));
    });

    // Calculate overall scores
    results.complexityScore = strokes.reduce((sum, stroke) => sum + calculateComplexityScore(stroke), 0) / strokes.length;
    results.regularityScore = strokes.reduce((sum, stroke) => sum + calculateRegularityScore(stroke), 0) / strokes.length;
    results.symmetryScore = calculateSymmetryScore(strokes);

    setAnalysisResult(results);
    onAnalysisUpdate(results);
    setIsAnalyzing(false);
  };

  // Run analysis when stroke data changes
  useEffect(() => {
    analyzeStrokes(strokeData);
  }, [strokeData]);

  if (!analysisResult) {
    return (
      <div className="text-center text-muted-foreground py-8">
        <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>Draw a letter to start analysis</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 text-white">
      {isAnalyzing && (
        <div className="flex items-center justify-center py-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
          <span className="ml-3 text-blue-200">Analyzing stroke patterns...</span>
        </div>
      )}

      {/* Overview Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <Card className="bg-blue-500/20 border-blue-400/30">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-blue-400" />
              <div>
                <div className="text-sm text-blue-200">Complexity</div>
                <div className="text-xl font-bold text-blue-100">
                  {analysisResult.complexityScore.toFixed(2)}
                </div>
                <Progress value={analysisResult.complexityScore * 10} className="mt-2 h-2" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-green-500/20 border-green-400/30">
          <CardContent className="p-4">
            <div className="flex items-center space-x-2">
              <Target className="w-5 h-5 text-green-400" />
              <div>
                <div className="text-sm text-green-200">Regularity</div>
                <div className="text-xl font-bold text-green-100">
                  {(analysisResult.regularityScore * 100).toFixed(0)}%
                </div>
                <Progress value={analysisResult.regularityScore * 100} className="mt-2 h-2" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Direction Analysis */}
      <div className="space-y-3">
        <h4 className="font-medium text-white flex items-center">
          <Zap className="w-4 h-4 mr-2" />
          Direction Sequences
        </h4>
        {analysisResult.directionSequences.map((sequence, index) => (
          <div key={index} className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="text-sm text-gray-300 mb-2">Stroke {index + 1}</div>
            <div className="font-mono text-lg text-white">
              {sequence.slice(0, 20).join(' ')}
              {sequence.length > 20 && <span className="text-gray-400"> ... (+{sequence.length - 20})</span>}
            </div>
          </div>
        ))}
      </div>

      {/* Intersection Analysis */}
      <div className="space-y-3">
        <h4 className="font-medium text-white flex items-center">
          <BarChart3 className="w-4 h-4 mr-2" />
          Intersection Counts
        </h4>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="text-sm text-gray-300">Horizontal</div>
            <div className="text-xl font-bold text-blue-400">
              {analysisResult.intersectionCounts.horizontal.reduce((sum, count) => sum + count, 0)}
            </div>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="text-sm text-gray-300">Vertical</div>
            <div className="text-xl font-bold text-green-400">
              {analysisResult.intersectionCounts.vertical.reduce((sum, count) => sum + count, 0)}
            </div>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="text-sm text-gray-300">Diagonal ↗</div>
            <div className="text-xl font-bold text-purple-400">
              {analysisResult.intersectionCounts.positiveDiagonal.reduce((sum, count) => sum + count, 0)}
            </div>
          </div>
          <div className="bg-white/5 rounded-lg p-3 border border-white/10">
            <div className="text-sm text-gray-300">Diagonal ↖</div>
            <div className="text-xl font-bold text-orange-400">
              {analysisResult.intersectionCounts.negativeDiagonal.reduce((sum, count) => sum + count, 0)}
            </div>
          </div>
        </div>
      </div>

      {/* Feature Summary */}
      <div className="grid grid-cols-3 gap-3 text-sm">
        <Badge variant="outline" className="bg-blue-500/20 text-blue-200 border-blue-400/30 justify-center py-2">
          Arc Length: {analysisResult.arcLengths.reduce((sum, length) => sum + length, 0).toFixed(2)}
        </Badge>
        <Badge variant="outline" className="bg-purple-500/20 text-purple-200 border-purple-400/30 justify-center py-2">
          Symmetry: {(analysisResult.symmetryScore * 100).toFixed(0)}%
        </Badge>
        <Badge variant="outline" className="bg-green-500/20 text-green-200 border-green-400/30 justify-center py-2">
          Fourier: {analysisResult.fourierCoefficients.length} sets
        </Badge>
      </div>
    </div>
  );
};

export default StrokeAnalysis;
