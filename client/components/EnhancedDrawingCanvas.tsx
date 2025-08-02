import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useResearchStore } from '@/store/researchStore';
import { analyzeRealTime, analyzeStroke } from '@/lib/strokeAnalysisProcessor';
import { cn } from '@/lib/utils';
import type { Point, Stroke } from '@/store/researchStore';

interface EnhancedDrawingCanvasProps {
  className?: string;
}

const EnhancedDrawingCanvas: React.FC<EnhancedDrawingCanvasProps> = ({ className }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentStroke, setCurrentStroke] = useState<Point[]>([]);
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 });
  const [strokeCount, setStrokeCount] = useState(0);
  
  const {
    setCurrentStroke: setStoreStroke,
    clearCurrentStroke,
    setProcessingStage,
    settings,
    currentStroke: storeStroke
  } = useResearchStore();

  const strokeColors = [
    '#3B82F6', '#EF4444', '#10B981', '#F59E0B', 
    '#8B5CF6', '#EC4899', '#06B6D4', '#84CC16'
  ];

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size with high DPI
    const container = canvas.parentElement;
    if (container) {
      const rect = container.getBoundingClientRect();
      const width = Math.min(600, rect.width - 32);
      const height = 400;
      
      canvas.width = width * 2;
      canvas.height = height * 2;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      
      ctx.scale(2, 2);
      setCanvasSize({ width, height });
    }

    // Set drawing style
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 3;

    // Clear canvas with dark background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw research grid
    drawResearchGrid(ctx, canvasSize.width, canvasSize.height);
  }, []);

  const drawResearchGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    // Primary grid (every 40px)
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.1)';
    ctx.lineWidth = 1;
    
    const majorGrid = 40;
    for (let x = 0; x <= width; x += majorGrid) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    for (let y = 0; y <= height; y += majorGrid) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Secondary grid (every 10px)
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.05)';
    const minorGrid = 10;
    for (let x = 0; x <= width; x += minorGrid) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    for (let y = 0; y <= height; y += minorGrid) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Center lines (more prominent)
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
    ctx.lineWidth = 2;
    
    // Vertical center
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height);
    ctx.stroke();
    
    // Horizontal center
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Corner markers for reference
    ctx.fillStyle = 'rgba(59, 130, 246, 0.3)';
    const markerSize = 4;
    [
      [0, 0], [width, 0], [0, height], [width, height],
      [width/2, 0], [width/2, height], [0, height/2], [width, height/2]
    ].forEach(([x, y]) => {
      ctx.fillRect(x - markerSize/2, y - markerSize/2, markerSize, markerSize);
    });
  };

  const getCanvasPoint = (e: React.MouseEvent | React.TouchEvent): Point => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    
    let clientX: number, clientY: number;
    
    if (e.type.startsWith('touch')) {
      const touchEvent = e as React.TouchEvent;
      clientX = touchEvent.touches[0]?.clientX || touchEvent.changedTouches[0]?.clientX;
      clientY = touchEvent.touches[0]?.clientY || touchEvent.changedTouches[0]?.clientY;
    } else {
      const mouseEvent = e as React.MouseEvent;
      clientX = mouseEvent.clientX;
      clientY = mouseEvent.clientY;
    }
    
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width) / 2,
      y: (clientY - rect.top) * (canvas.height / rect.height) / 2,
      time: Date.now(),
      pressure: 1.0
    };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    setIsDrawing(true);
    
    const point = getCanvasPoint(e);
    setCurrentStroke([point]);
    
    // Start drawing on canvas
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const strokeColor = strokeColors[strokeCount % strokeColors.length];
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
  };

  const continueDrawing = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    
    e.preventDefault();
    const point = getCanvasPoint(e);
    
    setCurrentStroke(prev => [...prev, point]);
    
    // Draw on canvas with advanced Catmull-Rom spline interpolation
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;

    if (currentStroke.length >= 3) {
      // Use Catmull-Rom splines for ultra-smooth rendering
      drawCatmullRomSpline(ctx, [...currentStroke, point]);
    } else if (currentStroke.length > 0) {
      // Fallback to quadratic curves for initial points
      const lastPoint = currentStroke[currentStroke.length - 1];
      const cpx = (lastPoint.x + point.x) / 2;
      const cpy = (lastPoint.y + point.y) / 2;
      ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, cpx, cpy);
      ctx.stroke();
    }
    
    // Real-time analysis for immediate feedback
    if (settings.enableAffineAnalysis && currentStroke.length > 10) {
      performRealTimeAnalysis();
    }
  }, [isDrawing, currentStroke, settings.enableAffineAnalysis]);

  const stopDrawing = async () => {
    if (!isDrawing || currentStroke.length < 2) {
      setIsDrawing(false);
      return;
    }
    
    setIsDrawing(false);
    
    // Create stroke object
    const stroke: Stroke = {
      id: `stroke-${Date.now()}`,
      points: currentStroke,
      completed: true,
      color: strokeColors[strokeCount % strokeColors.length]
    };
    
    setStrokeCount(prev => prev + 1);
    setCurrentStroke([]);
    
    // Trigger comprehensive analysis
    await performFullAnalysis(stroke);
  };

  const performRealTimeAnalysis = useCallback(async () => {
    if (currentStroke.length < 5) return;
    
    setProcessingStage('Real-time analysis...');
    
    try {
      const tempStroke: Stroke = {
        id: 'temp',
        points: currentStroke,
        completed: false,
        color: strokeColors[strokeCount % strokeColors.length]
      };
      
      // Quick analysis for real-time feedback
      await analyzeRealTime(tempStroke, 30); // 30ms max for real-time
    } catch (error) {
      console.error('Real-time analysis error:', error);
    }
  }, [currentStroke, strokeCount, setProcessingStage]);

  const performFullAnalysis = async (stroke: Stroke) => {
    setProcessingStage('Starting comprehensive analysis...');
    
    try {
      // Set the stroke in store first
      setStoreStroke(stroke);
      
      // Run full analysis
      const result = await analyzeStroke(stroke, {
        enableAffineAnalysis: settings.enableAffineAnalysis,
        enableTopologicalAnalysis: settings.enableTopologicalAnalysis,
        enablePathSignature: settings.enablePathSignature,
        enableSpectralAnalysis: settings.enableSpectralAnalysis,
        enableKnotTheory: true,
        targetProcessingTime: 200,
        adaptiveQuality: true,
        cacheResults: true,
        maxPoints: settings.numPoints
      });
      
      setProcessingStage('Analysis complete');
      
      // The analysis result will be automatically processed by the store listeners
      console.log(`Analysis completed in ${result.processingTime.toFixed(1)}ms with quality ${(result.qualityScore * 100).toFixed(0)}%`);
      
    } catch (error) {
      console.error('Analysis error:', error);
      setProcessingStage('Analysis failed');
    }
  };

  // Advanced Catmull-Rom spline drawing function
  const drawCatmullRomSpline = (ctx: CanvasRenderingContext2D, points: Point[]) => {
    if (points.length < 4) return;

    const tension = 0.5; // Adjust for smoothness (0 = sharp, 1 = very smooth)

    // Draw spline segments
    for (let i = 1; i < points.length - 2; i++) {
      const p0 = points[i - 1];
      const p1 = points[i];
      const p2 = points[i + 1];
      const p3 = points[i + 2];

      // Calculate control points for Catmull-Rom spline
      const cp1x = p1.x + (p2.x - p0.x) * tension / 6;
      const cp1y = p1.y + (p2.y - p0.y) * tension / 6;
      const cp2x = p2.x - (p3.x - p1.x) * tension / 6;
      const cp2y = p2.y - (p3.y - p1.y) * tension / 6;

      // Draw smooth curve segment
      if (i === 1) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
      }

      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
    }

    ctx.stroke();
  };

  // Enhanced stroke interpolation for real-time smoothing
  const interpolateStroke = (points: Point[], targetLength: number = 50): Point[] => {
    if (points.length < 2) return points;

    const interpolated: Point[] = [];
    const totalLength = calculatePathLength(points);
    const segmentLength = totalLength / (targetLength - 1);

    let currentLength = 0;
    let currentIndex = 0;

    interpolated.push(points[0]);

    for (let i = 1; i < targetLength - 1; i++) {
      const targetLength = i * segmentLength;

      // Find the segment containing this length
      while (currentIndex < points.length - 1) {
        const nextLength = currentLength + distance(points[currentIndex], points[currentIndex + 1]);
        if (nextLength >= targetLength) break;
        currentLength = nextLength;
        currentIndex++;
      }

      if (currentIndex < points.length - 1) {
        const segmentProgress = (targetLength - currentLength) / distance(points[currentIndex], points[currentIndex + 1]);
        const p1 = points[currentIndex];
        const p2 = points[currentIndex + 1];

        interpolated.push({
          x: p1.x + (p2.x - p1.x) * segmentProgress,
          y: p1.y + (p2.y - p1.y) * segmentProgress,
          time: p1.time + (p2.time - p1.time) * segmentProgress,
          pressure: p1.pressure + (p2.pressure - p1.pressure) * segmentProgress
        });
      }
    }

    interpolated.push(points[points.length - 1]);
    return interpolated;
  };

  const calculatePathLength = (points: Point[]): number => {
    let length = 0;
    for (let i = 1; i < points.length; i++) {
      length += distance(points[i - 1], points[i]);
    }
    return length;
  };

  const distance = (p1: Point, p2: Point): number => {
    return Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawResearchGrid(ctx, canvasSize.width, canvasSize.height);

    setCurrentStroke([]);
    setStrokeCount(0);
    clearCurrentStroke();
  };

  return (
    <div className={cn("relative", className)}>
      <canvas
        ref={canvasRef}
        className="border border-blue-400/30 rounded-lg cursor-crosshair bg-slate-900/50 backdrop-blur-sm shadow-lg shadow-blue-500/20"
        onMouseDown={startDrawing}
        onMouseMove={continueDrawing}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onTouchStart={startDrawing}
        onTouchMove={continueDrawing}
        onTouchEnd={stopDrawing}
        style={{ 
          touchAction: 'none',
          imageRendering: 'crisp-edges'
        }}
      />
      
      {/* Scientific overlay */}
      <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm rounded-lg px-3 py-2 text-sm text-white border border-blue-400/30">
        <div className="font-medium text-blue-200">Research Canvas</div>
        <div className="text-xs text-blue-300 mt-1">
          High-precision stroke capture • Multi-scale grid
        </div>
        {strokeCount > 0 && (
          <div className="text-xs text-green-300 mt-1">
            {strokeCount} stroke{strokeCount !== 1 ? 's' : ''} • {currentStroke.length} points
          </div>
        )}
      </div>
      
      {/* Clear button */}
      <button
        onClick={clearCanvas}
        className="absolute top-4 right-4 bg-red-500/20 hover:bg-red-500/30 border border-red-400/30 text-red-200 px-3 py-2 rounded-lg text-sm transition-all duration-200 backdrop-blur-sm"
      >
        Clear Canvas
      </button>
      
      {/* Real-time feedback */}
      {isDrawing && currentStroke.length > 5 && (
        <div className="absolute bottom-4 left-4 bg-blue-500/20 backdrop-blur-sm rounded-lg px-3 py-2 text-sm text-blue-200 border border-blue-400/30">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
            <span>Real-time analysis active</span>
          </div>
          <div className="text-xs text-blue-300 mt-1">
            {currentStroke.length} points captured
          </div>
        </div>
      )}
      
      {/* Analysis status */}
      {storeStroke.processed.length > 0 && (
        <div className="absolute bottom-4 right-4 bg-green-500/20 backdrop-blur-sm rounded-lg px-3 py-2 text-sm text-green-200 border border-green-400/30">
          <div className="font-medium">Analysis Complete</div>
          <div className="text-xs text-green-300 mt-1">
            {storeStroke.processed.length} processed points • {storeStroke.landmarks.length} landmarks
          </div>
        </div>
      )}
      
      {/* Instructions overlay */}
      {strokeCount === 0 && currentStroke.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center text-white/60">
            <div className="text-lg font-medium text-blue-200 mb-2">Advanced Stroke Analysis Laboratory</div>
            <div className="text-sm">
              Draw letters or symbols to analyze their mathematical properties
            </div>
            <div className="text-xs text-blue-300 mt-2 space-y-1">
              <div>• Real-time curvature analysis</div>
              <div>• Topological invariant detection</div>
              <div>• Affine geometry computation</div>
              <div>• Path signature extraction</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancedDrawingCanvas;
