import React, { useRef, useEffect, useState, useCallback } from 'react';
import { cn } from '@/lib/utils';

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

interface DrawingCanvasProps {
  onStrokeUpdate: (strokes: Stroke[]) => void;
  currentLetter: string;
  analysisMode: 'real-time' | 'on-complete';
  className?: string;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  onStrokeUpdate,
  currentLetter,
  analysisMode,
  className
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [currentStroke, setCurrentStroke] = useState<Stroke | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 });
  const [strokeCount, setStrokeCount] = useState(0);

  const strokeColors = [
    '#3B82F6', // blue
    '#EF4444', // red
    '#10B981', // green
    '#F59E0B', // amber
    '#8B5CF6', // violet
    '#EC4899', // pink
    '#06B6D4', // cyan
    '#84CC16'  // lime
  ];

  // Initialize canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const container = canvas.parentElement;
    if (container) {
      const rect = container.getBoundingClientRect();
      const width = Math.min(600, rect.width - 32);
      const height = 400;
      
      canvas.width = width * 2; // High DPI
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

    // Clear canvas
    ctx.fillStyle = 'rgba(15, 23, 42, 0.8)'; // Dark background
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    drawGrid(ctx, canvasSize.width, canvasSize.height);
  }, []);

  const drawGrid = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.1)';
    ctx.lineWidth = 1;
    
    const gridSize = 20;
    
    // Vertical lines
    for (let x = 0; x <= width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    
    // Horizontal lines
    for (let y = 0; y <= height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  };

  const getCanvasPoint = (e: React.MouseEvent | React.TouchEvent): Point => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    
    let clientX: number, clientY: number;
    
    if ('touches' in e) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
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
    const strokeColor = strokeColors[strokeCount % strokeColors.length];
    
    const newStroke: Stroke = {
      id: `stroke-${Date.now()}-${strokeCount}`,
      points: [point],
      completed: false,
      color: strokeColor
    };
    
    setCurrentStroke(newStroke);
    
    // Start drawing on canvas
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
  };

  const continueDrawing = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing || !currentStroke) return;
    
    e.preventDefault();
    const point = getCanvasPoint(e);
    
    // Add point to current stroke
    const updatedStroke = {
      ...currentStroke,
      points: [...currentStroke.points, point]
    };
    setCurrentStroke(updatedStroke);
    
    // Draw on canvas
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    
    // Real-time analysis
    if (analysisMode === 'real-time') {
      const updatedStrokes = [...strokes, updatedStroke];
      onStrokeUpdate(updatedStrokes);
    }
  }, [isDrawing, currentStroke, strokes, analysisMode, onStrokeUpdate]);

  const stopDrawing = () => {
    if (!isDrawing || !currentStroke) return;
    
    setIsDrawing(false);
    
    // Complete the stroke
    const completedStroke = {
      ...currentStroke,
      completed: true
    };
    
    const updatedStrokes = [...strokes, completedStroke];
    setStrokes(updatedStrokes);
    setCurrentStroke(null);
    setStrokeCount(prev => prev + 1);
    
    // Trigger analysis
    onStrokeUpdate(updatedStrokes);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGrid(ctx, canvasSize.width, canvasSize.height);
    
    setStrokes([]);
    setCurrentStroke(null);
    setStrokeCount(0);
    onStrokeUpdate([]);
  };

  const redrawStrokes = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d')!;
    
    // Clear and redraw grid
    ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawGrid(ctx, canvasSize.width, canvasSize.height);
    
    // Redraw all completed strokes
    strokes.forEach(stroke => {
      if (stroke.points.length < 2) return;
      
      ctx.strokeStyle = stroke.color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      
      stroke.points.slice(1).forEach(point => {
        ctx.lineTo(point.x, point.y);
      });
      
      ctx.stroke();
    });
    
    // Redraw current stroke if drawing
    if (currentStroke && currentStroke.points.length >= 2) {
      ctx.strokeStyle = currentStroke.color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(currentStroke.points[0].x, currentStroke.points[0].y);
      
      currentStroke.points.slice(1).forEach(point => {
        ctx.lineTo(point.x, point.y);
      });
      
      ctx.stroke();
    }
  }, [strokes, currentStroke, canvasSize]);

  // Redraw when strokes change
  useEffect(() => {
    redrawStrokes();
  }, [redrawStrokes]);

  return (
    <div className={cn("relative", className)}>
      <canvas
        ref={canvasRef}
        className="border border-white/20 rounded-lg cursor-crosshair bg-slate-900/50 backdrop-blur-sm"
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
      
      {/* Instructions */}
      <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2 text-sm text-white">
        <div className="font-medium">Draw letter: {currentLetter}</div>
        <div className="text-xs text-blue-200 mt-1">
          {strokes.length === 0 ? 'Click and drag to draw' : `${strokes.length} stroke${strokes.length !== 1 ? 's' : ''} drawn`}
        </div>
      </div>
      
      {/* Clear button */}
      <button
        onClick={clearCanvas}
        className="absolute top-4 right-4 bg-red-500/20 hover:bg-red-500/30 border border-red-400/30 text-red-200 px-3 py-2 rounded-lg text-sm transition-colors"
      >
        Clear Canvas
      </button>
      
      {/* Stroke count */}
      {strokes.length > 0 && (
        <div className="absolute bottom-4 left-4 bg-blue-500/20 backdrop-blur-sm rounded-lg px-3 py-2 text-sm text-blue-200 border border-blue-400/30">
          {strokes.length} stroke{strokes.length !== 1 ? 's' : ''} • {strokes.reduce((sum, stroke) => sum + stroke.points.length, 0)} points
        </div>
      )}
      
      {/* Stroke colors legend */}
      {strokes.length > 1 && (
        <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2 text-xs text-white border border-white/20">
          <div className="font-medium mb-1">Stroke Colors:</div>
          <div className="flex gap-2">
            {strokes.map((stroke, index) => (
              <div key={stroke.id} className="flex items-center gap-1">
                <div 
                  className="w-3 h-3 rounded-full" 
                  style={{ backgroundColor: stroke.color }}
                />
                <span>{index + 1}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default DrawingCanvas;
