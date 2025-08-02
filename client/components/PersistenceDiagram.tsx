import React, { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { BarChart3, Circle, Square, Triangle } from 'lucide-react';
import type { PersistenceInterval } from '@/lib/topologicalAnalysis';

interface PersistenceDiagramProps {
  intervals: PersistenceInterval[];
  className?: string;
}

const PersistenceDiagram: React.FC<PersistenceDiagramProps> = ({ intervals, className }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!canvasRef.current || intervals.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas size
    const size = 300;
    canvas.width = size * 2; // High DPI
    canvas.height = size * 2;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    ctx.scale(2, 2);
    
    // Clear canvas
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    ctx.fillRect(0, 0, size, size);
    
    // Find bounds for finite intervals
    const finiteIntervals = intervals.filter(i => i.persistence !== Infinity && i.persistence > 0);
    if (finiteIntervals.length === 0) return;
    
    const maxBirth = Math.max(...finiteIntervals.map(i => i.birth));
    const maxDeath = Math.max(...finiteIntervals.map(i => i.death));
    const maxValue = Math.max(maxBirth, maxDeath);
    const minValue = Math.min(...finiteIntervals.map(i => i.birth));
    
    const margin = 40;
    const plotSize = size - 2 * margin;
    
    // Draw coordinate system
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    
    // X and Y axes
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, size - margin);
    ctx.lineTo(size - margin, size - margin);
    ctx.stroke();
    
    // Diagonal line (birth = death)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(margin, size - margin);
    ctx.lineTo(size - margin, margin);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    const gridSteps = 5;
    for (let i = 1; i < gridSteps; i++) {
      const pos = margin + (plotSize * i) / gridSteps;
      
      // Vertical grid lines
      ctx.beginPath();
      ctx.moveTo(pos, margin);
      ctx.lineTo(pos, size - margin);
      ctx.stroke();
      
      // Horizontal grid lines
      ctx.beginPath();
      ctx.moveTo(margin, pos);
      ctx.lineTo(size - margin, pos);
      ctx.stroke();
    }
    
    // Draw points
    const dimensionColors = {
      0: '#3B82F6', // Blue for components
      1: '#EF4444', // Red for loops
      2: '#10B981'  // Green for voids
    };
    
    const dimensionShapes = {
      0: 'circle',
      1: 'square', 
      2: 'triangle'
    };
    
    finiteIntervals.forEach(interval => {
      const x = margin + ((interval.birth - minValue) / (maxValue - minValue)) * plotSize;
      const y = size - margin - ((interval.death - minValue) / (maxValue - minValue)) * plotSize;
      
      const color = dimensionColors[interval.dimension as keyof typeof dimensionColors] || '#9CA3AF';
      const shape = dimensionShapes[interval.dimension as keyof typeof dimensionShapes] || 'circle';
      
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      
      const pointSize = 4 + interval.persistence * 3; // Size based on persistence
      
      switch (shape) {
        case 'circle':
          ctx.beginPath();
          ctx.arc(x, y, pointSize, 0, 2 * Math.PI);
          ctx.fill();
          break;
          
        case 'square':
          ctx.fillRect(x - pointSize, y - pointSize, pointSize * 2, pointSize * 2);
          break;
          
        case 'triangle':
          ctx.beginPath();
          ctx.moveTo(x, y - pointSize);
          ctx.lineTo(x - pointSize, y + pointSize);
          ctx.lineTo(x + pointSize, y + pointSize);
          ctx.closePath();
          ctx.fill();
          break;
      }
    });
    
    // Draw labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    
    // X-axis label
    ctx.fillText('Birth', size / 2, size - 10);
    
    // Y-axis label
    ctx.save();
    ctx.translate(15, size / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Death', 0, 0);
    ctx.restore();
    
    // Axis values
    ctx.font = '10px monospace';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    
    for (let i = 0; i <= gridSteps; i++) {
      const value = minValue + ((maxValue - minValue) * i) / gridSteps;
      const pos = margin + (plotSize * i) / gridSteps;
      
      // X-axis values
      ctx.textAlign = 'center';
      ctx.fillText(value.toFixed(2), pos, size - 5);
      
      // Y-axis values
      ctx.textAlign = 'right';
      ctx.fillText(value.toFixed(2), margin - 5, size - pos);
    }
    
  }, [intervals]);
  
  // Calculate statistics
  const dimensionCounts = intervals.reduce((counts, interval) => {
    counts[interval.dimension] = (counts[interval.dimension] || 0) + 1;
    return counts;
  }, {} as Record<number, number>);
  
  const finiteIntervals = intervals.filter(i => i.persistence !== Infinity && i.persistence > 0);
  const avgPersistence = finiteIntervals.length > 0 ?
    finiteIntervals.reduce((sum, i) => sum + i.persistence, 0) / finiteIntervals.length : 0;
  
  const maxPersistence = finiteIntervals.length > 0 ?
    Math.max(...finiteIntervals.map(i => i.persistence)) : 0;
  
  return (
    <Card className={`analysis-card ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-purple-400" />
            Persistence Diagram
          </CardTitle>
          <Badge variant="outline" className="bg-purple-500/20 text-purple-200 border-purple-400/30">
            {intervals.length} intervals
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {intervals.length > 0 ? (
          <>
            {/* Diagram */}
            <div className="relative">
              <canvas
                ref={canvasRef}
                className="border border-white/20 rounded-lg bg-slate-900/50"
              />
              
              {/* Legend */}
              <div className="absolute top-2 right-2 bg-black/70 backdrop-blur-sm rounded-lg p-2 text-xs">
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <Circle className="w-3 h-3 text-blue-400" />
                    <span className="text-blue-200">Components (H₀)</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Square className="w-3 h-3 text-red-400" />
                    <span className="text-red-200">Loops (H₁)</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Triangle className="w-3 h-3 text-green-400" />
                    <span className="text-green-200">Voids (H₂)</span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Statistics */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-blue-500/10 p-3 rounded-lg border border-blue-400/30">
                <div className="text-blue-200 text-sm">Avg Persistence</div>
                <div className="text-xl font-bold text-blue-100">
                  {avgPersistence.toFixed(3)}
                </div>
              </div>
              
              <div className="bg-purple-500/10 p-3 rounded-lg border border-purple-400/30">
                <div className="text-purple-200 text-sm">Max Persistence</div>
                <div className="text-xl font-bold text-purple-100">
                  {maxPersistence.toFixed(3)}
                </div>
              </div>
            </div>
            
            {/* Dimension breakdown */}
            <div className="space-y-2">
              <h5 className="text-white font-medium">Dimension Breakdown</h5>
              <div className="space-y-1">
                {Object.entries(dimensionCounts).map(([dim, count]) => {
                  const dimension = parseInt(dim);
                  const color = dimension === 0 ? 'blue' : dimension === 1 ? 'red' : 'green';
                  const label = dimension === 0 ? 'Components' : dimension === 1 ? 'Loops' : 'Voids';
                  
                  return (
                    <div key={dim} className="flex items-center justify-between text-sm">
                      <span className={`text-${color}-200`}>H₃{dim} ({label}):</span>
                      <Badge variant="outline" className={`bg-${color}-500/20 text-${color}-200 border-${color}-400/30`}>
                        {count}
                      </Badge>
                    </div>
                  );
                })}
              </div>
            </div>
            
            {/* Topological summary */}
            <div className="bg-white/5 rounded-lg p-3 border border-white/10">
              <h5 className="text-white font-medium mb-2">Topological Summary</h5>
              <div className="text-sm text-gray-300 space-y-1">
                <div>Total Features: {intervals.length}</div>
                <div>Finite Features: {finiteIntervals.length}</div>
                <div>Infinite Features: {intervals.length - finiteIntervals.length}</div>
                <div>
                  Topological Complexity: {(avgPersistence * intervals.length).toFixed(2)}
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center text-gray-400 py-8">
            <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No persistence intervals detected</p>
            <p className="text-sm mt-2">Draw a more complex shape to see topological features</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default PersistenceDiagram;
