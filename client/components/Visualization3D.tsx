import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { generate3DSignature, type Stroke, type InvariantPoint } from '@/lib/signature-math';

// Dynamic import for Plotly to avoid SSR issues
let Plotly: any = null;
if (typeof window !== 'undefined') {
  import('plotly.js-dist-min').then((module) => {
    Plotly = module.default;
  });
}

interface Visualization3DProps {
  strokeData: Stroke[];
  invariantPoints: InvariantPoint[];
  settings: {
    showInvariantPoints: boolean;
    showConnections: boolean;
    showDensityCloud: boolean;
    pointSize: number[];
    lineWidth: number[];
    colorMode: string;
  };
}

const Visualization3D: React.FC<Visualization3DProps> = ({
  strokeData,
  invariantPoints,
  settings
}) => {
  const plotRef = useRef<HTMLDivElement>(null);
  const [isPlotlyLoaded, setIsPlotlyLoaded] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastStrokeDataHash, setLastStrokeDataHash] = useState<string>('');

  // Debug logging
  console.log('Visualization3D render:', {
    strokeDataLength: strokeData.length,
    invariantPointsLength: invariantPoints.length,
    isPlotlyLoaded,
    settings
  });

  // Check if Plotly is loaded
  useEffect(() => {
    const checkPlotly = () => {
      if (Plotly) {
        console.log('Plotly loaded successfully');
        setIsPlotlyLoaded(true);
      } else {
        console.log('Plotly not loaded yet, retrying...');
        setTimeout(checkPlotly, 100);
      }
    };
    checkPlotly();
  }, []);

  // Create a stable hash of stroke data to prevent unnecessary recalculations
  const strokeDataHash = useMemo(() => {
    return JSON.stringify(strokeData.map(stroke => ({
      id: stroke.id,
      pointsLength: stroke.points.length,
      completed: stroke.completed
    })));
  }, [strokeData]);

  // Memoized signature generation to prevent beating heart effect
  const memoizedSignature = useMemo(() => {
    console.log('Generating 3D signature:', {
      isPlotlyLoaded,
      strokeDataLength: strokeData.length,
      invariantPointsLength: invariantPoints.length
    });
    
    if (!isPlotlyLoaded || strokeData.length === 0) {
      console.log('Cannot generate signature - Plotly not loaded or no stroke data');
      return null;
    }
    
    // Only regenerate if stroke data has actually changed
    if (strokeDataHash === lastStrokeDataHash) {
      console.log('Stroke data unchanged, skipping regeneration');
      return null; // Return null to prevent unnecessary updates
    }
    
    setLastStrokeDataHash(strokeDataHash);
    
    try {
      const signature = generate3DSignature(strokeData, invariantPoints);
      console.log('3D signature generated:', signature);
      return signature;
    } catch (error) {
      console.error('Error generating 3D signature:', error);
      return null;
    }
  }, [strokeData, invariantPoints, isPlotlyLoaded, strokeDataHash, lastStrokeDataHash]);

  // Create density cloud for visualization
  const createDensityCloud = useCallback((points: any[]) => {
    const densityPoints: { x: number; y: number; z: number; density: number }[] = [];
    const gridSize = 2;
    
    if (points.length === 0) return densityPoints;
    
    // Create a 3D grid and calculate point density
    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const zs = points.map(p => p.z);
    
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    
    for (let x = minX; x <= maxX; x += gridSize) {
      for (let y = minY; y <= maxY; y += gridSize) {
        for (let z = minZ; z <= maxZ; z += gridSize) {
          const nearbyPoints = points.filter(p => 
            Math.abs(p.x - x) < gridSize &&
            Math.abs(p.y - y) < gridSize &&
            Math.abs(p.z - z) < gridSize
          );
          
          if (nearbyPoints.length > 1) {
            densityPoints.push({
              x, y, z,
              density: nearbyPoints.length / points.length
            });
          }
        }
      }
    }
    
    return densityPoints.slice(0, 100); // Limit for performance
  }, []);

  // Memoized visualization data generation
  const plotData = useMemo(() => {
    console.log('Generating plot data:', {
      hasPlotly: !!Plotly,
      hasSignature: !!memoizedSignature,
      signaturePoints: memoizedSignature?.points?.length || 0
    });
    
    if (!Plotly || !memoizedSignature) {
      console.log('Cannot generate plot data - missing Plotly or signature');
      return [];
    }
    
    const signature = memoizedSignature;
    const traces: any[] = [];
    
    // Group points by stroke
    const strokeGroups = new Map<number, typeof signature.points>();
    signature.points.forEach(point => {
      if (!strokeGroups.has(point.strokeIndex)) {
        strokeGroups.set(point.strokeIndex, []);
      }
      strokeGroups.get(point.strokeIndex)!.push(point);
    });
    
    console.log('Stroke groups:', strokeGroups.size, 'groups created');
    
    // Create traces for each stroke
    strokeGroups.forEach((points, strokeIndex) => {
      const stroke = strokeData[strokeIndex];
      if (!stroke) return;
      
      console.log(`Creating trace for stroke ${strokeIndex}:`, points.length, 'points');
      
      // Main stroke trace
      traces.push({
        type: 'scatter3d',
        mode: settings.showConnections ? 'lines+markers' : 'markers',
        x: points.map(p => p.x),
        y: points.map(p => p.y),
        z: points.map(p => p.z),
        marker: {
          size: points.map(p => settings.pointSize[0] * (1 + p.invarianceWeight * 2)),
          color: stroke.color,
          opacity: 0.8,
          symbol: 'circle',
          line: {
            color: 'rgba(255,255,255,0.3)',
            width: 1
          }
        },
        line: {
          color: stroke.color,
          width: settings.lineWidth[0]
        },
        name: `Stroke ${strokeIndex + 1}`,
        hovertemplate: 
          'X: %{x:.2f}<br>' +
          'Y: %{y:.2f}<br>' +
          'Z: %{z:.2f}<br>' +
          `Stroke: ${strokeIndex + 1}<br>` +
          'Point: %{pointNumber}<br>' +
          'Invariance: %{text}<br>' +
          '<extra></extra>',
        text: points.map(p => (p.invarianceWeight * 100).toFixed(1) + '%')
      });
    });
    
    // Add invariant points if enabled
    if (settings.showInvariantPoints && signature.invariantPoints.length > 0) {
      console.log('Adding invariant points trace:', signature.invariantPoints.length, 'points');
      traces.push({
        type: 'scatter3d',
        mode: 'markers',
        x: signature.invariantPoints.map(p => p.position.x),
        y: signature.invariantPoints.map(p => p.position.y),
        z: signature.invariantPoints.map(p => p.position.z),
        marker: {
          size: 8,
          color: 'yellow',
          symbol: 'diamond',
          line: {
            color: 'white',
            width: 2
          }
        },
        name: 'Invariant Points',
        hovertemplate: 
          'Invariant Point<br>' +
          'X: %{x:.2f}<br>' +
          'Y: %{y:.2f}<br>' +
          'Z: %{z:.2f}<br>' +
          'Stability: %{text}<br>' +
          '<extra></extra>',
        text: signature.invariantPoints.map(p => (p.stability * 100).toFixed(1) + '%')
      });
    }
    
    console.log('Generated traces:', traces.length, 'traces');
    return traces;
  }, [memoizedSignature, strokeData, settings]);

  // Stable plot layout to prevent re-renders
  const plotLayout = useMemo(() => ({
    scene: {
      xaxis: { 
        title: 'Geometric Complexity',
        gridcolor: 'rgba(255,255,255,0.1)',
        zerolinecolor: 'rgba(255,255,255,0.3)',
        showbackground: true,
        backgroundcolor: 'rgba(0,0,0,0.1)'
      },
      yaxis: { 
        title: 'Temporal Features',
        gridcolor: 'rgba(255,255,255,0.1)',
        zerolinecolor: 'rgba(255,255,255,0.3)',
        showbackground: true,
        backgroundcolor: 'rgba(0,0,0,0.1)'
      },
      zaxis: { 
        title: 'Direction Patterns',
        gridcolor: 'rgba(255,255,255,0.1)',
        zerolinecolor: 'rgba(255,255,255,0.3)',
        showbackground: true,
        backgroundcolor: 'rgba(0,0,0,0.1)'
      },
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.5 }
      }
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: 'white' },
    margin: { l: 0, r: 0, b: 0, t: 30 },
    showlegend: true,
    legend: {
      font: { color: 'white' },
      bgcolor: 'rgba(0,0,0,0.5)',
      bordercolor: 'rgba(255,255,255,0.3)',
      borderwidth: 1
    }
  }), []);

  // Update visualization with memoized data
  const updateVisualization = useCallback(async () => {
    console.log('updateVisualization called:', {
      hasPlotly: !!Plotly,
      hasPlotRef: !!plotRef.current,
      plotDataLength: plotData.length,
      isProcessing
    });
    
    if (!Plotly || !plotRef.current || plotData.length === 0) {
      console.log('Cannot update visualization - missing requirements');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      console.log('Rendering Plotly visualization with data:', plotData);
      
      const config = {
        displayModeBar: true,
        modeBarButtonsToRemove: ['sendDataToCloud'],
        displaylogo: false,
        responsive: true,
        modeBarButtonsToAdd: [{
          name: 'Reset View',
          icon: Plotly.Icons.home,
          click: () => resetView()
        }]
      };
      
      await Plotly.newPlot(plotRef.current, plotData, plotLayout, config);
      console.log('Plotly visualization rendered successfully');
    } catch (error) {
      console.error('Error updating visualization:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [plotData, plotLayout]);

  // Update when data changes - only when we have new signature data
  useEffect(() => {
    console.log('Visualization useEffect triggered:', {
      isPlotlyLoaded,
      plotDataLength: plotData.length,
      hasSignature: !!memoizedSignature,
      isProcessing
    });
    
    if (isPlotlyLoaded && plotData.length > 0 && memoizedSignature) {
      console.log('Triggering visualization update');
      updateVisualization();
    } else {
      console.log('Not updating visualization - conditions not met');
    }
  }, [isPlotlyLoaded, updateVisualization, plotData, memoizedSignature]);

  const resetView = useCallback(() => {
    if (Plotly && plotRef.current) {
      Plotly.relayout(plotRef.current, {
        'scene.camera': {
          eye: { x: 1.5, y: 1.5, z: 1.5 }
        }
      });
    }
  }, []);

  if (!isPlotlyLoaded) {
    return (
      <div className="flex items-center justify-center h-96 bg-white/5 rounded-lg border border-white/10">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <p className="text-white">Loading 3D visualization engine...</p>
          <p className="text-sm text-gray-400 mt-2">Initializing Plotly.js</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Controls */}
      <div className="absolute top-4 right-4 z-10 flex space-x-2">
        <Button
          size="sm"
          variant="outline"
          onClick={resetView}
          className="bg-black/60 border-white/20 text-white hover:bg-black/80"
        >
          <RotateCcw className="w-4 h-4" />
        </Button>
      </div>

      {/* Processing indicator */}
      {isProcessing && (
        <div className="absolute top-4 left-4 z-10 bg-blue-500/20 backdrop-blur-sm rounded-lg px-3 py-2 text-sm text-blue-200 border border-blue-400/30 flex items-center">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400 mr-2"></div>
          Generating 3D signature...
        </div>
      )}

      {/* Signature metrics */}
      {memoizedSignature && strokeData.length > 0 && (
        <div className="absolute top-4 left-4 z-10 bg-black/60 backdrop-blur-sm rounded-lg px-3 py-2 text-sm text-white border border-white/20">
          <div className="font-medium mb-1">Signature Quality</div>
          <div className="text-xs space-y-1">
            <div>Separation Ratio: {memoizedSignature.signatureMetrics.separationRatio.toFixed(2)}</div>
            <div>Stability Index: {(memoizedSignature.signatureMetrics.stabilityIndex * 100).toFixed(0)}%</div>
            <div>Variance: {memoizedSignature.signatureMetrics.intraClusterVariance.toFixed(2)}</div>
          </div>
        </div>
      )}

      {/* 3D Plot */}
      <div className="relative">
        <div 
          ref={plotRef} 
          className="w-full h-96 bg-slate-900/50 rounded-lg border border-white/20"
          style={{ minHeight: '400px' }}
        />
        
        {strokeData.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-white/60">
              <AlertCircle className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium">Advanced 3D Stroke Signature</p>
              <p className="text-sm mt-2 max-w-md">
                Draw a letter to see its unique 3D signature pattern. The visualization shows geometric complexity, 
                temporal features, and direction patterns with invariant points highlighted.
              </p>
              <div className="mt-4 text-xs text-blue-200 space-y-1">
                <div><strong>X-axis:</strong> Geometric complexity (curvature + intersections)</div>
                <div><strong>Y-axis:</strong> Temporal features (velocity + arc position)</div>
                <div><strong>Z-axis:</strong> Direction patterns (angles + symmetry)</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Signature info panel */}
      {memoizedSignature && strokeData.length > 0 && (
        <div className="mt-4 grid grid-cols-4 gap-3 text-sm">
          <div className="bg-blue-500/20 p-3 rounded-lg border border-blue-400/30 text-center">
            <div className="text-blue-200">3D Points</div>
            <div className="text-xl font-bold text-blue-100">{memoizedSignature.points.length}</div>
          </div>
          <div className="bg-green-500/20 p-3 rounded-lg border border-green-400/30 text-center">
            <div className="text-green-200">Invariant Points</div>
            <div className="text-xl font-bold text-green-100">{memoizedSignature.invariantPoints.length}</div>
          </div>
          <div className="bg-purple-500/20 p-3 rounded-lg border border-purple-400/30 text-center">
            <div className="text-purple-200">Stability</div>
            <div className="text-xl font-bold text-purple-100">
              {(memoizedSignature.signatureMetrics.stabilityIndex * 100).toFixed(0)}%
            </div>
          </div>
          <div className="bg-orange-500/20 p-3 rounded-lg border border-orange-400/30 text-center">
            <div className="text-orange-200">Quality Score</div>
            <div className="text-xl font-bold text-orange-100">
              {Math.min(memoizedSignature.signatureMetrics.separationRatio * 10, 10).toFixed(1)}/10
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Visualization3D;
