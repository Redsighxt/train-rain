import React, { useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Brain, LineChart, Settings, Download, RotateCcw, 
  Activity, Zap, Target, BarChart3, TrendingUp,
  Database, FileText, Layers, Microscope
} from 'lucide-react';

import { useResearchStore } from '@/store/researchStore';
import EnhancedDrawingCanvas from '@/components/EnhancedDrawingCanvas';
import Visualization3D from '@/components/Visualization3D';
import DatasetManagementPanel from '@/components/DatasetManagementPanel';
import { getPerformanceStats } from '@/lib/strokeAnalysisProcessor';

const StrokeAnalysisPage: React.FC = () => {
  const {
    currentStroke,
    settings,
    updateSettings,
    dataset,
    addToDataset,
    setActivePanel,
    activePanel,
    isProcessing,
    processingStage,
    clearCurrentStroke
  } = useResearchStore();

  const [performanceStats, setPerformanceStats] = React.useState(getPerformanceStats());
  const [currentLetter, setCurrentLetter] = React.useState('A');

  // Update performance stats periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setPerformanceStats(getPerformanceStats());
    }, 1000);
    
    return () => clearInterval(interval);
  }, []);

  const exportAnalysisData = () => {
    if (!currentStroke.raw || !currentStroke.invariants) return;
    
    const exportData = {
      metadata: {
        timestamp: new Date().toISOString(),
        letter: currentLetter,
        version: '1.0'
      },
      stroke: {
        raw: currentStroke.raw,
        processed: currentStroke.processed,
        landmarks: currentStroke.landmarks
      },
      analysis: {
        invariants: currentStroke.invariants,
        signature: currentStroke.signature
      },
      settings: settings
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stroke-analysis-${currentLetter}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const addToTrainingSet = () => {
    if (currentStroke.raw && currentStroke.invariants && currentStroke.signature) {
      addToDataset(currentLetter);
    }
  };

  return (
    <div className="min-h-screen research-gradient">
      {/* Advanced Header */}
      <header className="border-b border-white/10 bg-black/30 backdrop-blur-md">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-400 via-purple-500 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-research">Stroke Invariant Research Laboratory</h1>
                <p className="text-sm text-blue-200">Advanced Mathematical Analysis & Pattern Recognition</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Performance indicator */}
              <div className="bg-black/40 rounded-lg px-3 py-2 border border-white/10">
                <div className="text-xs text-gray-300">Performance</div>
                <div className="text-sm font-bold text-green-400">
                  {performanceStats.averageTime.toFixed(0)}ms avg
                </div>
              </div>
              
              {/* Current letter */}
              <Badge variant="outline" className="bg-blue-500/20 text-blue-200 border-blue-400/30 px-3 py-1">
                Letter: {currentLetter}
              </Badge>
              
              <Button 
                variant="outline" 
                size="sm" 
                onClick={exportAnalysisData}
                disabled={!currentStroke.invariants}
                className="text-white border-white/20 hover:bg-white/10"
              >
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-6">
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          
          {/* Left Panel - Drawing and Controls */}
          <div className="xl:col-span-5 space-y-6">
            
            {/* Enhanced Drawing Canvas */}
            <Card className="analysis-card neon-blue">
              <CardHeader className="pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white flex items-center">
                    <Microscope className="w-5 h-5 mr-2 text-blue-400" />
                    Research Canvas
                  </CardTitle>
                  <div className="flex items-center space-x-2">
                    {isProcessing && (
                      <div className="flex items-center space-x-2 text-blue-300">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                        <span className="text-xs">{processingStage}</span>
                      </div>
                    )}
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={clearCurrentStroke}
                      className="text-white border-white/20 hover:bg-white/10"
                    >
                      <RotateCcw className="w-4 h-4 mr-2" />
                      Clear
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <EnhancedDrawingCanvas />
              </CardContent>
            </Card>

            {/* Advanced Controls */}
            <Card className="analysis-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-white flex items-center">
                  <Settings className="w-5 h-5 mr-2 text-purple-400" />
                  Analysis Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                
                {/* Target Letter Selection */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-white">Target Letter</Label>
                    <select 
                      value={currentLetter} 
                      onChange={(e) => setCurrentLetter(e.target.value)}
                      className="w-full mt-2 px-3 py-2 bg-white/10 border border-white/20 rounded-md text-white backdrop-blur-sm"
                    >
                      {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('').map(letter => (
                        <option key={letter} value={letter} className="bg-slate-800">{letter}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <Label className="text-white">Processing Points</Label>
                    <div className="mt-2">
                      <Slider
                        value={[settings.numPoints]}
                        onValueChange={(value) => updateSettings({ numPoints: value[0] })}
                        max={200}
                        min={25}
                        step={5}
                        className="w-full"
                      />
                      <div className="text-xs text-blue-300 mt-1">{settings.numPoints} points</div>
                    </div>
                  </div>
                </div>

                <Separator className="bg-white/10" />

                {/* Analysis Modules */}
                <div className="space-y-4">
                  <h4 className="text-white font-medium flex items-center">
                    <Zap className="w-4 h-4 mr-2 text-yellow-400" />
                    Mathematical Modules
                  </h4>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10">
                      <Label className="text-white text-sm">Affine Geometry</Label>
                      <Switch 
                        checked={settings.enableAffineAnalysis}
                        onCheckedChange={(checked) => updateSettings({ enableAffineAnalysis: checked })}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10">
                      <Label className="text-white text-sm">Topology (TDA)</Label>
                      <Switch 
                        checked={settings.enableTopologicalAnalysis}
                        onCheckedChange={(checked) => updateSettings({ enableTopologicalAnalysis: checked })}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10">
                      <Label className="text-white text-sm">Path Signature</Label>
                      <Switch 
                        checked={settings.enablePathSignature}
                        onCheckedChange={(checked) => updateSettings({ enablePathSignature: checked })}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10">
                      <Label className="text-white text-sm">Spectral Analysis</Label>
                      <Switch 
                        checked={settings.enableSpectralAnalysis}
                        onCheckedChange={(checked) => updateSettings({ enableSpectralAnalysis: checked })}
                      />
                    </div>
                  </div>
                </div>

                <Separator className="bg-white/10" />

                {/* Visualization Settings */}
                <div className="space-y-4">
                  <h4 className="text-white font-medium flex items-center">
                    <Layers className="w-4 h-4 mr-2 text-green-400" />
                    Visualization
                  </h4>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label className="text-white">Show Landmarks</Label>
                      <Switch 
                        checked={settings.showLandmarks}
                        onCheckedChange={(checked) => updateSettings({ showLandmarks: checked })}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <Label className="text-white">Density Cloud</Label>
                      <Switch 
                        checked={settings.showDensityCloud}
                        onCheckedChange={(checked) => updateSettings({ showDensityCloud: checked })}
                      />
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <Label className="text-white">Invariant Connections</Label>
                      <Switch 
                        checked={settings.showInvariantConnections}
                        onCheckedChange={(checked) => updateSettings({ showInvariantConnections: checked })}
                      />
                    </div>
                  </div>
                </div>

                {/* Training Data */}
                <Separator className="bg-white/10" />
                
                <div className="space-y-3">
                  <h4 className="text-white font-medium flex items-center">
                    <Database className="w-4 h-4 mr-2 text-orange-400" />
                    Training Data
                  </h4>
                  
                  <div className="flex space-x-2">
                    <Button 
                      onClick={addToTrainingSet}
                      disabled={!currentStroke.invariants}
                      className="flex-1 bg-green-600/20 hover:bg-green-600/30 border border-green-500/30 text-green-200"
                    >
                      Add to Dataset
                    </Button>
                    <Badge variant="outline" className="bg-blue-500/20 text-blue-200 border-blue-400/30 px-3 py-2">
                      {dataset.length} samples
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Dataset & Model Management */}
            <DatasetManagementPanel />
          </div>

          {/* Right Panel - Visualization and Analysis */}
          <div className="xl:col-span-7 space-y-6">
            
            {/* 3D Signature Visualization */}
            <Card className="analysis-card neon-purple">
              <CardHeader className="pb-4">
                <CardTitle className="text-white flex items-center">
                  <Activity className="w-5 h-5 mr-2 text-purple-400" />
                  3D Invariant Signature
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Visualization3D 
                  strokeData={currentStroke.raw ? [currentStroke.raw] : []}
                  invariantPoints={currentStroke.landmarks?.map((landmark, index) => ({
                    id: `inv-${index}`,
                    index: Math.floor(landmark.position.x * (currentStroke.raw?.points.length || 1)),
                    strokeIndex: 0,
                    position: landmark.position,
                    stabilityScore: landmark.stability,
                    type: 'primary' as const,
                    category: 'geometric' as const,
                    description: landmark.type,
                    confidence: landmark.stability
                  })) || []}
                  settings={{
                    showInvariantPoints: settings.showLandmarks,
                    showConnections: settings.showInvariantConnections,
                    showDensityCloud: settings.showDensityCloud,
                    pointSize: [5],
                    lineWidth: [2],
                    colorMode: 'stroke'
                  }}
                />
              </CardContent>
            </Card>

            {/* Analysis Results Tabs */}
            <Card className="analysis-card">
              <CardHeader className="pb-4">
                <CardTitle className="text-white flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
                  Mathematical Analysis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="invariants" className="w-full">
                  <TabsList className="grid w-full grid-cols-4 bg-white/10">
                    <TabsTrigger value="invariants" className="text-white data-[state=active]:bg-blue-600">
                      Invariants
                    </TabsTrigger>
                    <TabsTrigger value="landmarks" className="text-white data-[state=active]:bg-purple-600">
                      Landmarks
                    </TabsTrigger>
                    <TabsTrigger value="signature" className="text-white data-[state=active]:bg-green-600">
                      Signature
                    </TabsTrigger>
                    <TabsTrigger value="metrics" className="text-white data-[state=active]:bg-orange-600">
                      Metrics
                    </TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="invariants" className="mt-6 space-y-4">
                    {currentStroke.invariants ? (
                      <div className="grid grid-cols-2 gap-4">
                        
                        {/* Geometric Invariants */}
                        <div className="bg-blue-500/10 p-4 rounded-lg border border-blue-400/30">
                          <h5 className="text-blue-200 font-medium mb-3 flex items-center">
                            <TrendingUp className="w-4 h-4 mr-2" />
                            Geometric
                          </h5>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-300">Arc Length:</span>
                              <span className="text-blue-200">{currentStroke.invariants.arcLength.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Total Turning:</span>
                              <span className="text-blue-200">{currentStroke.invariants.totalTurning.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Writhe:</span>
                              <span className="text-blue-200">{currentStroke.invariants.writhe.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Winding Number:</span>
                              <span className="text-blue-200">{currentStroke.invariants.windingNumber.toFixed(3)}</span>
                            </div>
                          </div>
                        </div>

                        {/* Topological Invariants */}
                        <div className="bg-purple-500/10 p-4 rounded-lg border border-purple-400/30">
                          <h5 className="text-purple-200 font-medium mb-3 flex items-center">
                            <Target className="w-4 h-4 mr-2" />
                            Topological
                          </h5>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-300">β₀ (Components):</span>
                              <span className="text-purple-200">{currentStroke.invariants.bettiNumbers.b0}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">β₁ (Loops):</span>
                              <span className="text-purple-200">{currentStroke.invariants.bettiNumbers.b1}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Persistence:</span>
                              <span className="text-purple-200">{currentStroke.invariants.persistenceDiagram.length}</span>
                            </div>
                          </div>
                        </div>

                        {/* Statistical Invariants */}
                        <div className="bg-green-500/10 p-4 rounded-lg border border-green-400/30">
                          <h5 className="text-green-200 font-medium mb-3">Statistical</h5>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-300">Complexity:</span>
                              <span className="text-green-200">{currentStroke.invariants.complexityScore.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Regularity:</span>
                              <span className="text-green-200">{currentStroke.invariants.regularityScore.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Symmetry:</span>
                              <span className="text-green-200">{currentStroke.invariants.symmetryScore.toFixed(3)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Stability:</span>
                              <span className="text-green-200">{currentStroke.invariants.stabilityIndex.toFixed(3)}</span>
                            </div>
                          </div>
                        </div>

                        {/* Path Signature */}
                        <div className="bg-orange-500/10 p-4 rounded-lg border border-orange-400/30">
                          <h5 className="text-orange-200 font-medium mb-3">Path Signature</h5>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-300">Level 1 Norm:</span>
                              <span className="text-orange-200">
                                {Math.sqrt(currentStroke.invariants.pathSignature.level1.reduce((sum, x) => sum + x*x, 0)).toFixed(3)}
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Level 2 Terms:</span>
                              <span className="text-orange-200">{currentStroke.invariants.pathSignature.level2.length}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-300">Log Signature:</span>
                              <span className="text-orange-200">{currentStroke.invariants.pathSignature.logSignature.length}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-gray-400 py-8">
                        <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Draw a letter to see mathematical invariants</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="landmarks" className="mt-6">
                    {currentStroke.landmarks.length > 0 ? (
                      <div className="space-y-3 max-h-96 overflow-y-auto">
                        {currentStroke.landmarks.map((landmark, index) => (
                          <div key={landmark.id} className="bg-white/5 rounded-lg p-3 border border-white/10">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                <div className={`w-3 h-3 rounded-full ${
                                  landmark.type === 'endpoint' ? 'bg-red-400' :
                                  landmark.type === 'peak' ? 'bg-yellow-400' :
                                  landmark.type === 'inflection' ? 'bg-blue-400' :
                                  'bg-green-400'
                                }`}></div>
                                <div>
                                  <div className="text-sm font-medium text-white capitalize">{landmark.type}</div>
                                  <div className="text-xs text-gray-400">
                                    Position: ({landmark.position.x.toFixed(3)}, {landmark.position.y.toFixed(3)})
                                  </div>
                                </div>
                              </div>
                              <div className="text-right">
                                <div className="text-sm font-bold text-blue-400">
                                  {(landmark.stability * 100).toFixed(0)}%
                                </div>
                                <div className="text-xs text-gray-400">stability</div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center text-gray-400 py-8">
                        <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>No landmark points detected</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="signature" className="mt-6">
                    {currentStroke.signature ? (
                      <div className="space-y-4">
                        <div className="grid grid-cols-3 gap-4">
                          <div className="metric-card bg-blue-500/10 p-4 rounded-lg border border-blue-400/30">
                            <div className="text-blue-200 text-sm">3D Points</div>
                            <div className="text-2xl font-bold text-blue-100">{currentStroke.signature.coordinates.length}</div>
                          </div>
                          <div className="metric-card bg-green-500/10 p-4 rounded-lg border border-green-400/30">
                            <div className="text-green-200 text-sm">Separation Ratio</div>
                            <div className="text-2xl font-bold text-green-100">
                              {currentStroke.signature.qualityMetrics.separationRatio.toFixed(2)}
                            </div>
                          </div>
                          <div className="metric-card bg-purple-500/10 p-4 rounded-lg border border-purple-400/30">
                            <div className="text-purple-200 text-sm">Stability Index</div>
                            <div className="text-2xl font-bold text-purple-100">
                              {(currentStroke.signature.qualityMetrics.stabilityIndex * 100).toFixed(0)}%
                            </div>
                          </div>
                        </div>
                        
                        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                          <h5 className="text-white font-medium mb-3">Axis Mapping</h5>
                          <div className="grid grid-cols-3 gap-3 text-sm">
                            <div>
                              <span className="text-red-300">X-axis:</span>
                              <div className="text-red-200 font-medium">{currentStroke.signature.axisMapping.x}</div>
                            </div>
                            <div>
                              <span className="text-green-300">Y-axis:</span>
                              <div className="text-green-200 font-medium">{currentStroke.signature.axisMapping.y}</div>
                            </div>
                            <div>
                              <span className="text-blue-300">Z-axis:</span>
                              <div className="text-blue-200 font-medium">{currentStroke.signature.axisMapping.z}</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center text-gray-400 py-8">
                        <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>No 3D signature generated yet</p>
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="metrics" className="mt-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="metric-card bg-blue-500/10 p-4 rounded-lg border border-blue-400/30">
                        <div className="text-blue-200 text-sm">Processing Time</div>
                        <div className="text-2xl font-bold text-blue-100">
                          {performanceStats.averageTime.toFixed(0)}ms
                        </div>
                        <div className="text-xs text-blue-300 mt-1">
                          Trend: {performanceStats.recentTrend}
                        </div>
                      </div>
                      
                      <div className="metric-card bg-green-500/10 p-4 rounded-lg border border-green-400/30">
                        <div className="text-green-200 text-sm">Success Rate</div>
                        <div className="text-2xl font-bold text-green-100">
                          {(performanceStats.successRate * 100).toFixed(0)}%
                        </div>
                      </div>
                      
                      <div className="metric-card bg-purple-500/10 p-4 rounded-lg border border-purple-400/30">
                        <div className="text-purple-200 text-sm">Processed Points</div>
                        <div className="text-2xl font-bold text-purple-100">
                          {currentStroke.processed.length}
                        </div>
                      </div>
                      
                      <div className="metric-card bg-orange-500/10 p-4 rounded-lg border border-orange-400/30">
                        <div className="text-orange-200 text-sm">Dataset Size</div>
                        <div className="text-2xl font-bold text-orange-100">
                          {dataset.length}
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StrokeAnalysisPage;
