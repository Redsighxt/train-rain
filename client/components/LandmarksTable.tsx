import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { 
  Target, Edit2, Save, X, Star, Circle, Triangle, 
  Diamond, Plus, Trash2 
} from 'lucide-react';
import { useResearchStore } from '@/store/researchStore';
import type { LandmarkPoint } from '@/store/researchStore';

interface LandmarksTableProps {
  className?: string;
}

const LandmarksTable: React.FC<LandmarksTableProps> = ({ className }) => {
  const { currentStroke } = useResearchStore();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editLabel, setEditLabel] = useState('');
  const [sortBy, setSortBy] = useState<'stability' | 'type' | 'position'>('stability');
  const [filterType, setFilterType] = useState<string>('all');
  
  const landmarks = currentStroke.landmarks || [];
  
  const getLandmarkIcon = (type: string) => {
    switch (type) {
      case 'peak': return <Star className="w-4 h-4 text-yellow-400" />;
      case 'valley': return <Circle className="w-4 h-4 text-blue-400" />;
      case 'inflection': return <Triangle className="w-4 h-4 text-purple-400" />;
      case 'endpoint': return <Diamond className="w-4 h-4 text-red-400" />;
      case 'intersection': return <Plus className="w-4 h-4 text-green-400" />;
      default: return <Target className="w-4 h-4 text-gray-400" />;
    }
  };
  
  const getTypeColor = (type: string) => {
    switch (type) {
      case 'peak': return 'bg-yellow-500/20 text-yellow-200 border-yellow-400/30';
      case 'valley': return 'bg-blue-500/20 text-blue-200 border-blue-400/30';
      case 'inflection': return 'bg-purple-500/20 text-purple-200 border-purple-400/30';
      case 'endpoint': return 'bg-red-500/20 text-red-200 border-red-400/30';
      case 'intersection': return 'bg-green-500/20 text-green-200 border-green-400/30';
      default: return 'bg-gray-500/20 text-gray-200 border-gray-400/30';
    }
  };
  
  const startEditing = (landmark: LandmarkPoint) => {
    setEditingId(landmark.id);
    setEditLabel(landmark.userLabel || '');
  };
  
  const saveLabel = () => {
    // In a full implementation, this would update the landmark in the store
    console.log(`Saving label "${editLabel}" for landmark ${editingId}`);
    setEditingId(null);
    setEditLabel('');
  };
  
  const cancelEditing = () => {
    setEditingId(null);
    setEditLabel('');
  };
  
  const deleteLandmark = (id: string) => {
    // In a full implementation, this would remove the landmark from the store
    console.log(`Deleting landmark ${id}`);
  };
  
  // Filter and sort landmarks
  const filteredLandmarks = landmarks
    .filter(landmark => filterType === 'all' || landmark.type === filterType)
    .sort((a, b) => {
      switch (sortBy) {
        case 'stability':
          return b.stability - a.stability;
        case 'type':
          return a.type.localeCompare(b.type);
        case 'position':
          return a.position.x - b.position.x;
        default:
          return 0;
      }
    });
  
  const landmarkTypes = [...new Set(landmarks.map(l => l.type))];
  
  return (
    <Card className={`analysis-card ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center">
            <Target className="w-5 h-5 mr-2 text-orange-400" />
            Landmark Points
          </CardTitle>
          <Badge variant="outline" className="bg-orange-500/20 text-orange-200 border-orange-400/30">
            {landmarks.length} detected
          </Badge>
        </div>
        
        {/* Controls */}
        {landmarks.length > 0 && (
          <div className="flex items-center space-x-4">
            {/* Sort control */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-300">Sort by:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="bg-white/10 border border-white/20 rounded px-2 py-1 text-sm text-white"
              >
                <option value="stability" className="bg-slate-800">Stability</option>
                <option value="type" className="bg-slate-800">Type</option>
                <option value="position" className="bg-slate-800">Position</option>
              </select>
            </div>
            
            {/* Filter control */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-300">Filter:</span>
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value)}
                className="bg-white/10 border border-white/20 rounded px-2 py-1 text-sm text-white"
              >
                <option value="all" className="bg-slate-800">All Types</option>
                {landmarkTypes.map(type => (
                  <option key={type} value={type} className="bg-slate-800 capitalize">
                    {type}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </CardHeader>
      
      <CardContent>
        {filteredLandmarks.length > 0 ? (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {filteredLandmarks.map((landmark, index) => (
              <div
                key={landmark.id}
                className="bg-white/5 rounded-lg p-3 border border-white/10 hover:bg-white/10 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3 flex-1">
                    {/* Icon and type */}
                    <div className="flex items-center space-x-2">
                      {getLandmarkIcon(landmark.type)}
                      <Badge variant="outline" className={getTypeColor(landmark.type)}>
                        {landmark.type}
                      </Badge>
                    </div>
                    
                    {/* Position and details */}
                    <div className="flex-1">
                      <div className="text-sm text-white">
                        Position: ({landmark.position.x.toFixed(3)}, {landmark.position.y.toFixed(3)})
                      </div>
                      <div className="text-xs text-gray-400">
                        Magnitude: {landmark.magnitude.toFixed(3)} • 
                        Stability: {(landmark.stability * 100).toFixed(0)}%
                      </div>
                    </div>
                    
                    {/* User label */}
                    <div className="min-w-32">
                      {editingId === landmark.id ? (
                        <div className="flex items-center space-x-1">
                          <Input
                            value={editLabel}
                            onChange={(e) => setEditLabel(e.target.value)}
                            placeholder="Label..."
                            className="h-8 text-sm bg-white/10 border-white/20 text-white"
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') saveLabel();
                              if (e.key === 'Escape') cancelEditing();
                            }}
                            autoFocus
                          />
                          <Button
                            size="sm"
                            onClick={saveLabel}
                            className="h-8 w-8 p-0 bg-green-600 hover:bg-green-700"
                          >
                            <Save className="w-3 h-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={cancelEditing}
                            className="h-8 w-8 p-0 border-white/20"
                          >
                            <X className="w-3 h-3" />
                          </Button>
                        </div>
                      ) : (
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-blue-200 font-mono">
                            {landmark.userLabel || 'Unlabeled'}
                          </span>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => startEditing(landmark)}
                            className="h-6 w-6 p-0 text-gray-400 hover:text-white"
                          >
                            <Edit2 className="w-3 h-3" />
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Actions */}
                  <div className="flex items-center space-x-1">
                    {/* Stability indicator */}
                    <div className="w-16 bg-gray-700 rounded-full h-2 mr-2">
                      <div
                        className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-2 rounded-full transition-all"
                        style={{ width: `${landmark.stability * 100}%` }}
                      />
                    </div>
                    
                    {/* Delete button */}
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => deleteLandmark(landmark.id)}
                      className="h-6 w-6 p-0 text-red-400 hover:text-red-300 hover:bg-red-500/20"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : landmarks.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No landmark points detected</p>
            <p className="text-sm mt-2">Draw a letter to identify key structural points</p>
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No landmarks match the current filter</p>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setFilterType('all')}
              className="mt-2 text-white border-white/20"
            >
              Show All Types
            </Button>
          </div>
        )}
        
        {/* Summary statistics */}
        {landmarks.length > 0 && (
          <div className="mt-4 pt-4 border-t border-white/10">
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="text-gray-300">Average Stability</div>
                <div className="text-white font-bold">
                  {(landmarks.reduce((sum, l) => sum + l.stability, 0) / landmarks.length * 100).toFixed(0)}%
                </div>
              </div>
              <div className="text-center">
                <div className="text-gray-300">Most Stable Type</div>
                <div className="text-white font-bold">
                  {landmarkTypes.reduce((prev, type) => {
                    const typeStability = landmarks
                      .filter(l => l.type === type)
                      .reduce((sum, l) => sum + l.stability, 0) / landmarks.filter(l => l.type === type).length;
                    const prevStability = landmarks
                      .filter(l => l.type === prev)
                      .reduce((sum, l) => sum + l.stability, 0) / landmarks.filter(l => l.type === prev).length;
                    return typeStability > prevStability ? type : prev;
                  })}
                </div>
              </div>
              <div className="text-center">
                <div className="text-gray-300">Labeled</div>
                <div className="text-white font-bold">
                  {landmarks.filter(l => l.userLabel).length}/{landmarks.length}
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default LandmarksTable;
