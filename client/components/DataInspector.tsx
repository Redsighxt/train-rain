import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ChevronDown, ChevronRight, Copy, Eye, FileText } from 'lucide-react';
import { useResearchStore } from '@/store/researchStore';

interface DataInspectorProps {
  className?: string;
}

interface TreeNodeProps {
  label: string;
  value: any;
  level: number;
  path: string;
}

const TreeNode: React.FC<TreeNodeProps> = ({ label, value, level, path }) => {
  const [isExpanded, setIsExpanded] = useState(level < 2);
  
  const getValueType = (val: any): string => {
    if (val === null) return 'null';
    if (Array.isArray(val)) return 'array';
    return typeof val;
  };
  
  const getValuePreview = (val: any): string => {
    const type = getValueType(val);
    
    switch (type) {
      case 'string':
        return `"${val.length > 50 ? val.substring(0, 50) + '...' : val}"`;
      case 'number':
        return val.toString();
      case 'boolean':
        return val.toString();
      case 'null':
        return 'null';
      case 'array':
        return `Array(${val.length})`;
      case 'object':
        return `Object(${Object.keys(val).length} keys)`;
      default:
        return String(val);
    }
  };
  
  const getTypeColor = (type: string): string => {
    switch (type) {
      case 'string': return 'text-green-400';
      case 'number': return 'text-blue-400';
      case 'boolean': return 'text-purple-400';
      case 'null': return 'text-gray-400';
      case 'array': return 'text-orange-400';
      case 'object': return 'text-yellow-400';
      default: return 'text-white';
    }
  };
  
  const copyValue = () => {
    const textValue = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
    navigator.clipboard.writeText(textValue);
  };
  
  const isExpandable = value && (typeof value === 'object') && Object.keys(value).length > 0;
  const valueType = getValueType(value);
  const typeColor = getTypeColor(valueType);
  
  return (
    <div className="select-none">
      <div 
        className={`flex items-center py-1 px-2 hover:bg-white/5 rounded cursor-pointer ${
          level > 0 ? 'ml-' + (level * 4) : ''
        }`}
        onClick={() => isExpandable && setIsExpanded(!isExpanded)}
      >
        {isExpandable ? (
          isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400 mr-1" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400 mr-1" />
          )
        ) : (
          <div className="w-5 mr-1" />
        )}
        
        <span className="text-blue-300 font-mono mr-2">{label}:</span>
        <Badge variant="outline" className={`${typeColor} border-current/30 mr-2 text-xs`}>
          {valueType}
        </Badge>
        <span className={`${typeColor} font-mono text-sm flex-1`}>
          {getValuePreview(value)}
        </span>
        
        <Button
          size="sm"
          variant="ghost"
          onClick={(e) => {
            e.stopPropagation();
            copyValue();
          }}
          className="ml-2 h-6 w-6 p-0 text-gray-400 hover:text-white"
        >
          <Copy className="w-3 h-3" />
        </Button>
      </div>
      
      {isExpandable && isExpanded && (
        <div className="ml-4 border-l border-white/10 pl-2">
          {Array.isArray(value) ? (
            value.map((item, index) => (
              <TreeNode
                key={index}
                label={`[${index}]`}
                value={item}
                level={level + 1}
                path={`${path}[${index}]`}
              />
            ))
          ) : (
            Object.entries(value).map(([key, val]) => (
              <TreeNode
                key={key}
                label={key}
                value={val}
                level={level + 1}
                path={`${path}.${key}`}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
};

const DataInspector: React.FC<DataInspectorProps> = ({ className }) => {
  const { currentStroke, settings } = useResearchStore();
  const [selectedView, setSelectedView] = useState<'stroke' | 'analysis' | 'settings'>('analysis');
  
  const getInspectionData = () => {
    switch (selectedView) {
      case 'stroke':
        return {
          raw: currentStroke.raw,
          processed: currentStroke.processed.slice(0, 5), // Limit for performance
          landmarks: currentStroke.landmarks
        };
      case 'analysis':
        return {
          invariants: currentStroke.invariants,
          signature: currentStroke.signature
        };
      case 'settings':
        return settings;
      default:
        return {};
    }
  };
  
  const exportData = () => {
    const data = getInspectionData();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stroke-data-${selectedView}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const inspectionData = getInspectionData();
  const hasData = Object.keys(inspectionData).length > 0;
  
  return (
    <Card className={`analysis-card ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center">
            <Eye className="w-5 h-5 mr-2 text-green-400" />
            Data Inspector
          </CardTitle>
          <div className="flex items-center space-x-2">
            <Button
              size="sm"
              variant="outline"
              onClick={exportData}
              disabled={!hasData}
              className="text-white border-white/20"
            >
              <FileText className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
        
        {/* View selector */}
        <div className="flex space-x-1 bg-white/5 rounded-lg p-1">
          {[
            { key: 'stroke', label: 'Stroke Data' },
            { key: 'analysis', label: 'Analysis' },
            { key: 'settings', label: 'Settings' }
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setSelectedView(key as any)}
              className={`flex-1 px-3 py-2 text-sm rounded-md transition-all ${
                selectedView === key
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </CardHeader>
      
      <CardContent>
        {hasData ? (
          <div className="max-h-96 overflow-y-auto bg-black/20 rounded-lg p-3 border border-white/10">
            <div className="font-mono text-sm">
              {Object.entries(inspectionData).map(([key, value]) => (
                <TreeNode
                  key={key}
                  label={key}
                  value={value}
                  level={0}
                  path={key}
                />
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No data available for inspection</p>
            <p className="text-sm mt-2">Draw a letter to generate analysis data</p>
          </div>
        )}
        
        {hasData && (
          <div className="mt-4 text-xs text-gray-400">
            <div className="flex items-center justify-between">
              <span>Total objects: {Object.keys(inspectionData).length}</span>
              <span>View: {selectedView}</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default DataInspector;
