import { subscribeWithSelector } from 'zustand/middleware';
import { useResearchStore } from './researchStore';
import { analyzeStroke, defaultAnalysisConfig } from '@/lib/strokeAnalysisProcessor';
import type { Stroke, ProcessedPoint, LandmarkPoint, InvariantMetrics, ThreeDSignature } from './researchStore';

/**
 * Analysis Processor Integration
 * Connects the research store with the comprehensive analysis pipeline
 */

// Subscribe to store changes and trigger analysis
useResearchStore.subscribe(
  (state) => state.currentStroke.raw,
  async (stroke, previousStroke) => {
    if (!stroke || stroke === previousStroke) return;
    
    const state = useResearchStore.getState();
    
    try {
      // Set processing state
      state.setProcessingStage('Initializing analysis...');
      
      // Configure analysis based on settings
      const analysisConfig = {
        ...defaultAnalysisConfig,
        enableAffineAnalysis: state.settings.enableAffineAnalysis,
        enableTopologicalAnalysis: state.settings.enableTopologicalAnalysis,
        enablePathSignature: state.settings.enablePathSignature,
        enableSpectralAnalysis: state.settings.enableSpectralAnalysis,
        maxPoints: state.settings.numPoints,
        targetProcessingTime: 150, // 150ms for interactive use
        adaptiveQuality: true,
        cacheResults: true
      };
      
      state.setProcessingStage('Running mathematical analysis...');
      
      // Perform comprehensive analysis
      const result = await analyzeStroke(stroke, analysisConfig);
      
      // Update store with results
      useResearchStore.setState((currentState) => ({
        currentStroke: {
          ...currentState.currentStroke,
          processed: result.processed,
          landmarks: result.landmarks,
          invariants: result.invariants
        },
        isProcessing: false,
        processingStage: ''
      }));
      
      // Generate 3D signature if we have invariants
      if (result.invariants) {
        state.setProcessingStage('Generating 3D signature...');
        
        // Import the generate3DSignature function from signature-math
        const { generate3DSignature } = await import('@/lib/signature-math');
        
        // Create stroke data in the format expected by generate3DSignature
        const strokeForSignature = {
          id: stroke.id,
          points: stroke.points,
          completed: stroke.completed,
          color: stroke.color
        };
        
        // Create invariant points in the format expected by generate3DSignature
        const invariantPointsForSignature = result.landmarks.map((landmark, index) => ({
          id: `inv-${index}`,
          index: Math.floor(landmark.position.x * result.processed.length),
          strokeIndex: 0,
          position: landmark.position,
          stabilityScore: landmark.stability,
          type: 'primary' as const,
          category: 'geometric' as const,
          description: landmark.type,
          confidence: landmark.stability
        }));
        
        // Generate 3D signature using the proper function
        const signature = generate3DSignature([strokeForSignature], invariantPointsForSignature);
        
        // Update store with 3D signature
        useResearchStore.setState((currentState) => ({
          currentStroke: {
            ...currentState.currentStroke,
            signature
          },
          isProcessing: false,
          processingStage: ''
        }));
      }
      
      console.log(`Analysis completed in ${result.processingTime.toFixed(1)}ms with quality ${(result.qualityScore * 100).toFixed(0)}%`);
      
    } catch (error) {
      console.error('Analysis failed:', error);
      
      useResearchStore.setState((currentState) => ({
        isProcessing: false,
        processingStage: 'Analysis failed'
      }));
      
      // Clear error after a delay
      setTimeout(() => {
        useResearchStore.setState((currentState) => ({
          processingStage: ''
        }));
      }, 3000);
    }
  }
);

export default {};
