/**
 * API client for Stroke Lab Backend
 * 
 * Provides typed interfaces for all backend API endpoints
 */

export interface Point {
  x: number;
  y: number;
  time: number;
  pressure?: number;
}

export interface StrokeData {
  label: string;
  points: Point[];
  source_type?: string;
  quality_score?: number;
}

export interface TrainingRequest {
  model_name: string;
  model_type?: string;
  description?: string;
  training_config?: Record<string, any>;
  labels?: string[];
  min_quality?: number;
}

export interface TrainingStatus {
  session_id: string;
  status: string;
  current_epoch: number;
  total_epochs: number;
  current_loss?: number;
  current_accuracy?: number;
  progress_percentage: number;
}

export interface ModelInfo {
  id: number;
  name: string;
  version: string;
  model_type: string;
  description?: string;
  training_accuracy?: number;
  validation_accuracy?: number;
  is_active: boolean;
  created_at: string;
}

export interface ImportStatus {
  import_id: string;
  status: string;
  total_files: number;
  processed_files: number;
  failed_files: number;
  success_rate?: number;
  created_stroke_records: number;
  errors: string[];
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  database: Record<string, any>;
  system: Record<string, any>;
}

class APIClient {
  private baseURL: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        
        try {
          const errorData = await response.json();
          if (errorData.error && errorData.error.message) {
            errorMessage = errorData.error.message;
          } else if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // Keep the original error message if JSON parsing fails
        }
        
        throw new Error(errorMessage);
      }

      // Handle empty responses
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      } else {
        return response as unknown as T;
      }
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error(`Network error: ${error}`);
    }
  }

  // Health and System Endpoints
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/health');
  }

  async getSystemInfo(): Promise<Record<string, any>> {
    return this.request<Record<string, any>>('/api/info');
  }

  // Stroke Data Endpoints
  async createStrokeData(strokeData: StrokeData): Promise<{ success: boolean; stroke_id: number; external_id: string }> {
    return this.request('/api/strokes', {
      method: 'POST',
      body: JSON.stringify(strokeData),
    });
  }

  async listStrokeData(options: {
    skip?: number;
    limit?: number;
    label?: string;
    validated_only?: boolean;
  } = {}): Promise<{
    strokes: any[];
    total_count: number;
    skip: number;
    limit: number;
  }> {
    const params = new URLSearchParams();
    if (options.skip !== undefined) params.append('skip', options.skip.toString());
    if (options.limit !== undefined) params.append('limit', options.limit.toString());
    if (options.label) params.append('label', options.label);
    if (options.validated_only !== undefined) params.append('validated_only', options.validated_only.toString());

    return this.request(`/api/strokes?${params.toString()}`);
  }

  async getStrokeData(strokeId: number): Promise<any> {
    return this.request(`/api/strokes/${strokeId}`);
  }

  async deleteStrokeData(strokeId: number): Promise<{ success: boolean; message: string }> {
    return this.request(`/api/strokes/${strokeId}`, {
      method: 'DELETE',
    });
  }

  async getStrokeStatistics(): Promise<Record<string, any>> {
    return this.request('/api/strokes/statistics');
  }

  // Dataset Import Endpoints
  async importDatasetFiles(
    files: File[],
    sourceType: string = 'upload'
  ): Promise<{
    success: boolean;
    import_id: string;
    message: string;
    status_endpoint: string;
  }> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('source_type', sourceType);

    return this.request('/api/dataset/import', {
      method: 'POST',
      headers: {}, // Remove Content-Type to let browser set it for FormData
      body: formData,
    });
  }

  async importDatasetFromPath(request: {
    source_path: string;
    source_type?: string;
    labels_filter?: string[];
    min_quality?: number;
    import_config?: Record<string, any>;
    sample_size?: number;
  }): Promise<{
    success: boolean;
    import_id: string;
    message: string;
    status_endpoint: string;
  }> {
    return this.request('/api/dataset/import/path', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async importHandwrittenDataset(request: {
    sample_size?: number;
    labels_filter?: string[];
    min_quality?: number;
  }): Promise<{
    success: boolean;
    import_id: string;
    message: string;
    status_endpoint: string;
    dataset_info: Record<string, any>;
  }> {
    const params = new URLSearchParams();
    if (request.sample_size) params.append('sample_size', request.sample_size.toString());
    if (request.labels_filter) params.append('labels_filter', JSON.stringify(request.labels_filter));
    if (request.min_quality) params.append('min_quality', request.min_quality.toString());

    return this.request(`/api/dataset/import/handwritten?${params.toString()}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async getImportStatus(importId: string): Promise<ImportStatus> {
    return this.request(`/api/dataset/import/${importId}/status`);
  }

  // Training Endpoints
  async startTraining(request: TrainingRequest): Promise<{
    success: boolean;
    session_id: string;
    message: string;
    training_data_count: number;
    status_endpoint: string;
  }> {
    return this.request('/api/train', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getTrainingStatus(sessionId: string): Promise<TrainingStatus> {
    return this.request(`/api/train/${sessionId}/status`);
  }

  async stopTraining(sessionId: string): Promise<{ success: boolean; message: string }> {
    return this.request(`/api/train/${sessionId}/stop`, {
      method: 'POST',
    });
  }

  async listTrainingSessions(limit: number = 10): Promise<{ sessions: any[] }> {
    return this.request(`/api/train/sessions?limit=${limit}`);
  }

  async getActiveTrainingSessions(): Promise<{ active_sessions: any[]; count: number }> {
    return this.request('/api/train/active');
  }

  // Model Management Endpoints
  async listModels(options: {
    skip?: number;
    limit?: number;
    active_only?: boolean;
    model_type?: string;
  } = {}): Promise<{ models: ModelInfo[]; count: number }> {
    const params = new URLSearchParams();
    if (options.skip !== undefined) params.append('skip', options.skip.toString());
    if (options.limit !== undefined) params.append('limit', options.limit.toString());
    if (options.active_only !== undefined) params.append('active_only', options.active_only.toString());
    if (options.model_type) params.append('model_type', options.model_type);

    return this.request(`/api/models?${params.toString()}`);
  }

  async getModel(modelId: number): Promise<ModelInfo> {
    return this.request(`/api/models/${modelId}`);
  }

  async getBestModel(metric: string = 'validation_accuracy'): Promise<ModelInfo> {
    return this.request(`/api/models/best?metric=${metric}`);
  }

  async deleteModel(modelId: number): Promise<{ success: boolean; message: string }> {
    return this.request(`/api/models/${modelId}`, {
      method: 'DELETE',
    });
  }

  async cleanupOldModels(keepLatest: number = 5): Promise<{
    success: boolean;
    deactivated_count: number;
    message: string;
  }> {
    return this.request('/api/models/cleanup', {
      method: 'POST',
      body: JSON.stringify({ keep_latest: keepLatest }),
    });
  }

  // Model Export Endpoints
  async exportModel(modelId: number): Promise<Response> {
    const url = `${this.baseURL}/api/models/${modelId}/export`;
    const response = await fetch(url, {
      headers: this.defaultHeaders,
    });
    
    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`);
    }
    
    return response;
  }

  async getModelExportMetadata(modelId: number): Promise<{
    model_id: number;
    model_name: string;
    model_version: string;
    export_format: string;
    estimated_size_bytes: number;
    estimated_size_mb: number;
    includes_model_file: boolean;
    model_file_path: string;
    export_available: boolean;
    performance_summary: Record<string, any>;
    export_urls: Record<string, string>;
  }> {
    return this.request(`/api/models/${modelId}/export/metadata`);
  }

  async exportModelsBatch(
    modelIds: number[],
    format: string = 'json'
  ): Promise<any> {
    return this.request('/api/models/export/batch', {
      method: 'POST',
      body: JSON.stringify({ model_ids: modelIds, format }),
    });
  }

  // Utility Methods
  async downloadModelExport(modelId: number, filename?: string): Promise<void> {
    try {
      const response = await this.exportModel(modelId);
      
      // Get filename from response headers or use provided filename
      const contentDisposition = response.headers.get('Content-Disposition');
      let downloadFilename = filename;
      
      if (!downloadFilename && contentDisposition) {
        const matches = contentDisposition.match(/filename="?([^"]+)"?/);
        if (matches) {
          downloadFilename = matches[1];
        }
      }
      
      if (!downloadFilename) {
        downloadFilename = `model_${modelId}_export.pth`;
      }

      // Create blob and download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = downloadFilename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      throw new Error(`Download failed: ${error}`);
    }
  }

  // Real-time polling utilities
  async pollTrainingProgress(
    sessionId: string,
    onProgress: (status: TrainingStatus) => void,
    onComplete: (finalStatus: TrainingStatus) => void,
    onError: (error: Error) => void,
    pollInterval: number = 1000
  ): Promise<() => void> {
    let isPolling = true;
    
    const poll = async () => {
      try {
        while (isPolling) {
          const status = await this.getTrainingStatus(sessionId);
          onProgress(status);
          
          // Check if training is complete
          if (status.status === 'completed' || status.status === 'failed' || status.status === 'stopped') {
            onComplete(status);
            break;
          }
          
          // Wait before next poll
          await new Promise(resolve => setTimeout(resolve, pollInterval));
        }
      } catch (error) {
        onError(error instanceof Error ? error : new Error(String(error)));
      }
    };
    
    // Start polling
    poll();
    
    // Return stop function
    return () => {
      isPolling = false;
    };
  }

  async pollImportProgress(
    importId: string,
    onProgress: (status: ImportStatus) => void,
    onComplete: (finalStatus: ImportStatus) => void,
    onError: (error: Error) => void,
    pollInterval: number = 2000
  ): Promise<() => void> {
    let isPolling = true;
    
    const poll = async () => {
      try {
        while (isPolling) {
          const status = await this.getImportStatus(importId);
          onProgress(status);
          
          // Check if import is complete
          if (status.status === 'completed' || status.success_rate !== undefined) {
            onComplete(status);
            break;
          }
          
          // Wait before next poll
          await new Promise(resolve => setTimeout(resolve, pollInterval));
        }
      } catch (error) {
        onError(error instanceof Error ? error : new Error(String(error)));
      }
    };
    
    // Start polling
    poll();
    
    // Return stop function
    return () => {
      isPolling = false;
    };
  }

  // Connection testing
  async testConnection(): Promise<boolean> {
    try {
      await this.getHealth();
      return true;
    } catch {
      return false;
    }
  }
}

// Create singleton instance
export const apiClient = new APIClient(
  import.meta.env.VITE_API_URL || 'http://localhost:8000'
);

// Hook for React components
export function useAPI() {
  return apiClient;
}

// Error handling utilities
export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export function isAPIError(error: unknown): error is APIError {
  return error instanceof APIError;
}

// Type guards
export function isTrainingComplete(status: TrainingStatus): boolean {
  return ['completed', 'failed', 'stopped'].includes(status.status);
}

export function isImportComplete(status: ImportStatus): boolean {
  return status.status === 'completed' || status.success_rate !== undefined;
}
