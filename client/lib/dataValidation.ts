/**
 * Data format validation utilities for frontend-backend consistency
 * 
 * Ensures that data exchanged between frontend and backend follows
 * the correct schema and validates data integrity.
 */

import { Point, StrokeData } from './api';

// Schema validation interfaces
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

export interface StrokeValidationOptions {
  minPoints?: number;
  maxPoints?: number;
  requireMonotonicTime?: boolean;
  requireValidPressure?: boolean;
  allowNegativeCoordinates?: boolean;
  maxCoordinateValue?: number;
}

// Default validation options
const DEFAULT_STROKE_OPTIONS: Required<StrokeValidationOptions> = {
  minPoints: 2,
  maxPoints: 1000,
  requireMonotonicTime: true,
  requireValidPressure: false,
  allowNegativeCoordinates: true,
  maxCoordinateValue: 10000
};

/**
 * Validates a point object structure and values
 */
export function validatePoint(point: any, index?: number): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  const prefix = index !== undefined ? `Point ${index}: ` : 'Point: ';
  
  // Check required fields
  if (typeof point !== 'object' || point === null) {
    errors.push(`${prefix}Point must be an object`);
    return { isValid: false, errors, warnings };
  }
  
  // Validate x coordinate
  if (typeof point.x !== 'number') {
    errors.push(`${prefix}x coordinate must be a number`);
  } else if (!isFinite(point.x)) {
    errors.push(`${prefix}x coordinate must be finite`);
  }
  
  // Validate y coordinate
  if (typeof point.y !== 'number') {
    errors.push(`${prefix}y coordinate must be a number`);
  } else if (!isFinite(point.y)) {
    errors.push(`${prefix}y coordinate must be finite`);
  }
  
  // Validate time
  if (typeof point.time !== 'number') {
    errors.push(`${prefix}time must be a number`);
  } else if (!isFinite(point.time)) {
    errors.push(`${prefix}time must be finite`);
  } else if (point.time < 0) {
    warnings.push(`${prefix}time is negative`);
  }
  
  // Validate optional pressure
  if (point.pressure !== undefined) {
    if (typeof point.pressure !== 'number') {
      errors.push(`${prefix}pressure must be a number`);
    } else if (!isFinite(point.pressure)) {
      errors.push(`${prefix}pressure must be finite`);
    } else if (point.pressure < 0 || point.pressure > 1) {
      warnings.push(`${prefix}pressure should be between 0 and 1`);
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Validates a stroke data object
 */
export function validateStrokeData(
  strokeData: any, 
  options: StrokeValidationOptions = {}
): ValidationResult {
  const opts = { ...DEFAULT_STROKE_OPTIONS, ...options };
  const errors: string[] = [];
  const warnings: string[] = [];
  
  // Check if strokeData is an object
  if (typeof strokeData !== 'object' || strokeData === null) {
    errors.push('Stroke data must be an object');
    return { isValid: false, errors, warnings };
  }
  
  // Validate label
  if (typeof strokeData.label !== 'string') {
    errors.push('Label must be a string');
  } else if (strokeData.label.length === 0) {
    errors.push('Label cannot be empty');
  } else if (strokeData.label.length > 10) {
    warnings.push('Label is unusually long (>10 characters)');
  }
  
  // Validate points array
  if (!Array.isArray(strokeData.points)) {
    errors.push('Points must be an array');
    return { isValid: false, errors, warnings };
  }
  
  // Check points array length
  if (strokeData.points.length < opts.minPoints) {
    errors.push(`Stroke must have at least ${opts.minPoints} points`);
  }
  
  if (strokeData.points.length > opts.maxPoints) {
    errors.push(`Stroke cannot have more than ${opts.maxPoints} points`);
  }
  
  // Validate each point
  let previousTime: number | null = null;
  for (let i = 0; i < strokeData.points.length; i++) {
    const pointResult = validatePoint(strokeData.points[i], i);
    errors.push(...pointResult.errors);
    warnings.push(...pointResult.warnings);
    
    if (pointResult.isValid) {
      const point = strokeData.points[i];
      
      // Check coordinate bounds
      if (!opts.allowNegativeCoordinates) {
        if (point.x < 0 || point.y < 0) {
          warnings.push(`Point ${i}: Negative coordinates not recommended`);
        }
      }
      
      if (Math.abs(point.x) > opts.maxCoordinateValue || Math.abs(point.y) > opts.maxCoordinateValue) {
        warnings.push(`Point ${i}: Coordinates are very large`);
      }
      
      // Check time monotonicity
      if (opts.requireMonotonicTime && previousTime !== null) {
        if (point.time < previousTime) {
          errors.push(`Point ${i}: Time must be monotonically increasing`);
        } else if (point.time === previousTime) {
          warnings.push(`Point ${i}: Duplicate timestamp`);
        }
      }
      
      previousTime = point.time;
      
      // Check pressure if required
      if (opts.requireValidPressure && point.pressure === undefined) {
        warnings.push(`Point ${i}: Missing pressure value`);
      }
    }
  }
  
  // Validate optional fields
  if (strokeData.source_type !== undefined) {
    if (typeof strokeData.source_type !== 'string') {
      warnings.push('source_type should be a string');
    }
  }
  
  if (strokeData.quality_score !== undefined) {
    if (typeof strokeData.quality_score !== 'number') {
      warnings.push('quality_score should be a number');
    } else if (strokeData.quality_score < 0 || strokeData.quality_score > 1) {
      warnings.push('quality_score should be between 0 and 1');
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Validates training request data
 */
export function validateTrainingRequest(request: any): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  if (typeof request !== 'object' || request === null) {
    errors.push('Training request must be an object');
    return { isValid: false, errors, warnings };
  }
  
  // Validate model_name
  if (typeof request.model_name !== 'string') {
    errors.push('model_name must be a string');
  } else if (request.model_name.length === 0) {
    errors.push('model_name cannot be empty');
  }
  
  // Validate optional fields
  if (request.model_type !== undefined) {
    if (typeof request.model_type !== 'string') {
      errors.push('model_type must be a string');
    } else {
      const validTypes = ['cnn', 'transformer', 'hybrid', 'deep_cnn'];
      if (!validTypes.includes(request.model_type.toLowerCase())) {
        warnings.push(`Unknown model_type: ${request.model_type}`);
      }
    }
  }
  
  if (request.labels !== undefined) {
    if (!Array.isArray(request.labels)) {
      errors.push('labels must be an array');
    } else {
      for (let i = 0; i < request.labels.length; i++) {
        if (typeof request.labels[i] !== 'string') {
          errors.push(`labels[${i}] must be a string`);
        }
      }
    }
  }
  
  if (request.min_quality !== undefined) {
    if (typeof request.min_quality !== 'number') {
      errors.push('min_quality must be a number');
    } else if (request.min_quality < 0 || request.min_quality > 1) {
      warnings.push('min_quality should be between 0 and 1');
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Normalizes stroke data to ensure consistency
 */
export function normalizeStrokeData(strokeData: any): StrokeData {
  // Create a clean copy
  const normalized: StrokeData = {
    label: String(strokeData.label || '').toUpperCase(),
    points: [],
    source_type: strokeData.source_type || 'manual',
    quality_score: strokeData.quality_score
  };
  
  // Normalize points
  if (Array.isArray(strokeData.points)) {
    let timeOffset = 0;
    const firstTime = strokeData.points[0]?.time;
    
    normalized.points = strokeData.points.map((point: any, index: number) => {
      const normalizedPoint: Point = {
        x: Number(point.x) || 0,
        y: Number(point.y) || 0,
        time: Number(point.time) || (firstTime + index * 10), // Default 10ms intervals
      };
      
      if (point.pressure !== undefined) {
        normalizedPoint.pressure = Math.max(0, Math.min(1, Number(point.pressure)));
      }
      
      return normalizedPoint;
    });
    
    // Ensure monotonic time
    for (let i = 1; i < normalized.points.length; i++) {
      if (normalized.points[i].time <= normalized.points[i - 1].time) {
        normalized.points[i].time = normalized.points[i - 1].time + 1;
      }
    }
  }
  
  return normalized;
}

/**
 * Validates data format consistency between frontend and backend responses
 */
export function validateAPIResponse(response: any, expectedType: string): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  if (typeof response !== 'object' || response === null) {
    errors.push('API response must be an object');
    return { isValid: false, errors, warnings };
  }
  
  switch (expectedType) {
    case 'training_status':
      if (typeof response.session_id !== 'string') {
        errors.push('session_id must be a string');
      }
      if (typeof response.status !== 'string') {
        errors.push('status must be a string');
      }
      if (typeof response.progress_percentage !== 'number') {
        errors.push('progress_percentage must be a number');
      }
      break;
      
    case 'model_info':
      if (typeof response.id !== 'number') {
        errors.push('id must be a number');
      }
      if (typeof response.name !== 'string') {
        errors.push('name must be a string');
      }
      if (typeof response.version !== 'string') {
        errors.push('version must be a string');
      }
      break;
      
    case 'health_response':
      if (typeof response.status !== 'string') {
        errors.push('status must be a string');
      }
      if (typeof response.timestamp !== 'string') {
        errors.push('timestamp must be a string');
      }
      break;
      
    default:
      warnings.push(`Unknown expected type: ${expectedType}`);
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Converts stroke data from frontend format to backend format
 */
export function toBackendFormat(strokeData: StrokeData): any {
  return {
    label: strokeData.label,
    points: strokeData.points.map(point => ({
      x: point.x,
      y: point.y,
      time: point.time,
      ...(point.pressure !== undefined && { pressure: point.pressure })
    })),
    source_type: strokeData.source_type || 'manual',
    ...(strokeData.quality_score !== undefined && { quality_score: strokeData.quality_score })
  };
}

/**
 * Converts stroke data from backend format to frontend format
 */
export function fromBackendFormat(backendData: any): StrokeData {
  return normalizeStrokeData(backendData);
}

/**
 * Comprehensive data validation for a complete dataset
 */
export function validateDataset(dataset: any[]): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  if (!Array.isArray(dataset)) {
    errors.push('Dataset must be an array');
    return { isValid: false, errors, warnings };
  }
  
  if (dataset.length === 0) {
    warnings.push('Dataset is empty');
    return { isValid: true, errors, warnings };
  }
  
  // Validate each stroke in the dataset
  const labelCounts: Record<string, number> = {};
  
  for (let i = 0; i < dataset.length; i++) {
    const strokeResult = validateStrokeData(dataset[i]);
    
    // Add context to errors
    strokeResult.errors.forEach(error => {
      errors.push(`Dataset[${i}]: ${error}`);
    });
    
    strokeResult.warnings.forEach(warning => {
      warnings.push(`Dataset[${i}]: ${warning}`);
    });
    
    // Count labels for balance analysis
    if (strokeResult.isValid && dataset[i].label) {
      const label = dataset[i].label;
      labelCounts[label] = (labelCounts[label] || 0) + 1;
    }
  }
  
  // Analyze label distribution
  const labels = Object.keys(labelCounts);
  if (labels.length > 0) {
    const counts = Object.values(labelCounts);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);
    const avgCount = counts.reduce((a, b) => a + b, 0) / counts.length;
    
    // Check for class imbalance
    if (maxCount > minCount * 3) {
      warnings.push(`Class imbalance detected: ${labels[counts.indexOf(maxCount)]} has ${maxCount} samples while ${labels[counts.indexOf(minCount)]} has ${minCount}`);
    }
    
    // Check for minimum samples per class
    if (minCount < 5) {
      warnings.push(`Some classes have very few samples (minimum: ${minCount})`);
    }
  }
  
  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Format validation errors and warnings for display
 */
export function formatValidationResults(result: ValidationResult): string {
  const messages: string[] = [];
  
  if (result.errors.length > 0) {
    messages.push('❌ Errors:');
    result.errors.forEach(error => messages.push(`  • ${error}`));
  }
  
  if (result.warnings.length > 0) {
    messages.push('⚠️ Warnings:');
    result.warnings.forEach(warning => messages.push(`  • ${warning}`));
  }
  
  if (result.isValid && result.warnings.length === 0) {
    messages.push('✅ Data validation passed');
  }
  
  return messages.join('\n');
}

// Export validation utilities
export const dataValidation = {
  validatePoint,
  validateStrokeData,
  validateTrainingRequest,
  validateAPIResponse,
  validateDataset,
  normalizeStrokeData,
  toBackendFormat,
  fromBackendFormat,
  formatValidationResults
};
