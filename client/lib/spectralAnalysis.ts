import type { Point } from '@/store/researchStore';

/**
 * Spectral Analysis Module
 * Implements FFT analysis, wavelet transforms, and frequency domain features
 * for advanced stroke signature characterization
 */

export interface SpectralMetrics {
  fourierMagnitudes: number[];
  fourierPhases: number[];
  powerSpectrum: number[];
  spectralCentroid: number;
  spectralRolloff: number;
  spectralFlux: number;
  spectralKurtosis: number;
  dominantFrequencies: number[];
  waveletCoefficients: WaveletData;
  mfccCoefficients: number[];
  spectralComplexity: number;
}

export interface WaveletData {
  approximation: number[];
  details: number[][];
  scales: number[];
  energyDistribution: number[];
  waveletEntropy: number;
}

export interface FrequencyFeatures {
  lowFrequencyEnergy: number;
  midFrequencyEnergy: number;
  highFrequencyEnergy: number;
  frequencySpread: number;
  harmonicRatio: number;
  noiseRatio: number;
}

/**
 * Fast Fourier Transform implementation
 */
export function fft(signal: number[]): { real: number[]; imag: number[] } {
  const n = signal.length;
  
  // Ensure power of 2 for FFT
  const paddedLength = Math.pow(2, Math.ceil(Math.log2(n)));
  const paddedSignal = [...signal, ...new Array(paddedLength - n).fill(0)];
  
  return fftRecursive(paddedSignal.map(x => ({ real: x, imag: 0 })));
}

/**
 * Recursive FFT implementation
 */
function fftRecursive(signal: Array<{ real: number; imag: number }>): { real: number[]; imag: number[] } {
  const n = signal.length;
  
  if (n <= 1) {
    return {
      real: signal.map(x => x.real),
      imag: signal.map(x => x.imag)
    };
  }
  
  // Divide
  const even = signal.filter((_, i) => i % 2 === 0);
  const odd = signal.filter((_, i) => i % 2 === 1);
  
  // Conquer
  const evenFFT = fftRecursive(even);
  const oddFFT = fftRecursive(odd);
  
  // Combine
  const real = new Array(n);
  const imag = new Array(n);
  
  for (let k = 0; k < n / 2; k++) {
    const angle = -2 * Math.PI * k / n;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    
    const tReal = cos * oddFFT.real[k] - sin * oddFFT.imag[k];
    const tImag = sin * oddFFT.real[k] + cos * oddFFT.imag[k];
    
    real[k] = evenFFT.real[k] + tReal;
    imag[k] = evenFFT.imag[k] + tImag;
    real[k + n / 2] = evenFFT.real[k] - tReal;
    imag[k + n / 2] = evenFFT.imag[k] - tImag;
  }
  
  return { real, imag };
}

/**
 * Calculate power spectrum from FFT
 */
export function calculatePowerSpectrum(fftResult: { real: number[]; imag: number[] }): number[] {
  const powerSpectrum: number[] = [];
  
  for (let i = 0; i < fftResult.real.length; i++) {
    const magnitude = Math.sqrt(fftResult.real[i] ** 2 + fftResult.imag[i] ** 2);
    powerSpectrum.push(magnitude ** 2);
  }
  
  return powerSpectrum;
}

/**
 * Calculate spectral centroid (center of mass of spectrum)
 */
export function calculateSpectralCentroid(powerSpectrum: number[]): number {
  let numerator = 0;
  let denominator = 0;
  
  for (let i = 0; i < powerSpectrum.length; i++) {
    numerator += i * powerSpectrum[i];
    denominator += powerSpectrum[i];
  }
  
  return denominator > 0 ? numerator / denominator : 0;
}

/**
 * Calculate spectral rolloff (frequency below which 85% of energy lies)
 */
export function calculateSpectralRolloff(powerSpectrum: number[], threshold: number = 0.85): number {
  const totalEnergy = powerSpectrum.reduce((sum, power) => sum + power, 0);
  const targetEnergy = totalEnergy * threshold;
  
  let cumulativeEnergy = 0;
  for (let i = 0; i < powerSpectrum.length; i++) {
    cumulativeEnergy += powerSpectrum[i];
    if (cumulativeEnergy >= targetEnergy) {
      return i / powerSpectrum.length;
    }
  }
  
  return 1.0;
}

/**
 * Calculate spectral flux (measure of rate of change in spectrum)
 */
export function calculateSpectralFlux(powerSpectrum: number[], previousSpectrum?: number[]): number {
  if (!previousSpectrum || previousSpectrum.length !== powerSpectrum.length) {
    return 0;
  }
  
  let flux = 0;
  for (let i = 0; i < powerSpectrum.length; i++) {
    const diff = powerSpectrum[i] - previousSpectrum[i];
    flux += Math.max(0, diff); // Half-wave rectification
  }
  
  return flux;
}

/**
 * Calculate spectral kurtosis (measure of spectrum peakiness)
 */
export function calculateSpectralKurtosis(powerSpectrum: number[]): number {
  const mean = powerSpectrum.reduce((sum, x) => sum + x, 0) / powerSpectrum.length;
  const variance = powerSpectrum.reduce((sum, x) => sum + (x - mean) ** 2, 0) / powerSpectrum.length;
  
  if (variance === 0) return 0;
  
  const fourthMoment = powerSpectrum.reduce((sum, x) => sum + (x - mean) ** 4, 0) / powerSpectrum.length;
  
  return fourthMoment / (variance ** 2) - 3;
}

/**
 * Find dominant frequencies in the spectrum
 */
export function findDominantFrequencies(powerSpectrum: number[], numPeaks: number = 5): number[] {
  const peaks: Array<{ index: number; magnitude: number }> = [];
  
  // Find local maxima
  for (let i = 1; i < powerSpectrum.length - 1; i++) {
    if (powerSpectrum[i] > powerSpectrum[i - 1] && 
        powerSpectrum[i] > powerSpectrum[i + 1] &&
        powerSpectrum[i] > 0.01) { // Minimum threshold
      peaks.push({ index: i, magnitude: powerSpectrum[i] });
    }
  }
  
  // Sort by magnitude and take top peaks
  peaks.sort((a, b) => b.magnitude - a.magnitude);
  return peaks.slice(0, numPeaks).map(peak => peak.index / powerSpectrum.length);
}

/**
 * Discrete Wavelet Transform (Haar wavelet)
 */
export function calculateWaveletTransform(signal: number[], levels: number = 4): WaveletData {
  const result: WaveletData = {
    approximation: [...signal],
    details: [],
    scales: [],
    energyDistribution: [],
    waveletEntropy: 0
  };
  
  let current = [...signal];
  
  for (let level = 0; level < levels; level++) {
    const { approximation, detail } = haarWaveletStep(current);
    
    result.details.push([...detail]);
    result.scales.push(Math.pow(2, level + 1));
    current = approximation;
  }
  
  result.approximation = current;
  
  // Calculate energy distribution across scales
  const totalEnergy = signal.reduce((sum, x) => sum + x * x, 0);
  
  result.energyDistribution = result.details.map(detail => {
    const energy = detail.reduce((sum, x) => sum + x * x, 0);
    return totalEnergy > 0 ? energy / totalEnergy : 0;
  });
  
  // Calculate wavelet entropy
  result.waveletEntropy = calculateWaveletEntropy(result.energyDistribution);
  
  return result;
}

/**
 * Single step of Haar wavelet transform
 */
function haarWaveletStep(signal: number[]): { approximation: number[]; detail: number[] } {
  const n = signal.length;
  const halfN = Math.floor(n / 2);
  
  const approximation: number[] = [];
  const detail: number[] = [];
  
  for (let i = 0; i < halfN; i++) {
    const even = signal[2 * i] || 0;
    const odd = signal[2 * i + 1] || 0;
    
    approximation.push((even + odd) / Math.sqrt(2));
    detail.push((even - odd) / Math.sqrt(2));
  }
  
  return { approximation, detail };
}

/**
 * Calculate wavelet entropy
 */
function calculateWaveletEntropy(energyDistribution: number[]): number {
  let entropy = 0;
  const total = energyDistribution.reduce((sum, e) => sum + e, 0);
  
  if (total === 0) return 0;
  
  for (const energy of energyDistribution) {
    if (energy > 0) {
      const probability = energy / total;
      entropy -= probability * Math.log2(probability);
    }
  }
  
  return entropy;
}

/**
 * Calculate Mel-Frequency Cepstral Coefficients (MFCC)
 */
export function calculateMFCC(powerSpectrum: number[], numCoefficients: number = 13): number[] {
  const numFilters = 26;
  const melFilters = createMelFilterBank(powerSpectrum.length, numFilters);
  
  // Apply mel filters
  const melEnergies: number[] = [];
  for (const filter of melFilters) {
    let energy = 0;
    for (let i = 0; i < powerSpectrum.length; i++) {
      energy += powerSpectrum[i] * filter[i];
    }
    melEnergies.push(Math.log(Math.max(energy, 1e-10))); // Avoid log(0)
  }
  
  // Apply DCT to get cepstral coefficients
  const mfcc: number[] = [];
  for (let i = 0; i < numCoefficients; i++) {
    let coefficient = 0;
    for (let j = 0; j < melEnergies.length; j++) {
      coefficient += melEnergies[j] * Math.cos(Math.PI * i * (j + 0.5) / melEnergies.length);
    }
    mfcc.push(coefficient);
  }
  
  return mfcc;
}

/**
 * Create mel filter bank
 */
function createMelFilterBank(fftSize: number, numFilters: number): number[][] {
  const melFilters: number[][] = [];
  
  // Convert to mel scale
  const melLow = 0;
  const melHigh = 2595 * Math.log10(1 + (fftSize / 2) / 700);
  
  const melPoints = [];
  for (let i = 0; i <= numFilters + 1; i++) {
    const mel = melLow + (melHigh - melLow) * i / (numFilters + 1);
    const freq = 700 * (Math.pow(10, mel / 2595) - 1);
    melPoints.push(Math.floor(freq * fftSize / (fftSize / 2)));
  }
  
  // Create triangular filters
  for (let i = 1; i <= numFilters; i++) {
    const filter = new Array(fftSize).fill(0);
    
    const left = melPoints[i - 1];
    const center = melPoints[i];
    const right = melPoints[i + 1];
    
    // Rising edge
    for (let j = left; j < center; j++) {
      if (j < fftSize) {
        filter[j] = (j - left) / (center - left);
      }
    }
    
    // Falling edge
    for (let j = center; j < right; j++) {
      if (j < fftSize) {
        filter[j] = (right - j) / (right - center);
      }
    }
    
    melFilters.push(filter);
  }
  
  return melFilters;
}

/**
 * Extract frequency domain features
 */
export function extractFrequencyFeatures(powerSpectrum: number[]): FrequencyFeatures {
  const totalEnergy = powerSpectrum.reduce((sum, power) => sum + power, 0);
  
  if (totalEnergy === 0) {
    return {
      lowFrequencyEnergy: 0,
      midFrequencyEnergy: 0,
      highFrequencyEnergy: 0,
      frequencySpread: 0,
      harmonicRatio: 0,
      noiseRatio: 0
    };
  }
  
  const n = powerSpectrum.length;
  const lowBand = Math.floor(n * 0.1);
  const midBand = Math.floor(n * 0.5);
  
  // Energy in different frequency bands
  const lowFrequencyEnergy = powerSpectrum.slice(0, lowBand).reduce((sum, p) => sum + p, 0) / totalEnergy;
  const midFrequencyEnergy = powerSpectrum.slice(lowBand, midBand).reduce((sum, p) => sum + p, 0) / totalEnergy;
  const highFrequencyEnergy = powerSpectrum.slice(midBand).reduce((sum, p) => sum + p, 0) / totalEnergy;
  
  // Frequency spread (variance of spectrum)
  const centroid = calculateSpectralCentroid(powerSpectrum);
  let frequencySpread = 0;
  for (let i = 0; i < powerSpectrum.length; i++) {
    frequencySpread += (i - centroid) ** 2 * powerSpectrum[i];
  }
  frequencySpread = Math.sqrt(frequencySpread / totalEnergy);
  
  // Harmonic ratio (ratio of harmonic peaks to total energy)
  const dominantFreqs = findDominantFrequencies(powerSpectrum, 10);
  const harmonicEnergy = dominantFreqs.reduce((sum, freq) => {
    const index = Math.floor(freq * powerSpectrum.length);
    return sum + (powerSpectrum[index] || 0);
  }, 0);
  const harmonicRatio = harmonicEnergy / totalEnergy;
  
  // Noise ratio (complement of harmonic ratio)
  const noiseRatio = 1 - harmonicRatio;
  
  return {
    lowFrequencyEnergy,
    midFrequencyEnergy,
    highFrequencyEnergy,
    frequencySpread,
    harmonicRatio,
    noiseRatio
  };
}

/**
 * Calculate spectral complexity
 */
export function calculateSpectralComplexity(spectralMetrics: SpectralMetrics): number {
  // Combine multiple spectral measures
  const centroidNormalized = spectralMetrics.spectralCentroid / spectralMetrics.fourierMagnitudes.length;
  const rolloffNormalized = spectralMetrics.spectralRolloff;
  const kurtosisNormalized = Math.min(1, Math.abs(spectralMetrics.spectralKurtosis) / 10);
  const waveletEntropy = spectralMetrics.waveletCoefficients.waveletEntropy;
  
  const dominantFreqComplexity = spectralMetrics.dominantFrequencies.length / 10;
  
  return (
    centroidNormalized * 0.2 +
    rolloffNormalized * 0.2 +
    kurtosisNormalized * 0.2 +
    waveletEntropy * 0.2 +
    dominantFreqComplexity * 0.2
  );
}

/**
 * Main function to compute spectral analysis
 */
export function computeSpectralMetrics(tangentAngles: number[]): SpectralMetrics {
  if (tangentAngles.length === 0) {
    return {
      fourierMagnitudes: [],
      fourierPhases: [],
      powerSpectrum: [],
      spectralCentroid: 0,
      spectralRolloff: 0,
      spectralFlux: 0,
      spectralKurtosis: 0,
      dominantFrequencies: [],
      waveletCoefficients: {
        approximation: [],
        details: [],
        scales: [],
        energyDistribution: [],
        waveletEntropy: 0
      },
      mfccCoefficients: [],
      spectralComplexity: 0
    };
  }
  
  // Perform FFT
  const fftResult = fft(tangentAngles);
  const powerSpectrum = calculatePowerSpectrum(fftResult);
  
  // Calculate magnitudes and phases
  const fourierMagnitudes = fftResult.real.map((real, i) => 
    Math.sqrt(real ** 2 + fftResult.imag[i] ** 2)
  );
  const fourierPhases = fftResult.real.map((real, i) => 
    Math.atan2(fftResult.imag[i], real)
  );
  
  // Calculate spectral features
  const spectralCentroid = calculateSpectralCentroid(powerSpectrum);
  const spectralRolloff = calculateSpectralRolloff(powerSpectrum);
  const spectralFlux = 0; // Would need previous frame for comparison
  const spectralKurtosis = calculateSpectralKurtosis(powerSpectrum);
  const dominantFrequencies = findDominantFrequencies(powerSpectrum);
  
  // Wavelet analysis
  const waveletCoefficients = calculateWaveletTransform(tangentAngles);
  
  // MFCC analysis
  const mfccCoefficients = calculateMFCC(powerSpectrum);
  
  const metrics = {
    fourierMagnitudes,
    fourierPhases,
    powerSpectrum,
    spectralCentroid,
    spectralRolloff,
    spectralFlux,
    spectralKurtosis,
    dominantFrequencies,
    waveletCoefficients,
    mfccCoefficients,
    spectralComplexity: 0 // Will be calculated below
  };
  
  // Calculate overall spectral complexity
  metrics.spectralComplexity = calculateSpectralComplexity(metrics);
  
  return metrics;
}

/**
 * Compare spectral signatures for similarity
 */
export function compareSpectralSignatures(
  metrics1: SpectralMetrics, 
  metrics2: SpectralMetrics
): number {
  // Compare power spectra using normalized cross-correlation
  const spec1 = metrics1.powerSpectrum;
  const spec2 = metrics2.powerSpectrum;
  
  if (spec1.length === 0 || spec2.length === 0) return 0;
  
  const minLength = Math.min(spec1.length, spec2.length);
  let correlation = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < minLength; i++) {
    correlation += spec1[i] * spec2[i];
    norm1 += spec1[i] ** 2;
    norm2 += spec2[i] ** 2;
  }
  
  const normalizer = Math.sqrt(norm1 * norm2);
  const spectralSimilarity = normalizer > 0 ? correlation / normalizer : 0;
  
  // Compare MFCC coefficients
  const mfcc1 = metrics1.mfccCoefficients;
  const mfcc2 = metrics2.mfccCoefficients;
  
  let mfccSimilarity = 0;
  if (mfcc1.length > 0 && mfcc2.length > 0) {
    const minMfccLength = Math.min(mfcc1.length, mfcc2.length);
    let mfccDistance = 0;
    
    for (let i = 0; i < minMfccLength; i++) {
      mfccDistance += (mfcc1[i] - mfcc2[i]) ** 2;
    }
    
    mfccSimilarity = Math.exp(-Math.sqrt(mfccDistance) / 10);
  }
  
  // Combine similarities
  return 0.6 * spectralSimilarity + 0.4 * mfccSimilarity;
}

/**
 * Extract spectral features for machine learning
 */
export function extractSpectralFeatures(metrics: SpectralMetrics): number[] {
  const features: number[] = [];
  
  // Basic spectral statistics
  features.push(metrics.spectralCentroid);
  features.push(metrics.spectralRolloff);
  features.push(metrics.spectralKurtosis);
  features.push(metrics.spectralComplexity);
  
  // Dominant frequency features
  features.push(metrics.dominantFrequencies.length);
  for (let i = 0; i < 5; i++) {
    features.push(metrics.dominantFrequencies[i] || 0);
  }
  
  // Wavelet features
  features.push(metrics.waveletCoefficients.waveletEntropy);
  for (let i = 0; i < Math.min(3, metrics.waveletCoefficients.energyDistribution.length); i++) {
    features.push(metrics.waveletCoefficients.energyDistribution[i]);
  }
  
  // MFCC features (first 8 coefficients)
  for (let i = 0; i < Math.min(8, metrics.mfccCoefficients.length); i++) {
    features.push(metrics.mfccCoefficients[i]);
  }
  
  // Pad with zeros if needed
  while (features.length < 25) {
    features.push(0);
  }
  
  return features.slice(0, 25); // Return exactly 25 features
}
