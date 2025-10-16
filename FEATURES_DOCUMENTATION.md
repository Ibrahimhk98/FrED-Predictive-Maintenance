# Rich Features Documentation

This document explains the audio features extracted at each level in the `rich_features.py` pipeline. The pipeline offers three levels of feature extraction:

- **Basic**: Simple time-domain statistical features
- **Standard**: Adds spectral, MFCC, and other common audio features  
- **Advanced**: Includes time-series complexity features and higher-order statistics

Each higher level includes all features from lower levels.

## Basic Features (11 features)

### Time-Domain Statistics
- **mean**: Average amplitude of the signal
- **std**: Standard deviation of amplitude values
- **var**: Variance of amplitude values
- **median**: Middle value when amplitudes are sorted
- **min**: Minimum amplitude value
- **max**: Maximum amplitude value

### Energy and Shape Features
- **rms**: Root Mean Square - measure of signal power/energy
- **energy**: Total energy (sum of squared amplitudes)
- **zcr**: Zero Crossing Rate - frequency of sign changes, indicates pitch/noisiness
- **skewness**: Asymmetry of the amplitude distribution
- **kurtosis**: "Tailedness" of the amplitude distribution (peakedness)

### Quantiles
- **quantile_25**: 25th percentile of amplitude values
- **quantile_50**: 50th percentile (same as median)
- **quantile_75**: 75th percentile of amplitude values

## Standard Features (~50+ features)

Includes all **Basic** features plus:

### Spectral Features
These analyze the frequency content of the signal:

- **spectral_centroid_mean/std**: Center of mass of the spectrum (brightness)
- **spectral_bandwidth_mean/std**: Width of the spectrum around the centroid
- **spectral_rolloff_mean/std**: Frequency below which 85% of energy is contained
- **spectral_flatness_mean/std**: Measure of how noise-like vs tonal the spectrum is

### Spectral Contrast
- **spectral_contrast_b{i}_mean**: Contrast between peaks and valleys in different frequency bands
  - Multiple bands (typically 7 bands) provide detailed frequency analysis

### MFCC Features (Mel-Frequency Cepstral Coefficients)
MFCCs are widely used in audio analysis and mimic human auditory perception:
- **mfcc_{i}_mean/std**: Mean and standard deviation of each MFCC coefficient (default: 13 coefficients)
  - mfcc_0: Related to overall energy
  - mfcc_1-12: Capture spectral shape and timbre characteristics

### Harmonic Features
- **chroma_mean/std**: Harmonic content related to musical pitch classes
  - Useful for detecting tonal vs atonal content

### Temporal Features  
- **onset_strength_mean**: Average strength of note/event onsets
- **tempo**: Estimated beats per minute (rhythmic content)

## Advanced Features (~65+ features)

Includes all **Standard** features plus:

### Hjorth Parameters
Time-series complexity measures from EEG analysis, adapted for audio:
- **hjorth_activity**: Variance of the signal (measure of power)
- **hjorth_mobility**: Mean frequency or mobility of the signal
- **hjorth_complexity**: Change in frequency, measure of similarity to a sine wave

### Signal Quality Measures
- **crest_factor**: Peak-to-RMS ratio - indicates impulsiveness vs steady-state
- **autocorr_lag1**: Lag-1 autocorrelation - measures short-term predictability

### Entropy-Based Complexity
- **sample_entropy**: Regularity measure - lower values = more regular/predictable
- **permutation_entropy**: Complexity based on ordinal patterns in the time series

### Advanced Spectral Features
- **spectral_entropy**: Disorder in the frequency domain - higher = more noise-like
- **dominant_freq**: Frequency with highest power in the spectrum

## Feature Selection Guidelines

### For Bearing Fault Detection:
- **Basic**: Good starting point for simple fault vs normal classification
- **Standard**: Recommended - spectral features capture bearing defect frequencies, MFCCs provide robust representation
- **Advanced**: Best for complex scenarios - entropy measures detect irregularities, Hjorth parameters capture fault dynamics

### Computational Considerations:
- **Basic**: Fastest, minimal dependencies
- **Standard**: Moderate computation, requires `librosa` 
- **Advanced**: Highest computation, includes entropy calculations

### Data Requirements:
- **Basic**: Works with any segment length
- **Standard**: Requires segments long enough for meaningful spectral analysis (typically ≥0.1s)
- **Advanced**: Entropy measures need sufficient data points for reliable estimates (typically ≥0.5s)

## Usage Example

```python
from rich_features import extract_features_for_list

# Extract features for a list of audio segments
segments = [audio_segment1, audio_segment2, ...]  # List of numpy arrays
sample_rate = 40000

# Choose feature level
X_basic, names_basic = extract_features_for_list(segments, sample_rate, level='basic')
X_standard, names_standard = extract_features_for_list(segments, sample_rate, level='standard') 
X_advanced, names_advanced = extract_features_for_list(segments, sample_rate, level='advanced')

print(f"Basic features: {len(names_basic)} features")
print(f"Standard features: {len(names_standard)} features") 
print(f"Advanced features: {len(names_advanced)} features")
```

## Notes

- All features are designed to be robust to different signal lengths and conditions
- Features include fallback values (typically 0.0) when computation fails
- The pipeline handles edge cases like empty segments or insufficient data
- Feature names are consistent and descriptive for interpretability