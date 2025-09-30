"""Audio file loading utilities.

This module handles loading long audio files with consistent format:
1. Automatic conversion to mono (averaging channels if needed)  
2. Standard numpy array and sample rate output format
3. Support for common audio formats through soundfile (wav, flac, ogg)
4. Helper functions for sample indexing and debugging

The loading functions handle multi-channel audio transparently by
converting to mono, making downstream processing simpler.
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np
import soundfile as sf
import librosa

# Target sample rate for all pipeline audio
TARGET_SAMPLE_RATE = 40000


def load_long_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load a long audio file and return mono numpy array and sample rate.

    This function guarantees the returned audio is mono and resampled to ``target_sr``Hz.

    Args:
        path: Path to audio file
        target_sr: desired sample rate for returned audio (default: TARGET_SAMPLE_RATE)

    Returns:
        data (1-D numpy array, dtype=float32), sample_rate (int)
    """
    data, sr = sf.read(str(path))
    # convert multi-channel to mono
    if data.ndim > 1:
        # average across channels
        data = np.mean(data, axis=1)

    # ensure floating point array
    data = np.asarray(data, dtype=np.float32)

    # resample if needed
    if sr != target_sr:
        try:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception:
            # fallback: if librosa fails, attempt scipy (if available) or return original
            try:
                from scipy.signal import resample
                num = int(len(data) * float(target_sr) / float(sr))
                data = resample(data, num)
                sr = target_sr
            except Exception:
                # if resampling fails, return original data and sr
                pass

    return data, sr


def index_audio(data: np.ndarray) -> np.ndarray:
    """Return an index array for samples (0..N-1). Useful for debugging/visualization.

    Args:
        data: 1-D audio samples
    Returns:
        sample indices as numpy array
    """
    return np.arange(data.shape[0])
