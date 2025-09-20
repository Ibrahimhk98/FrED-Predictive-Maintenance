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


def load_long_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load a long audio file and return mono numpy array and sample rate.

    Args:
        path: Path to audio file
    Returns:
        data (1-D numpy array), sample_rate (int)
    """
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, sr


def index_audio(data: np.ndarray) -> np.ndarray:
    """Return an index array for samples (0..N-1). Useful for debugging/visualization.

    Args:
        data: 1-D audio samples
    Returns:
        sample indices as numpy array
    """
    return np.arange(data.shape[0])
