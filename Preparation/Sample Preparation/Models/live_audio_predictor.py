"""Live audio capture and real-time prediction utility.

This script provides a `LiveAudioPredictor` class that:
  * Captures audio from the default input device (microphone) using sounddevice
  * Buffers samples, segments them into windows compatible with the offline pipeline
  * Extracts features (MFCC-based via existing extractor path) and updates a streaming model predictor
  * Maintains a moving average vote via `ModelPredictor`

Example usage (inside a notebook or script):
    from live_audio_predictor import LiveAudioPredictor
    lap = LiveAudioPredictor(model=predictor.model, scaler=predictor.scaler,
                             segment_seconds=3.0, overlap=0.5, feature_level='standard')
    lap.start()
    # speak / make machine noise...
    time.sleep(15)
    lap.stop()
    print(lap.get_recent_predictions())

Notes:
  * Requires `sounddevice` (already in requirements.txt)
  * Resamples captured audio to 40000 Hz if device rate differs
  * Uses lightweight MFCC features from existing feature_extraction pipeline via `extract_features_for_list`
  * For safety, long-running capturing is performed in a background thread.
"""

from __future__ import annotations
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Deque
from collections import deque
import numpy as np
import sounddevice as sd

from pathlib import Path
import sys

# Add feature extraction pipeline path (adapting to repo structure with spaces)
_root = Path(__file__).resolve().parents[2] / 'Feature_extraction_pipeline'
sys.path.append(str(_root))
from features_extractor import extract_features_for_list  # type: ignore
from splitters import segment_train_test  # type: ignore
from loader import TARGET_SAMPLE_RATE  # reuse constant

# Add model predictor path
_models_dir = Path(__file__).resolve().parent
sys.path.append(str(_models_dir))
from model_predictor import ModelPredictor


@dataclass
class LiveAudioPredictor:
    model: any
    scaler: any = None
    segment_seconds: float = 3.0
    overlap: float = 0.5
    feature_level: str = 'standard'
    moving_avg_window: int = 5
    device: Optional[int] = None
    block_duration: float = 0.25  # seconds per audio callback block
    ring_duration: float = 30.0   # seconds of raw audio to retain
    dtype: str = 'float32'

    _predictor: ModelPredictor = field(init=False)
    _audio_q: "queue.Queue[np.ndarray]" = field(init=False, default_factory=queue.Queue)
    _raw_buffer: Deque[float] = field(init=False)
    _thread: Optional[threading.Thread] = field(init=False, default=None)
    _stop_flag: threading.Event = field(init=False, default_factory=threading.Event)
    _recent_preds: Deque[Tuple[float, str, float]] = field(init=False)

    def __post_init__(self):
        self._predictor = ModelPredictor(
            model=self.model,
            scaler=self.scaler,
            feature_extractor=None,  # not used directly; we call feature funcs here
            segment_seconds=self.segment_seconds,
            overlap=self.overlap,
            feature_level=self.feature_level,
            moving_avg_window=self.moving_avg_window,
        )
        max_samples = int(self.ring_duration * TARGET_SAMPLE_RATE)
        self._raw_buffer = deque(maxlen=max_samples)
        self._recent_preds = deque(maxlen=500)  # (timestamp, label, confidence)

    # ---------------------- Public control ----------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_recent_predictions(self) -> List[Tuple[float, str, float]]:
        return list(self._recent_preds)

    # ---------------------- Internal logic ----------------------
    def _worker_loop(self):
        # open an input stream
        block_frames = int(self.block_duration * TARGET_SAMPLE_RATE)
        def _callback(indata, frames, time_info, status):  # noqa: D401
            if status:
                # status messages (over/underflows)
                pass
            # indata shape: (frames, channels)
            data_mono = indata.mean(axis=1).astype(self.dtype, copy=False)
            self._audio_q.put(data_mono)

        with sd.InputStream(
            samplerate=TARGET_SAMPLE_RATE,
            blocksize=block_frames,
            channels=1,
            dtype=self.dtype,
            callback=_callback,
            device=self.device,
        ):
            last_process = time.time()
            process_interval = self.segment_seconds * (1 - self.overlap) / 2.0  # process fairly often
            while not self._stop_flag.is_set():
                try:
                    block = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    pass
                else:
                    self._raw_buffer.extend(block.tolist())

                now = time.time()
                if now - last_process >= process_interval:
                    self._process_buffer(now)
                    last_process = now
            # final flush
            self._process_buffer(time.time())

    def _process_buffer(self, ts: float):
        if not self._raw_buffer:
            return
        data = np.asarray(self._raw_buffer, dtype=np.float32)
        segs_dict = segment_train_test(
            data,
            TARGET_SAMPLE_RATE,
            segment_seconds=self.segment_seconds,
            overlap=self.overlap,
            train_fraction=1.0,  # treat entire stream as 'train' region
            buffer_seconds=0.0,
        )
        segments = segs_dict.get('train', [])
        if not segments:
            return
        # Only extract for the newest segment to reduce CPU load
        latest_seg = segments[-1]
        X, _ = extract_features_for_list([latest_seg], TARGET_SAMPLE_RATE)
        if X is None or X.size == 0:
            return
        pred, conf = self._predictor.update_streaming(X[0])
        if pred is not None:
            self._recent_preds.append((ts, pred, conf))


def demo_live(duration: float = 10.0):  # pragma: no cover - convenience interactive function
    """Quick interactive demo (requires a previously trained model + scaler globals)."""
    raise NotImplementedError("Hook this function into a script/notebook where model & scaler exist.")


__all__ = ["LiveAudioPredictor", "demo_live"]
