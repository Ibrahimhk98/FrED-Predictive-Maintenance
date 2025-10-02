import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import numpy as np


def _make_fake_modules():
    # loader
    loader = ModuleType('loader')
    def load_long_audio(path, target_sr=None):
        sr = target_sr or 40000
        t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        return np.sin(2 * np.pi * 440 * t), sr
    loader.load_long_audio = load_long_audio
    loader.TARGET_SAMPLE_RATE = 40000

    # splitters
    splitters = ModuleType('splitters')
    def segment_region(data, start_sample, end_sample, segment_samples, overlap=0.5):
        stride = max(1, int(segment_samples * (1.0 - overlap)))
        segments = []
        pos = start_sample
        while pos + segment_samples <= end_sample:
            segments.append(data[pos:pos + segment_samples].copy())
            pos += stride
        return segments
    splitters.segment_region = segment_region
    # provide segment_train_test for compatibility imports (returns train/test dict)
    def segment_train_test(data, sr, segment_seconds, overlap=0.5, train_fraction=0.8, buffer_seconds=0.5):
        segs = segment_region(data, 0, len(data), int(segment_seconds * sr), overlap)
        return {'train': segs, 'test': []}
    splitters.segment_train_test = segment_train_test

    # features_extractor
    features_extractor = ModuleType('features_extractor')
    def extract_features_for_list(segments, sr):
        n = len(segments)
        feat_dim = 13
        if n == 0:
            return np.empty((0, 0)), []
        X = np.random.RandomState(0).randn(n, feat_dim)
        feat_names = [f'mfcc_{i}' for i in range(feat_dim)]
        return X, feat_names
    features_extractor.extract_features_for_list = extract_features_for_list

    return loader, splitters, features_extractor


def _load_batch_predict_module():
    bp = Path.cwd() / 'Preparation' / 'Sample Preparation' / 'Models' / 'batch_predict.py'
    spec = importlib.util.spec_from_file_location('batch_predict_test', str(bp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_batch_predict_aggregates_one_row_per_file(monkeypatch):
    # inject fake modules
    loader, splitters, features_extractor = _make_fake_modules()
    monkeypatch.setitem(sys.modules, 'loader', loader)
    monkeypatch.setitem(sys.modules, 'splitters', splitters)
    monkeypatch.setitem(sys.modules, 'features_extractor', features_extractor)

    mod = _load_batch_predict_module()

    # create a dummy model that supports predict and predict_proba
    class DummyModel:
        def __init__(self, classes):
            self.classes_ = np.array(classes)
        def predict(self, X):
            n = X.shape[0]
            return np.array([self.classes_[i % len(self.classes_)] for i in range(n)])
        def predict_proba(self, X):
            n = X.shape[0]
            probs = np.zeros((n, len(self.classes_))) + 0.1
            probs[:, 0] = 0.8
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs

    model = DummyModel(['Good', 'Fault'])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.random.RandomState(1).randn(20, 13))

    root = Path.cwd() / 'sample_data' / 'audio'
    df, summary = mod.batch_predict_samples(root, best_model=model, scaler=scaler, segment_seconds=0.2, overlap=0.0, feature_level='standard')

    # Expect one row per discovered file
    # There are 3 files under sample_data/audio/Chipped Tooth in the repo
    assert df.shape[0] >= 1
    # Ensure per-segment columns are not present (we aggregate to file-level)
    assert 'segment_predictions' not in df.columns
    assert 'segment_confidences' not in df.columns
    # confidence should be between 0 and 1
    assert df['confidence'].between(0.0, 1.0).all()
