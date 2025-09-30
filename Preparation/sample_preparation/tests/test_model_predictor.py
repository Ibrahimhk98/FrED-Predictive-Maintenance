import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path as _P

# Add models directory to path (spaces in directories make direct package import awkward)
_models_dir = _P(__file__).resolve().parents[3] / 'Preparation' / 'Sample Preparation' / 'Models'
sys.path.append(str(_models_dir))
from model_predictor import ModelPredictor

# minimal fake feature extractor that returns dict shape like orchestrator

def fake_extractor(path_or_none, segment_seconds=1.0, overlap=0.5, feature_level='standard', audio_data=None, sr=None):
    # produce deterministic features: 5 segments x 4 features
    X = np.tile(np.array([1, 2, 3, 4], dtype=float), (5, 1))
    return {"train": {"X": X, "y": ['a']*5}}

def test_predict_file_basic():
    X_train = np.random.randn(20, 4)
    y_train = np.random.choice(['good', 'bad'], size=20)
    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(X_train, y_train)
    predictor = ModelPredictor(model=rf, feature_extractor=fake_extractor)
    res = predictor.predict_file(Path('dummy.wav'))
    assert 'prediction' in res and res['prediction'] is not None
    assert 'confidence' in res and 0.0 <= res['confidence'] <= 1.0


def test_streaming_update():
    X_train = np.random.randn(10, 4)
    y_train = np.random.choice(['x', 'y'], size=10)
    rf = RandomForestClassifier(n_estimators=3, random_state=1)
    rf.fit(X_train, y_train)
    predictor = ModelPredictor(model=rf, feature_extractor=fake_extractor, moving_avg_window=3)
    # simulate features
    for _ in range(5):
        fv = np.random.randn(4)
        pred, conf = predictor.update_streaming(fv)
    assert pred in ['x', 'y']
    assert 0.0 <= conf <= 1.0
