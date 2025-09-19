"""Smoke test for orchestrators: lazy-load rich_features and run on synthetic data if available.

This script is safe to run without heavy deps; it will report fallbacks when rich extractor is not available.
"""
from importlib import util
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[2]

def load_module_by_path(rel_path):
    p = BASE / rel_path
    spec = util.spec_from_file_location('orch_test', str(p))
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

modules = [
    'Preparation/Sample Preparation/pipeline/orchestrator.py',
    'Preparation/Sample Preparation/Feature_extraction_pipeline/orchestrator.py'
]

# synthetic segments: 2 segments of 1600 samples
segs = [np.random.randn(1600).astype(float), np.random.randn(1600).astype(float)]
sr = 16000

for mpath in modules:
    print('\n---')
    print('Module:', mpath)
    try:
        mod = load_module_by_path(mpath)
        getter = getattr(mod, '_get_rich_extractor', None)
        print('has getter:', callable(getter))
        if callable(getter):
            fn = getter()
            if fn is None:
                print('Rich extractor not available; fallback will be used at runtime.')
            else:
                print('Rich extractor loaded. Calling on synthetic segments...')
                X, names = fn(segs, sr, level='standard')
                print('X shape:', getattr(X, 'shape', None))
                print('feature count:', len(names) if names is not None else None)
    except Exception as e:
        print('Failed to load module:', e)

print('\nSmoke test finished.')
