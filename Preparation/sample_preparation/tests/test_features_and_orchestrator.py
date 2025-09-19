import numpy as np
from pathlib import Path
import importlib.util

BASE = Path(__file__).resolve().parents[3]


def load_module_by_path(rel_path):
    p = BASE / rel_path
    spec = importlib.util.spec_from_file_location('orch_test', str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extract_basic_features():
    # simple constant signal: load rich_features by path and test basic extractor
    # skip if librosa not available
    try:
        import importlib
        importlib.import_module('librosa')
    except Exception:
        import pytest
        pytest.skip('librosa not installed in test environment')

    rf_path = BASE / 'Preparation' / 'Sample Preparation' / 'Feature_extraction_pipeline' / 'rich_features.py'
    spec = importlib.util.spec_from_file_location('rf_test', str(rf_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    extract_basic_features = getattr(mod, 'extract_basic_features')
    x = np.ones(100)
    vec, names = extract_basic_features(x, sr=16000)
    assert vec.shape[0] == len(names)
    assert 'mean' in names and 'rms' in names


def test_lazy_loader_and_orchestrator_imports():
    # load orchestrators and ensure lazy getter exists
    p_pipeline = BASE / 'Preparation' / 'Sample Preparation' / 'pipeline' / 'orchestrator.py'
    p_fep = BASE / 'Preparation' / 'Sample Preparation' / 'Feature_extraction_pipeline' / 'orchestrator.py'
    # prefer the older pipeline path if present, otherwise use the consolidated one
    if p_pipeline.exists():
        mod1 = load_module_by_path(str(p_pipeline.relative_to(BASE)))
    else:
        mod1 = load_module_by_path(str(p_fep.relative_to(BASE)))
    mod2 = load_module_by_path(str(p_fep.relative_to(BASE)))
    assert hasattr(mod1, '_get_rich_extractor')
    assert callable(getattr(mod1, '_get_rich_extractor'))
    assert hasattr(mod2, '_get_rich_extractor')
    assert callable(getattr(mod2, '_get_rich_extractor'))

    fn1 = mod1._get_rich_extractor()
    fn2 = mod2._get_rich_extractor()
    # either loader returns a callable or None; both are acceptable but should not raise
    assert (fn1 is None) or callable(fn1)
    assert (fn2 is None) or callable(fn2)


def test_orchestrator_return_shape():
    # call run_pipeline_on_file on synthetic data by writing a temp wav
    import soundfile as sf
    import tempfile
    sr = 8000
    data = np.random.randn(sr * 2).astype('float32')
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'test.wav'
        sf.write(str(p), data, sr)
        # prefer the older pipeline orchestrator path if present, otherwise use the consolidated one
        p_pipeline = BASE / 'Preparation' / 'Sample Preparation' / 'pipeline' / 'orchestrator.py'
        p_fep = BASE / 'Preparation' / 'Sample Preparation' / 'Feature_extraction_pipeline' / 'orchestrator.py'
        if p_pipeline.exists():
            mod = load_module_by_path(str(p_pipeline.relative_to(BASE)))
        else:
            mod = load_module_by_path(str(p_fep.relative_to(BASE)))
        res = mod.run_pipeline_on_file(p, segment_seconds=0.5, overlap=0.2, train_fraction=0.5, buffer_seconds=0.01, feature_level='standard')
        assert 'train' in res and 'test' in res
        for k in ('train', 'test'):
            assert isinstance(res[k], dict)
            assert 'X' in res[k] and 'y' in res[k] and 'meta' in res[k] and 'feature_names' in res[k]


def test_feature_dimension_consistency_file_vs_dataset():
    # create a small temporary wav and ensure file-level and dataset-level feature dims match
    import soundfile as sf
    import tempfile
    sr = 8000
    data = np.random.randn(sr * 2).astype('float32')
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # create class subfolder to mimic dataset layout
        cls_dir = tmpdir / 'classA'
        cls_dir.mkdir()
        p = cls_dir / 'test.wav'
        sf.write(str(p), data, sr)
        # load the new orchestrator implementation from Feature_extraction_pipeline
        mod_path = BASE / 'Preparation' / 'Sample Preparation' / 'Feature_extraction_pipeline' / 'orchestrator.py'
        spec = importlib.util.spec_from_file_location('orch_test2', str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        res_file = mod.run_pipeline_on_file(p, segment_seconds=0.5, overlap=0.2, train_fraction=0.5, buffer_seconds=0.01, feature_level='standard')
        res_dataset = mod.run_pipeline_on_dataset(tmpdir, segment_seconds=0.5, overlap=0.2, train_fraction=0.5, buffer_seconds=0.01, feature_level='standard')

        for split in ('train', 'test'):
            xf = res_file.get(split, {}).get('X')
            xd = res_dataset.get(split, {}).get('X')
            # if both are empty, that's acceptable; otherwise their feature dim should match
            if getattr(xf, 'size', 0) and getattr(xd, 'size', 0):
                assert xf.shape[1] == xd.shape[1], f"Feature count mismatch for split {split}: file={xf.shape} dataset={xd.shape}"
