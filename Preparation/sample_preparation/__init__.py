"""Compatibility package wrapping the on-disk 'Sample Preparation' folder.

This module uses importlib to dynamically load modules from the folder whose
name contains spaces, allowing standard imports like
`from Preparation.sample_preparation import pipeline`.
"""
from importlib import util
from pathlib import Path
_root = Path(__file__).resolve().parents[1] / 'Sample Preparation'

def _load_module(name, relpath):
    candidate = _root / relpath
    if candidate.exists():
        spec = util.spec_from_file_location(name, str(candidate))
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"Could not load {name} from {candidate}")

# re-export commonly used subpackages/modules
try:
    pipeline = _load_module('pipeline', 'pipeline/orchestrator.py')
except Exception:
    pipeline = None
