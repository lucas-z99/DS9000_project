"""Resolve model artifact folders after organizing an experiment."""
import json
from pathlib import Path


def model_run_dir(comparison_dir, model_type):
    root = Path(comparison_dir)
    manifest = root / 'artifact_paths.json'
    paths = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else {}
    return (root / paths.get(model_type, '.')).resolve()
