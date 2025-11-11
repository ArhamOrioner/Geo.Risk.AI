# ----------------------------------------------------------------------------
# GeoRiskAI - Lightweight Model Registry (Local PoC)
# ----------------------------------------------------------------------------
# Purpose: Track trained model artifacts, metadata, and versions locally.
# Storage: Models saved via joblib; metadata JSON index.
# ----------------------------------------------------------------------------

import json
import time
from pathlib import Path
from typing import Any, Dict
import joblib
import hashlib
import subprocess
import sys

REGISTRY_DIR = Path("model_registry")
INDEX_FILE = REGISTRY_DIR / "index.json"

def _ensure_registry():
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text(json.dumps({}), encoding="utf-8")

# <<< NEW FUNCTION >>>
def _sha256_file(path):
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# <<< NEW FUNCTION >>>
def _snapshot_requirements(out_path):
    """Saves the current Python environment's packages to a file."""
    try:
        reqs = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        with open(out_path, "w") as f:
            f.write(reqs)
        return True
    except Exception as e:
        logging.error(f"Could not snapshot requirements: {e}")
        return False

# <<< MODIFIED >>>
def register_model(model: Any, metadata: Dict[str, Any]) -> str:
    """Save model and register metadata, including hash and requirements."""
    _ensure_registry()
    idx = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    version = str(int(time.time()))
    
    model_path = REGISTRY_DIR / f"model_{version}.joblib"
    meta_path = REGISTRY_DIR / f"model_{version}.json"
    reqs_path = REGISTRY_DIR / f"requirements_{version}.txt"

    # Save artifacts
    joblib.dump(model, model_path)
    _snapshot_requirements(reqs_path)
    
    # Create metadata
    meta = {
        "version": version,
        "artifact_sha256": _sha256_file(model_path),
        "python_version": sys.version,
        **metadata
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Update index
    idx[version] = {"model": model_path.name, "meta": meta_path.name, "requirements": reqs_path.name}
    INDEX_FILE.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    
    logging.info(f"Registered model version {version} with SHA256: {meta['artifact_sha256']}")
    return version


def load_model(version: str) -> Any:
    _ensure_registry()
    idx = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if version not in idx:
        raise ValueError(f"Model version {version} not found")
    model_file = REGISTRY_DIR / idx[version]["model"]
    return joblib.load(model_file)


def latest_version() -> str | None:
    _ensure_registry()
    idx = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    return max(idx.keys()) if idx else None

