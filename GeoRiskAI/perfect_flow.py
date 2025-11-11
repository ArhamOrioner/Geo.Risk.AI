# ----------------------------------------------------------------------------
# GeoRiskAI - Prefect Pipeline PoC
# ----------------------------------------------------------------------------
# Tasks:
#  - ingest_dfo
#  - enrich_events (with feature store caching)
#  - train_model (with spatial CV)
#  - register_model
#  - predict_roi
# ----------------------------------------------------------------------------

from prefect import flow, task
import pandas as pd
from datetime import timedelta

import ML as ml
import gee as gee
import analysis as analysis
from feature_store import load_features, save_features
import model_registry as registry


@task(retries=2, retry_delay_seconds=30)
def ingest_events() -> pd.DataFrame:
    # Use GFD as authoritative source
    df = ml.download_and_prepare_gfd_data()
    if df is None:
        raise ValueError("Failed to load GFD events")
    # Augment with negatives for robust training
    return ml.augment_with_negative_samples(df)


@task(retries=2, retry_delay_seconds=60)
def enrich_events(events_df: pd.DataFrame) -> pd.DataFrame:
    # Attempt to load from feature store by year batch key; else compute and cache
    key = {"type": "training_events", "years": ",".join(map(str, sorted(events_df['event_year'].unique())))[:64]}
    cached = load_features(key)
    if cached is not None and len(cached) >= len(events_df) * 0.9:
        return cached
    from main import enrich_events_with_gee
    df = enrich_events_with_gee(events_df)
    save_features(df, key)
    return df


@task
def train_model(enriched_df: pd.DataFrame) -> ml.ProductionRiskModel:
    X = ml.engineer_features_vectorized(enriched_df)
    y = enriched_df['y_binary']
    try:
        groups = analysis.assign_spatial_clusters(enriched_df[['latitude', 'longitude']].copy())
    except Exception:
        groups = enriched_df.get('event_year')
    model = ml.ProductionRiskModel()
    model.train(X, y, groups=groups)
    return model


@task
def register_trained_model(model: ml.ProductionRiskModel) -> str:
    meta = {"type": "xgb_binary", "threshold": getattr(model, 'risk_threshold', 0.5)}
    return registry.register_model(model, meta)


@flow(name="GeoRiskAI_PoC")
def georiskai_flow():
    events = ingest_events()
    enriched = enrich_events(events)
    model = train_model(enriched)
    version = register_trained_model(model)
    return version


if __name__ == "__main__":
    georiskai_flow()









# ----------------------------------------------------------------------------
# GeoRiskAI - Lightweight Feature Store (Local PoC)
# ----------------------------------------------------------------------------
# Purpose: Cache enriched features keyed by event_id or ROI/date to avoid
# recomputation and ensure training/serving consistency.
# Storage: Parquet files in a local directory.
# ----------------------------------------------------------------------------

import hashlib
from pathlib import Path
import pandas as pd

DEFAULT_STORE_DIR = Path("feature_store")


def _hash_key(key_dict: dict) -> str:
    """Create a stable hash key from a dict of identifiers."""
    canon = ",".join(f"{k}={key_dict[k]}" for k in sorted(key_dict))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def ensure_store(dir_path: Path | None = None) -> Path:
    path = dir_path or DEFAULT_STORE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_features(df: pd.DataFrame, key: dict, dir_path: Path | None = None) -> Path:
    """Save feature DataFrame under a hashed key; returns file path."""
    store = ensure_store(dir_path)
    file = store / f"{_hash_key(key)}.parquet"
    df.to_parquet(file, index=False)
    return file


def load_features(key: dict, dir_path: Path | None = None) -> pd.DataFrame | None:
    """Load feature DataFrame if present; else return None."""
    store = ensure_store(dir_path)
    file = store / f"{_hash_key(key)}.parquet"
    if file.exists():
        return pd.read_parquet(file)
    return None


def has_features(key: dict, dir_path: Path | None = None) -> bool:
    store = ensure_store(dir_path)
    file = store / f"{_hash_key(key)}.parquet"
    return file.exists()









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


