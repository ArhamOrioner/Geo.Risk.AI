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


