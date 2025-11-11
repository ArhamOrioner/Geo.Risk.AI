from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import joblib
import numpy as np
import pandas as pd
import shap

import config


LOGGER = logging.getLogger(__name__)


SUPPORT_CLASS_NAMES = ("WeightedEnsemble", "IsotonicCalibratedModel")


STATIC_FEATURES: List[str] = [
    "gee_api_weighted_mm",
    "gee_aspect_mean",
    "gee_elevation_mean",
    "gee_flashiness_index",
    "gee_gsw_occurrence_mean",
    "gee_imerg_intensity_7d_mm_per_day",
    "gee_imerg_max_daily_7d_mm",
    "gee_imerg_sum_7d_before_mm",
    "gee_merit_upa_mean",
    "gee_ndbi_mean_30d",
    "gee_ndvi_mean_30d",
    "gee_ndwi_mean_30d",
    "gee_saturation_proxy",
    "gee_slope_mean",
    "gee_smap_subsurface_soil_moisture",
    "gee_smap_surface_soil_moisture",
    "gee_twi",
]


DYNAMIC_FEATURES: List[str] = [
    "gee_antecedent_moisture_proxy",
    "gee_flashiness_index_7d",
    "gee_imerg_intensity_3d_mm_per_day",
    "gee_imerg_max_1h_intensity_mm",
    "gee_imerg_max_3h_intensity_mm",
    "gee_imerg_max_6h_intensity_mm",
    "gee_imerg_sum_24h_mm",
    "gee_imerg_sum_3d_before_mm",
    "gee_precip_x_slope",
    "gee_runoff_potential",
    "gee_smap_soil_moisture_anomaly",
    "glofas_forecast_10th_percentile",
    "glofas_forecast_90th_percentile",
    "glofas_forecast_control",
    "glofas_forecast_mean",
    "glofas_forecast_median",
    "glofas_forecast_std_dev",
    "imerg_max_1h_intensity_mm",
    "imerg_mean_1h_intensity_mm",
]


FULL_FEATURES: List[str] = [
    "imerg_max_1h_intensity_mm",
    "imerg_mean_1h_intensity_mm",
    "gee_imerg_sum_24h_mm",
    "gee_imerg_max_1h_intensity_mm",
    "gee_imerg_max_3h_intensity_mm",
    "gee_imerg_max_6h_intensity_mm",
    "gee_imerg_sum_3d_before_mm",
    "gee_imerg_sum_7d_before_mm",
    "gee_imerg_max_daily_7d_mm",
    "gee_imerg_intensity_3d_mm_per_day",
    "gee_imerg_intensity_7d_mm_per_day",
    "gee_smap_surface_soil_moisture",
    "gee_smap_subsurface_soil_moisture",
    "gee_smap_soil_moisture_anomaly",
    "gee_antecedent_moisture_proxy",
    "gee_flashiness_index_7d",
    "gee_api_weighted_mm",
    "gee_flashiness_index",
    "gee_saturation_proxy",
    "gee_runoff_potential",
    "gee_aspect_mean",
    "gee_elevation_mean",
    "gee_slope_mean",
    "gee_gsw_occurrence_mean",
    "gee_merit_upa_mean",
    "gee_twi",
    "gee_ndvi_mean_30d",
    "gee_ndbi_mean_30d",
    "gee_ndwi_mean_30d",
    "gee_precip_x_slope",
    "glofas_forecast_control",
    "glofas_forecast_mean",
    "glofas_forecast_median",
    "glofas_forecast_std_dev",
    "glofas_forecast_10th_percentile",
    "glofas_forecast_90th_percentile",
]


def _prime_support_classes() -> None:
    """Ensure custom classes used during training are importable before unpickling."""

    # Training scripts define WeightedEnsemble/IsotonicCalibratedModel in 3.py.
    training_defs = Path(__file__).resolve().parent / "3.py"
    candidates = [training_defs]

    support_dir = Path(config.PRETRAINED_SUPPORT_PATH)
    if support_dir.exists():
        candidates.append(support_dir / "1.py")

    for candidate in candidates:
        if not candidate.exists():
            continue
        module_name = f"ensemble_support_{hash(candidate)}"
        spec = importlib.util.spec_from_file_location(module_name, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        if module_name not in sys.modules:
            sys.modules[module_name] = module
        spec.loader.exec_module(module)
        for attr in ["WeightedEnsemble", "IsotonicCalibratedModel"]:
            if hasattr(module, attr):
                setattr(sys.modules["__main__"], attr, getattr(module, attr))


def _coerce_feature_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    frame = df.reindex(columns=columns, copy=True)
    for col in columns:
        if col not in frame:
            frame[col] = np.nan
    return frame[columns].apply(pd.to_numeric, errors="coerce")


def _build_feature_frames(df: pd.DataFrame, feature_sets: Mapping[str, Iterable[str]]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for role, cols in feature_sets.items():
        frames[role] = _coerce_feature_frame(df, list(cols))
    return frames


class ProductionRiskModel:
    def __init__(self) -> None:
        self.ensemble = None
        self.feature_sets: Dict[str, List[str]] = {
            "static": STATIC_FEATURES,
            "dynamic": DYNAMIC_FEATURES,
            "full": FULL_FEATURES,
        }
        self.risk_threshold: float = 0.5

    @property
    def is_trained(self) -> bool:
        return self.ensemble is not None

    def load_pretrained_model(self, model_path: Optional[str] = None) -> None:
        target_path = Path(model_path or config.PRETRAINED_MODEL_PATH)
        if not target_path.exists():
            raise FileNotFoundError(f"Pretrained model not found at {target_path}")

        support_dir = Path(config.PRETRAINED_SUPPORT_PATH)
        if support_dir.exists() and str(support_dir) not in sys.path:
            sys.path.insert(0, str(support_dir))

        _prime_support_classes()

        LOGGER.info("Loading pretrained ensemble from %s", target_path)
        self.ensemble = joblib.load(target_path)

        if hasattr(self.ensemble, "feature_sets"):
            self.feature_sets = {
                role: list(cols)
                for role, cols in getattr(self.ensemble, "feature_sets").items()
            }

        if hasattr(self.ensemble, "threshold"):
            self.risk_threshold = float(getattr(self.ensemble, "threshold"))

    def predict_per_pixel(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Pretrained ensemble not loaded. Call load_pretrained_model() first.")

        frames = _build_feature_frames(df, self.feature_sets)
        blended = self.ensemble.predict_proba(feature_frames=frames)

        results = pd.DataFrame(
            {
                "Risk_Probability": blended,
                "Final_Risk_Score": blended,
                "Risk_Lower_90": np.maximum(0.0, blended - 0.0),
                "Risk_Upper_90": np.minimum(1.0, blended + 0.0),
                "Uncertainty": np.zeros_like(blended),
                "Prediction": (blended >= self.risk_threshold).astype(int),
            },
            index=df.index,
        )

        results["SHAP_Values"] = [[] for _ in range(len(results))]

        if hasattr(self.ensemble, "calibration_"):
            calib = getattr(self.ensemble, "calibration_") or {}
            results.attrs["calibration"] = calib

        return results
