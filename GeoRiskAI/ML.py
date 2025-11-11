# -----------------------------------------------------------------------------
# GeoRiskAI - Scientifically Defensible Prediction Engine
#
# Core changes:
# 1) TARGET: No normalization of ordinal severity; use binary target (Severity ≥ 1.5)
#    and provide an ordinal label for future ordinal-regression research.
# 2) FEATURES: Remove magic numbers; compute soil hydraulic conductivity (Ksat)
#    via Cosby et al. (1984) pedotransfer function using SoilGrids sand/clay.
# 3) MODEL: Calibrated XGBoost classifier outputting probability; select
#    operating threshold via cross-validated precision-recall analysis.
# 4) XAI: SHAP explainability remains for transparency.
# -----------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, brier_score_loss
import torch
import torch.nn as nn
import torch.optim as optim

# --- Authoritative Feature Engineering ---
def engineer_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized, scientifically grounded feature engineering.
    - Uses raw environmental predictors from GEE
    - Derives saturated hydraulic conductivity (Ksat) via Cosby et al. (1984) PTF
    """
    features = pd.DataFrame(index=df.index)

    # Core geophysical and hydro-meteorological predictors
    features['elevation'] = df.get('Elevation', np.nan)
    features['slope'] = df.get('Slope', np.nan)
    features['ndvi'] = df.get('NDVI', np.nan)
    features['ndwi'] = df.get('NDWI', np.nan)
    features['twi'] = df.get('TWI', np.nan)
    features['flow_accumulation'] = df.get('Flow_Accumulation', np.nan)
    features['dist_to_channel_px'] = df.get('Dist_To_Channel_Px', np.nan)
    features['channel_mask'] = df.get('Channel_Mask', np.nan)
    features['total_precipitation'] = df.get('Total_Precipitation', np.nan)
    features['max_daily_precipitation'] = df.get('Max_Daily_Precipitation', np.nan)
    # Robust antecedent wetness metrics
    features['api_3day'] = df.get('AP3', np.nan)
    features['api_7day'] = df.get('AP7', np.nan)
    # Decaying API with k=0.9 from GEE
    features['api_k09'] = df.get('API_k09', np.nan)
    # NRT precipitation and water proxies
    features['imerg_total_3day'] = df.get('IMERG_Total_3day', np.nan)
    features['imerg_maxdaily_3day'] = df.get('IMERG_MaxDaily_3day', np.nan)
    features['recent_water_proxy'] = df.get('Recent_Water_Proxy', np.nan)
    # GloFAS NRT features
    features['glofas_discharge'] = df.get('GloFAS_Discharge', np.nan)
    features['glofas_flood_prob'] = df.get('GloFAS_Flood_Prob', np.nan)
    features['viirs_flood_mask'] = df.get('viirs_flood_mask', np.nan)

    # Land cover signal (built-up indicator)
    if 'Land_Cover' in df.columns:
        features['built_up'] = (df['Land_Cover'] == 50).astype(int)
    else:
        features['built_up'] = 0

    # Soil hydraulic conductivity PTF ensemble (Cosby implemented; scaffold others)
    sand_pct_raw = df.get('Sand_Content', np.nan)
    clay_pct_raw = df.get('Clay_Content', np.nan)

    # If SoilGrids units are g/kg (0-1000), convert to %
    sand_pct = pd.to_numeric(sand_pct_raw, errors='coerce')
    clay_pct = pd.to_numeric(clay_pct_raw, errors='coerce')

    if sand_pct is not None:
        sand_pct = np.where(sand_pct > 100, sand_pct / 10.0, sand_pct)
    if clay_pct is not None:
        clay_pct = np.where(clay_pct > 100, clay_pct / 10.0, clay_pct)

    def _cosby_ksat_mps(sand_p, clay_p):
        with np.errstate(over='ignore', invalid='ignore'):
            ksat_cm_per_hr = 0.491 * np.exp(-0.884 + 0.0153 * sand_p - 0.000550 * (clay_p ** 2))
            return ksat_cm_per_hr * (0.01 / 3600.0)

    # Compute Cosby Ksat and add as explicit feature
    ksat_cosby = _cosby_ksat_mps(sand_pct, clay_pct) if sand_pct is not None and clay_pct is not None else np.nan
    features['ksat_cosby_mps'] = ksat_cosby

    # Placeholder for Saxton & Rawls and Rosetta (requires additional inputs). Set as NaN until implemented.
    # TODO: Implement Saxton & Rawls and Rosetta PTFs for ksat estimation when data available.
    features['ksat_saxton_rawls_mps'] = np.nan
    features['ksat_rosetta_mps'] = np.nan

    # Blended robust Ksat (median across available estimates)
    ksat_stack = np.vstack([
        np.atleast_1d(ksat_cosby if isinstance(ksat_cosby, np.ndarray) else np.full(len(features), ksat_cosby)),
        # Future: add arrays for saxton_rawls and rosetta when implemented
    ])
    # Median across available non-NaN rows
    with np.errstate(invalid='ignore'):
        ksat_blend = np.nanmedian(ksat_stack, axis=0)
    features['ksat_blend_mps'] = ksat_blend

    # Backward-compat: maintain 'ksat_mps' as the blended value
    features['ksat_mps'] = features['ksat_blend_mps']

    # Hydrologic interaction features
    features['ksat_slope_interaction'] = features['ksat_mps'] * features['slope']
    features['twi_precip_interaction'] = features['twi'] * features['total_precipitation']

    # Final clean-up
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce')
        
    return features

# --- Real-World Data Integration ---
def download_and_prepare_gfd_data() -> pd.DataFrame | None:
    """
    Download and prepare flood events from the Global Flood Database (GFD).
    Source: Cloud to Street MODIS GFD curated CSV.
    """
    gfd_url = (
        "https://raw.githubusercontent.com/cloudtostreet/MODIS_GlobalFloodDatabase/"
        "master/data/gfd_qcdatabase_2019_08_01.csv"
    )
    cache_path = Path("gfd_flood_catalog.csv")

    gfd_df: pd.DataFrame | None = None
    if cache_path.exists():
        logging.info("Loading cached GFD data...")
        gfd_df = pd.read_csv(cache_path, parse_dates=["Began", "Ended"])
    else:
        logging.info(f"Downloading GFD data from: {gfd_url}")
        try:
            df = pd.read_csv(gfd_url)
            df = df.rename(
                columns={
                    "start_date": "Began",
                    "end_date": "Ended",
                    "long": "longitude",
                    "lat": "latitude",
                }
            )

            keep_cols = ["Began", "Ended", "longitude", "latitude"]
            df = df[keep_cols].copy()
            df.dropna(inplace=True)
            df["Began"] = pd.to_datetime(df["Began"], errors="coerce")
            df["Ended"] = pd.to_datetime(df["Ended"], errors="coerce")
            df.dropna(subset=["Began", "Ended"], inplace=True)

            df["event_year"] = df["Began"].dt.year.astype(int)
            df["y_binary"] = 1
            df["y_ordinal"] = pd.Series([pd.NA] * len(df), dtype="Int64")

            df.to_csv(cache_path, index=False)
            logging.info(f"Successfully downloaded and cached {len(df)} GFD events.")
            gfd_df = df
        except Exception as e:
            logging.error(f"Failed to download or process GFD data: {e}", exc_info=True)
            return None

    if gfd_df is None:
        return None
        
    return gfd_df


def augment_with_negative_samples(
    positive_df: pd.DataFrame,
    neg_to_pos_ratio: float = 2.0,
    min_distance_km: float = 50.0,
    max_trials: int = 200000,
) -> pd.DataFrame:
    """
    Generate negative samples to counter selection bias.
    """
    if positive_df is None or len(positive_df) == 0:
        logging.warning("No positive events provided; skipping negative sampling.")
        return positive_df

    try:
        from sklearn.neighbors import BallTree
    except Exception as e:
        logging.error(f"scikit-learn is required for negative sampling (BallTree): {e}")
        return positive_df

    # Optional: Earth Engine water mask
    ee_ok = False
    try:
        import ee  # type: ignore
        ee.Initialize()
        water_mask = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(90)
        ee_ok = True
    except Exception:
        logging.warning("GEE not available; proceeding without permanent water screening.")
        water_mask = None

    latlon_pos = positive_df[["latitude", "longitude"]].to_numpy(dtype=float)
    pos_rad = np.deg2rad(latlon_pos)
    tree = BallTree(pos_rad, metric="haversine")

    lat_min, lon_min = latlon_pos.min(axis=0)
    lat_max, lon_max = latlon_pos.max(axis=0)
    lat_pad = max(0.5, 0.1 * (lat_max - lat_min + 1e-6))
    lon_pad = max(0.5, 0.1 * (lon_max - lon_min + 1e-6))
    lat_min_b, lat_max_b = np.clip([lat_min - lat_pad, lat_max + lat_pad], -90, 90)
    lon_min_b, lon_max_b = np.clip([lon_min - lon_pad, lon_max + lon_pad], -180, 180)

    target_neg = int(np.ceil(len(positive_df) * float(neg_to_pos_ratio)))
    neg_rows = []
    trials = 0
    r_earth_km = 6371.0088
    min_dist_rad = float(min_distance_km) / r_earth_km
    rng = np.random.RandomState(42)

    began_vals = positive_df["Began"].to_numpy()
    ended_vals = positive_df.get("Ended", positive_df["Began"]).to_numpy()
    years_vals = positive_df["event_year"].to_numpy()

    while len(neg_rows) < target_neg and trials < max_trials:
        trials += 1
        rand_lat = rng.uniform(lat_min_b, lat_max_b)
        rand_lon = rng.uniform(lon_min_b, lon_max_b)

        dist_rad, _ = tree.query(np.deg2rad([[rand_lat, rand_lon]]), k=1)
        if float(dist_rad[0][0]) < min_dist_rad:
            continue

        if ee_ok and water_mask is not None:
            try:
                point = ee.Geometry.Point([rand_lon, rand_lat])
                occ = water_mask.sample(region=point, scale=100).first()
                is_water = bool(ee.Number(occ.get("occurrence")).gt(0).getInfo()) if occ else False
                if is_water:
                    continue
            except Exception:
                pass

        idx = rng.randint(0, len(positive_df))
        neg_rows.append(
            {
                "Began": began_vals[idx],
                "Ended": ended_vals[idx],
                "longitude": rand_lon,
                "latitude": rand_lat,
                "event_year": int(years_vals[idx]) if not pd.isna(years_vals[idx]) else 0,
                "y_binary": 0,
                "y_ordinal": pd.NA,
            }
        )

    if len(neg_rows) < target_neg:
        logging.warning(
            f"Generated {len(neg_rows)} negatives (< target {target_neg}) after {trials} trials. Proceeding."
        )
    else:
        logging.info(f"Generated {len(neg_rows)} negative samples in {trials} trials.")

    negatives = pd.DataFrame(neg_rows)
    combined = (
        pd.concat([positive_df, negatives], ignore_index=True)
        .drop_duplicates(subset=["latitude", "longitude", "Began"], keep="first")
        .reset_index(drop=True)
    )
    return combined

class ProductionRiskModel:
    def __init__(self):
        self.classifier = None
        self.is_trained = False
        self.feature_columns = engineer_features_vectorized(pd.DataFrame()).columns.tolist()
        self.explainer = None
        self.risk_threshold = 0.5
        try:
            import config as _cfg
            self.conformal_alpha: float = float(getattr(_cfg, 'CONFORMAL_ALPHA', 0.1))
        except Exception:
            self.conformal_alpha: float = 0.1
        self.conformal_q_: float | None = None
        self.calibration_: Dict[str, float] | None = None

    def _determine_threshold_via_pr(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None) -> float:
        """
        Determine probability threshold via cross-validated Precision-Recall curve.
        """
        if groups is not None:
            import config as _cfg
            n_splits = int(getattr(_cfg, 'CV_N_SPLITS', 5))
            n_splits = min(n_splits, len(np.unique(groups)))
            # <<< FIX: Ensure n_splits is at least 2 for GroupKFold.
            if n_splits < 2:
                n_splits = 2
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(X, y, groups)
        else:
            import config as _cfg
            n_splits = int(getattr(_cfg, 'CV_N_SPLITS', 5))
            n_splits = min(n_splits, y.value_counts().min())
            # <<< FIX: Ensure n_splits is at least 2 for StratifiedKFold.
            if n_splits < 2:
                n_splits = 2
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = splitter.split(X, y)

        oof_probs = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            clf = xgb.XGBClassifier(
                objective='binary:logistic',
                base_score=0.5,
                n_estimators=300,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=42,
            )
            clf.fit(X_tr, y_tr)
            oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y, oof_probs)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
        best_idx = int(np.nanargmax(f1_scores))
        if best_idx == 0 or len(thresholds) == 0:
            best_threshold = 0.5
        else:
            best_threshold = float(thresholds[best_idx - 1])

        ap = average_precision_score(y, oof_probs)
        roc = roc_auc_score(y, oof_probs)
        brier = brier_score_loss(y, oof_probs)
        logging.info(f"Cross-validated metrics — AP: {ap:.3f}, ROC-AUC: {roc:.3f}, Brier: {brier:.3f}, Best PR threshold: {best_threshold:.3f}")
        return best_threshold

    def load_pretrained_model(self, model_path: str = "1.pkl"):
        """
        Load pretrained XGBoost model from pickle file.
        """
        import joblib
        import os
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained model not found at {model_path}")
        
        logging.info(f"Loading pretrained model from {model_path}")
        self.pipeline = joblib.load(model_path)
        
        # Extract the classifier and feature columns from the pipeline
        if hasattr(self.pipeline, 'named_steps'):
            # Sklearn pipeline
            self.classifier = self.pipeline.named_steps['classifier']
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Get feature names after preprocessing
            try:
                if hasattr(preprocessor, 'named_transformers_'):
                    num_cols = preprocessor.named_transformers_['num'].get_feature_names_out()
                    cat_cols = []
                    if 'cat' in preprocessor.named_transformers_:
                        # Check if the OneHotEncoder is fitted
                        ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
                        if hasattr(ohe, 'get_feature_names_out'):
                            try:
                                cat_cols = ohe.get_feature_names_out()
                            except Exception as e:
                                logging.warning(f"OneHotEncoder not fitted, using manual feature names: {e}")
                                # Use manual feature names based on the pretrained model requirements
                                cat_cols = []
                        else:
                            cat_cols = []
                    self.feature_columns = list(num_cols) + list(cat_cols)
                else:
                    self.feature_columns = None
            except Exception as e:
                logging.warning(f"Could not extract feature names from preprocessor: {e}")
                # Use the expected feature names from the pretrained model
                self.feature_columns = self._get_expected_feature_names()
        else:
            # Direct XGBoost model
            self.classifier = self.pipeline
            self.feature_columns = self._get_expected_feature_names()
        
        # Set threshold (you may need to adjust this based on your model)
        self.risk_threshold = 0.5
        
        # Initialize explainer
        self.explainer = shap.TreeExplainer(self.classifier)
        
        self.is_trained = True
        logging.info("Pretrained model loaded successfully")
    
    def _get_expected_feature_names(self):
        """
        Get the expected feature names for the pretrained model.
        Based on the features used in the pretrained model from requirements.txt.
        """
        return [
            'latitude', 'longitude', 'year', 'month', 'day_of_year', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
            'gee_chirps_precip_sum_1d_before', 'gee_chirps_precip_sum_3d_before', 'gee_chirps_precip_sum_7d_before', 
            'gee_chirps_precip_sum_30d_before', 'gee_chirps_precip_max_3d', 'gee_chirps_precip_max_7d', 
            'gee_chirps_precip_intensity_7d_before', 'gee_gpm_precip_sum_1d_before', 'gee_gpm_precip_sum_7d_before', 
            'gee_gpm_precip_max_rate_event_mmhr', 'gee_gpm_precip_sum_max_6h_event_mm', 'gee_elevation_mean', 
            'gee_slope_mean', 'gee_merit_upa_mean', 'gee_gsw_occurrence_mean', 'gee_gsw_seasonality_mean', 
            'gee_ndvi_mean', 'snowmelt_risk_spring', 'snow_region', 'temp_elevation_interaction', 
            'is_snowmelt_season', 'is_monsoon_season', 'is_ice_season', 'tropical_region', 'temperate_region', 
            'northern_region', 'confidence_score', 'num_admin_regions'
        ]

    def _engineer_features_for_pretrained_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features to match the pretrained XGBoost model requirements.
        Based on the features used in the pretrained model from requirements.txt.
        """
        features = pd.DataFrame(index=df.index)
        
        # Geographic features
        features['latitude'] = df.get('latitude', df.get('Latitude', np.nan))
        features['longitude'] = df.get('longitude', df.get('Longitude', np.nan))
        
        # Temporal features (if available)
        if 'Began' in df.columns:
            event_date = pd.to_datetime(df['Began'], errors='coerce')
            features['year'] = event_date.dt.year
            features['month'] = event_date.dt.month
            features['day_of_year'] = event_date.dt.dayofyear
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12.0)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12.0)
            features['doy_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365.0)
            features['doy_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365.0)
        else:
            # Default values if no date available
            features['year'] = 2024
            features['month'] = 6
            features['day_of_year'] = 150
            features['month_sin'] = 0.0
            features['month_cos'] = 1.0
            features['doy_sin'] = 0.0
            features['doy_cos'] = 1.0
        
        # Precipitation features (CHIRPS)
        features['gee_chirps_precip_sum_1d_before'] = df.get('gee_chirps_precip_sum_1d_before', 
                                                             df.get('Total_Precipitation', np.nan))
        features['gee_chirps_precip_sum_3d_before'] = df.get('gee_chirps_precip_sum_3d_before', np.nan)
        features['gee_chirps_precip_sum_7d_before'] = df.get('gee_chirps_precip_sum_7d_before', np.nan)
        features['gee_chirps_precip_sum_30d_before'] = df.get('gee_chirps_precip_sum_30d_before', np.nan)
        features['gee_chirps_precip_max_3d'] = df.get('gee_chirps_precip_max_3d', np.nan)
        features['gee_chirps_precip_max_7d'] = df.get('gee_chirps_precip_max_7d', np.nan)
        features['gee_chirps_precip_intensity_7d_before'] = df.get('gee_chirps_precip_intensity_7d_before', np.nan)
        
        # GPM precipitation features
        features['gee_gpm_precip_sum_1d_before'] = df.get('gee_gpm_precip_sum_1d_before', np.nan)
        features['gee_gpm_precip_sum_7d_before'] = df.get('gee_gpm_precip_sum_7d_before', np.nan)
        features['gee_gpm_precip_max_rate_event_mmhr'] = df.get('gee_gpm_precip_max_rate_event_mmhr', np.nan)
        features['gee_gpm_precip_sum_max_6h_event_mm'] = df.get('gee_gpm_precip_sum_max_6h_event_mm', np.nan)
        
        # Topographic features
        features['gee_elevation_mean'] = df.get('gee_elevation_mean', df.get('Elevation', np.nan))
        features['gee_slope_mean'] = df.get('gee_slope_mean', df.get('Slope', np.nan))
        features['gee_merit_upa_mean'] = df.get('gee_merit_upa_mean', np.nan)
        
        # Surface water features
        features['gee_gsw_occurrence_mean'] = df.get('gee_gsw_occurrence_mean', np.nan)
        features['gee_gsw_seasonality_mean'] = df.get('gee_gsw_seasonality_mean', np.nan)
        
        # Vegetation features
        features['gee_ndvi_mean'] = df.get('gee_ndvi_mean', df.get('NDVI', np.nan))
        
        # Engineered features
        features['snowmelt_risk_spring'] = df.get('snowmelt_risk_spring', 0)
        features['snow_region'] = df.get('snow_region', 0)
        features['temp_elevation_interaction'] = df.get('temp_elevation_interaction', np.nan)
        features['is_snowmelt_season'] = df.get('is_snowmelt_season', 0)
        features['is_monsoon_season'] = df.get('is_monsoon_season', 0)
        features['is_ice_season'] = df.get('is_ice_season', 0)
        features['tropical_region'] = df.get('tropical_region', 0)
        features['temperate_region'] = df.get('temperate_region', 0)
        features['northern_region'] = df.get('northern_region', 0)
        
        # Additional features that might be present
        features['confidence_score'] = df.get('confidence_score', 1.0)
        features['num_admin_regions'] = df.get('num_admin_regions', 1)
        
        # Fill NaN values with appropriate defaults
        features = features.fillna(0.0)
        
        return features

    def _fit_conformal_calibration(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Split-conformal calibration for probabilistic intervals."""
        class_counts = y.value_counts()
        if len(class_counts) < 2 or class_counts.min() < 2:
            logging.warning("Skipping conformal calibration: dataset too small or lacks samples in each class.")
            self.conformal_q_ = None
            self.calibration_ = None
            return

        try:
            from sklearn.model_selection import train_test_split
            X_tr, X_cal, y_tr, y_cal = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            calib_clf = xgb.XGBClassifier(
                objective='binary:logistic',
                base_score=0.5,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=42,
            )
            calib_clf.fit(X_tr, y_tr)
            p_cal = calib_clf.predict_proba(X_cal)[:, 1]
            residuals = np.abs(y_cal.astype(float).to_numpy() - p_cal)
            n = len(residuals)
            k = int(np.ceil((n + 1) * (1 - self.conformal_alpha)))
            k = min(max(1, k), n)
            q = float(np.partition(residuals, k - 1)[k - 1])
            self.conformal_q_ = q
            widths = np.minimum(1.0, p_cal + q) - np.maximum(0.0, p_cal - q)
            low_q = float(np.quantile(widths, 1.0 / 3.0))
            high_q = float(np.quantile(widths, 2.0 / 3.0))
            self.calibration_ = {
                "uncertainty_low": low_q,
                "uncertainty_high": high_q,
                "alpha": self.conformal_alpha,
                "conformal_q": q,
                "operating_threshold": self.risk_threshold,
            }
            logging.info(
                f"Conformal calibration: alpha={self.conformal_alpha:.3f}, q={q:.4f}, width tertiles=({low_q:.4f}, {high_q:.4f})"
            )
        except Exception as e:
            logging.error(f"Conformal calibration failed: {e}", exc_info=True)
            self.conformal_q_ = None
            self.calibration_ = None

    def predict_per_pixel(self, df_real_data: pd.DataFrame) -> pd.DataFrame:
        """Predicts per-pixel flood risk probability with SHAP explainability using pretrained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be loaded before prediction. Call load_pretrained_model() first.")

        logging.info(f"Predicting risk probability for {len(df_real_data)} pixels...")
        
        # Always use manual feature engineering to avoid pipeline issues
        feature_df = self._engineer_features_for_pretrained_model(df_real_data)
        
        # Ensure feature columns match what the model expects
        if self.feature_columns:
            # Reorder columns to match expected order
            missing_cols = [col for col in self.feature_columns if col not in feature_df.columns]
            for col in missing_cols:
                feature_df[col] = 0.0  # Fill missing columns with 0
            
            # Reorder to match expected order
            feature_df = feature_df[self.feature_columns]
        
        # Use the classifier directly
        proba = self.classifier.predict_proba(feature_df)[:, 1]

        if self.conformal_q_ is not None:
            q = float(self.conformal_q_)
            lower = np.clip(proba - q, 0.0, 1.0)
            upper = np.clip(proba + q, 0.0, 1.0)
            interval_width = upper - lower
        else:
            lower = np.clip(proba, 0.0, 1.0)
            upper = np.clip(proba, 0.0, 1.0)
            interval_width = np.zeros_like(proba)

        # SHAP explainability
        try:
            shap_values = self.explainer.shap_values(feature_df)
        except Exception as e:
            logging.warning(f"SHAP calculation failed: {e}")
            shap_values = np.zeros((len(df_real_data), len(self.feature_columns) if self.feature_columns else 10))

        results_df = pd.DataFrame({
            'Final_Risk_Score': np.clip(proba, 0, 1),
            'Risk_Probability': np.clip(proba, 0, 1),
            'Uncertainty': interval_width,
            'Risk_Lower_90': lower,
            'Risk_Upper_90': upper,
            'SHAP_Values': list(shap_values if isinstance(shap_values, np.ndarray) else shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values)
        }, index=df_real_data.index)

        logging.info("Per-pixel probability prediction with XAI complete.")
        return results_df

# LSTM training functions removed - using pretrained XGBoost model instead