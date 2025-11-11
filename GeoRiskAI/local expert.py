# local_expert.py
"""
Local expert + smart ensemble wrapper for your pretrained global ensemble.

Usage:
    from local_expert import SmartLocalEnsemble
    sle = SmartLocalEnsemble(global_model_path="models/ensemble_v1.pkl", global_dataset_path="data/global_dataset.csv")
    result = sle.predict_for_location(lat, lon, feature_row, event_date=..., radius_km=500)
"""

import os
import math
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, mean_squared_error
from catboost import CatBoostClassifier, Pool
from typing import Tuple, Dict, Any, Optional

# -------------------------------------------------------------------
# Tunables and defaults (adapt to your config)
# -------------------------------------------------------------------
DEFAULT_RADIUS_KM = 500.0
MIN_LOCAL_SAMPLES = 200               # if fewer, fall back to adjusted weighting
RECENT_YEARS = 10                     # prefer recent events (optional filter)
N_CALIBRATION_FOLDS = 5
RBF_SIGMA_KM = 250.0                  # gaussian kernel sigma for distance weighting
EPS = 1e-6

# Use your best CatBoost params (from 3.py)
LOCAL_CAT_PARAMS = {
    'iterations': 2050,
    'depth': 10,
    'learning_rate': 0.034497224452741705,
    'l2_leaf_reg': 1.1793622591307933,
    'subsample': 0.8118543026413315,
    'colsample_bylevel': 0.9195639780831663,
    'random_strength': 0.6616953985666554,
    'bagging_temperature': 1.088585694737754,
    'border_count': 512,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0,
    'random_seed': 42,
}

# Your feature list (must match the dataset)
# If you already import FORECASTING_FEATURE_SET from 3.py, you can reuse that instead.
# For safety we allow injection via constructor.
# -------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    # returns great-circle dist (km)
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def gaussian_kernel_distance_weight(dist_km, sigma_km=RBF_SIGMA_KM):
    return math.exp(-0.5 * (dist_km / (sigma_km + EPS))**2)

def safe_load_model(path):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Global model not found: {path}")
    try:
        mdl = joblib.load(path)
        return mdl
    except Exception as e:
        # try pickle
        import pickle
        with open(path, 'rb') as f:
            mdl = pickle.load(f)
        return mdl

class SmartLocalEnsemble:
    def __init__(
        self,
        global_model_path: str,
        global_dataset_path: str,
        feature_list: Optional[list] = None,
        static_features: Optional[list] = None,
        dynamic_features: Optional[list] = None,
        local_cat_params: dict = None,
    ):
        self.global_model_path = global_model_path
        self.global_dataset_path = global_dataset_path
        self.global_model = safe_load_model(global_model_path)
        self.feature_list = feature_list
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.local_cat_params = local_cat_params or LOCAL_CAT_PARAMS

        # load dataset index (lightweight) once
        if global_dataset_path and os.path.exists(global_dataset_path):
            self._global_df = pd.read_csv(global_dataset_path, low_memory=False)
            # ensure lat/lon present
            if not {'latitude', 'longitude'}.issubset(set(self._global_df.columns)):
                raise ValueError("global_dataset must contain latitude/longitude columns")
        else:
            self._global_df = None

    # ---------- local sample selection ----------
    def select_local_samples(self, lat: float, lon: float, radius_km: float = DEFAULT_RADIUS_KM, recent_years: int = RECENT_YEARS):
        df = self._global_df
        if df is None:
            return pd.DataFrame([])

        # compute distances (vectorized)
        lats = df['latitude'].to_numpy(dtype=float)
        lons = df['longitude'].to_numpy(dtype=float)
        vec_haversine = np.vectorize(haversine_km)
        dists = vec_haversine(lat, lon, lats, lons)
        df = df.assign(_dist_km=dists)

        # optional recency filter
        if 'flood_start_date' in df.columns:
            try:
                df['flood_start_date'] = pd.to_datetime(df['flood_start_date'], errors='coerce')
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * recent_years)
                local = df[(df['_dist_km'] <= radius_km) & (df['flood_start_date'] >= cutoff)]
                if len(local) < MIN_LOCAL_SAMPLES:
                    # relax recency if too few
                    local = df[df['_dist_km'] <= radius_km]
                # final sort by distance
                return local.sort_values('_dist_km').reset_index(drop=True)
            except Exception:
                return df[df['_dist_km'] <= radius_km].sort_values('_dist_km').reset_index(drop=True)
        else:
            return df[df['_dist_km'] <= radius_km].sort_values('_dist_km').reset_index(drop=True)

    # ---------- train local expert ----------
    def train_local_expert(self, local_df: pd.DataFrame, label_col: str = 'target_future', use_features: Optional[list] = None):
        if local_df is None or len(local_df) == 0:
            return None, None, {'n_samples': 0}

        features = use_features or self.feature_list
        if features is None:
            raise ValueError("feature list required for training local expert")

        # only keep necessary columns & drop rows missing label
        local = local_df.copy()
        if label_col not in local.columns:
            raise ValueError(f"label column {label_col} not in local dataset")
        local = local.dropna(subset=[label_col])

        X = local[features].copy()
        y = local[label_col].astype(int)

        # minimal preproc: replace infinities, keep numeric only for training
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = X.fillna(X.median())

        # If too few positive/neg samples, skip training
        if y.nunique() < 2 or len(y) < 30:
            return None, None, {'n_samples': len(y), 'reason': 'too_few_samples'}

        # train-catboost with CV to estimate oof variance (uncertainty)
        cv = StratifiedKFold(n_splits=min(N_CALIBRATION_FOLDS, max(2, int(len(y)/30))), shuffle=True, random_state=42)
        oof = np.zeros(len(y), dtype=float)
        idx = 0
        for train_idx, val_idx in cv.split(X, y):
            Xtr, Xva = X.iloc[train_idx], X.iloc[val_idx]
            ytr, yva = y.iloc[train_idx], y.iloc[val_idx]
            pool_tr = Pool(Xtr, ytr)
            pool_va = Pool(Xva, yva)
            model = CatBoostClassifier(**self.local_cat_params)
            model.fit(pool_tr, eval_set=pool_va, verbose=0, use_best_model=False)
            proba = model.predict_proba(Xva)[:, 1]
            oof[val_idx] = proba

        # fit final model on all local data
        final_model = CatBoostClassifier(**self.local_cat_params)
        final_model.fit(Pool(X, y), verbose=0)
        # calibrator
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof, y.to_numpy())

        # estimate oof variance (uncertainty proxy)
        variance = float(np.nanvar(oof))
        # estimate local performance by logloss on oof
        oof_logloss = float(log_loss(y.to_numpy(), np.clip(oof, EPS, 1-EPS)))
        oof_rmse = float(mean_squared_error(y.to_numpy(), oof, squared=False))

        meta = {
            'n_samples': len(y),
            'oof_var': variance,
            'oof_logloss': oof_logloss,
            'oof_rmse': oof_rmse,
        }
        return final_model, iso, meta

    # ---------- predict helpers ----------
    def _predict_global(self, features_row: pd.DataFrame, lat: float, lon: float):
        """
        A robust wrapper to obtain global ensemble prediction.
        The global_model might be:
          - a scikit-like object with predict_proba(X)
          - your custom WeightedEnsemble requiring multiple frames
        We'll try multiple options.
        """
        mdl = self.global_model
        if mdl is None:
            return None, {'reason': 'no_global_model'}

        # try direct predict_proba on a single DF
        try:
            if hasattr(mdl, 'predict_proba'):
                proba = mdl.predict_proba(features_row)[:, 1]
                return float(proba.ravel()[0]), {'mode': 'single_df_predict_proba'}
        except Exception:
            pass

        # try ensemble-like interface: predict_proba(X_static=..., X_dynamic=..., X_full=...)
        try:
            # build frames
            frames = {}
            if self.static_features:
                frames['static'] = features_row[self.static_features]
            if self.dynamic_features:
                frames['dynamic'] = features_row[self.dynamic_features]
            frames['full'] = features_row[self.feature_list] if self.feature_list else features_row

            if hasattr(mdl, 'predict_proba'):
                # some ensembles accept (X_static, X_dynamic, X_full) directly
                proba = mdl.predict_proba(frames) if isinstance(frames, (list, tuple)) else mdl.predict_proba(
                    X_static=frames.get('static'), X_dynamic=frames.get('dynamic'), X_full=frames.get('full'),
                    feature_frames=frames
                )
                # attempt to extract
                if isinstance(proba, np.ndarray):
                    return float(proba.ravel()[0]), {'mode': 'ensemble_call'}
        except Exception:
            pass

        # fallback: try calling predict and treat as proba
        try:
            p = mdl.predict(features_row)
            return float(p.ravel()[0]), {'mode': 'predict_fallback'}
        except Exception:
            return None, {'reason': 'global_predict_failed'}

    # ---------- weight computation ----------
    def compute_weights(self, global_meta: Dict[str, Any], local_meta: Dict[str, Any], local_df_len: int, avg_distance_km: float):
        """
        Combine:
          - performance: use 1 / RMSE (or exp(-logloss))
          - distance kernel: gaussian of avg_distance_km
          - density: normalized local sample count (min 1)
        Returns normalized [w_global, w_local]
        """
        # performance proxies
        perf_global = 1.0
        perf_local = 1.0
        # if meta contains oof_rmse or oof_logloss, use them
        if global_meta is None:
            perf_global = 1.0
        else:
            g_rmse = global_meta.get('oof_rmse', None)
            g_ll = global_meta.get('oof_logloss', None)
            if g_rmse:
                perf_global = 1.0 / (g_rmse + EPS)
            elif g_ll:
                perf_global = math.exp(-g_ll)

        if local_meta is None:
            perf_local = 0.5  # penalize missing local meta
        else:
            l_rmse = local_meta.get('oof_rmse', None)
            l_ll = local_meta.get('oof_logloss', None)
            if l_rmse:
                perf_local = 1.0 / (l_rmse + EPS)
            elif l_ll:
                perf_local = math.exp(-l_ll)

        # distance kernel (local closer -> boost local)
        dist_kernel = gaussian_kernel_distance_weight(avg_distance_km)
        # local density factor (sigmoid-like)
        density = float(local_df_len) / (float(local_df_len) + 1000.0)  # saturates at 1
        # uncertainty factor: prefer smaller variance (inverse-variance)
        local_var = local_meta.get('oof_var', None) if local_meta else None
        uncertainty_local = 1.0 / (local_var + EPS) if local_var is not None else 1.0

        # combine multiplicatively
        raw_global = perf_global * (1.0)  # global gets baseline
        raw_local = perf_local * dist_kernel * density * uncertainty_local

        # ensure non-negative
        raw_global = max(raw_global, EPS)
        raw_local = max(raw_local, EPS)

        # normalize to sum 1
        s = raw_global + raw_local
        w_global = raw_global / s
        w_local = raw_local / s
        return float(w_global), float(w_local), {
            'perf_global': perf_global,
            'perf_local': perf_local,
            'dist_kernel': dist_kernel,
            'density': density,
            'uncertainty_local': uncertainty_local,
            'raw': (raw_global, raw_local)
        }

    # ---------- main predict method ----------
    def predict_for_location(
        self,
        lat: float,
        lon: float,
        features_row: pd.Series,
        event_date: Optional[str] = None,
        radius_km: float = DEFAULT_RADIUS_KM,
        label_col: str = 'target_future',
        min_local_samples: int = MIN_LOCAL_SAMPLES
    ) -> Dict[str, Any]:
        """
        features_row: single-row pd.Series (feature names must match self.feature_list)
        returns dict with:
          - final_proba, final_label, confidence_score, weights, debug
        """
        # ensure features_row is DataFrame 1xN
        X_row = pd.DataFrame([features_row])

        # 1) global prediction
        try:
            global_proba, gmeta = self._predict_global(X_row, lat, lon)
        except Exception as e:
            global_proba, gmeta = None, {'error': str(e)}

        # 2) select local samples
        local_df = self.select_local_samples(lat, lon, radius_km=radius_km)
        n_local = 0 if local_df is None else len(local_df)

        # 3) train local expert if we have enough data
        local_model, local_calibrator, local_meta = None, None, None
        if n_local >= min_local_samples:
            try:
                local_model, local_calibrator, local_meta = self.train_local_expert(local_df, label_col=label_col, use_features=self.feature_list)
            except Exception as e:
                local_model, local_calibrator, local_meta = None, None, {'error': str(e)}
        else:
            local_meta = {'n_samples': n_local, 'reason': 'not_enough_local_samples'}

        # 4) global meta placeholder (if you have stored global oof metrics, you can attach them)
        global_meta = {}  # optionally populate with historic global ensemble oof stats

        # 5) compute avg distance among local samples (used by kernel)
        avg_dist = float(local_df['_dist_km'].mean()) if (local_df is not None and len(local_df)>0) else float(radius_km)

        # 6) compute weights
        w_global, w_local, weight_meta = self.compute_weights(global_meta, local_meta, n_local, avg_dist)

        # 7) local proba (if model exists) and calibration
        if local_model is not None:
            # get X row matching features used in local model
            X_row_local = X_row[self.feature_list]
            raw_local_proba = local_model.predict_proba(X_row_local)[:, 1]
            calibrated_local_proba = local_calibrator.predict(raw_local_proba)
            # clip
            local_proba = float(np.clip(calibrated_local_proba[0], EPS, 1-EPS))
            # estimate local uncertainty as variance from local_meta
            local_uncertainty = float(local_meta.get('oof_var', 0.0))
            local_perf = local_meta.get('oof_rmse', None)
        else:
            local_proba = None
            local_uncertainty = None
            local_perf = None

        # 8) combine
        # if global_proba missing -> return local_proba
        if global_proba is None and local_proba is not None:
            final_proba = local_proba
        elif global_proba is not None and local_proba is None:
            final_proba = global_proba
        elif global_proba is not None and local_proba is not None:
            final_proba = float(w_global * global_proba + w_local * local_proba)
        else:
            final_proba = None  # both missing

        # 9) compute confidence
        # simple heuristic: higher if both agree and low local variance and many local samples
        conf = 0.0
        if final_proba is not None:
            # agreement factor
            if local_proba is None:
                agree = 1.0
            else:
                agree = 1.0 - abs(global_proba - local_proba)
            # sample_count factor (saturates)
            sample_factor = min(1.0, n_local / 2000.0)
            # uncertainty factor
            uncertainty_factor = 1.0 / (1.0 + (local_meta.get('oof_var', 0.0) if local_meta else 0.0))
            conf = float(0.4 * agree + 0.3 * sample_factor + 0.3 * uncertainty_factor)
            conf = max(0.0, min(1.0, conf))

        debug = {
            'global_proba': global_proba,
            'local_proba': local_proba,
            'w_global': w_global,
            'w_local': w_local,
            'n_local': n_local,
            'avg_dist_km': avg_dist,
            'weight_meta': weight_meta,
            'local_meta': local_meta,
            'global_meta': global_meta,
            'gmeta_wrapper': gmeta
        }

        return {
            'final_proba': final_proba,
            'final_label': int(final_proba >= 0.5) if final_proba is not None else None,
            'confidence': conf,
            'weights': {'global': w_global, 'local': w_local},
            'debug': debug
        }
