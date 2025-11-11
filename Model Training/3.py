import sys
sys.path.insert(0, r"C:\Users\arham\CascadeProjects\Xgboost\1")
import os
import glob
import json
import itertools
import random
import time
import copy
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    make_scorer,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.0057  # final evaluation split (~0.57%)
N_FOLDS = 10  # cross-validation folds for all stages
STAGE1_VALIDATION_FRACTION = 0.20  # 80/20 split for weight selection
STAGE1_TRIALS = 5
STAGE1_FOLDS = 5

BEST_CATBOOST_PARAMS = {
    'iterations': 2050,
    'depth': 10,
    'learning_rate': 0.034497224452741705,
    'l2_leaf_reg': 1.1793622591307933,
    'subsample': 0.8118543026413315,
    'colsample_bylevel': 0.9195639780831663,
    'random_strength': 0.6616953985666554,
    'bagging_temperature': 1.088585694737754,
    'border_count': 512
}

BEST_XGBOOST_PARAMS = {
    'n_estimators': 850,
    'max_depth': 8,
    'learning_rate': 0.01737332862455038,
    'subsample': 0.9030614120011207,
    'colsample_bytree': 0.8539443128274192,
    'min_child_weight': 2.082400817315821,
    'gamma': 0.21544798764830475,
    'reg_alpha': 0.0444368331223305,
    'reg_lambda': 1.1980211816720427,
    'scale_pos_weight_factor': 1.1470689789882924
}

BEST_LIGHTGBM_PARAMS = {
    'n_estimators': 1350,
    'num_leaves': 95,
    'learning_rate': 0.07879855138490763,
    'max_depth': -1,
    'subsample': 0.929574246178696,
    'colsample_bytree': 0.8812337309009289,
    'min_child_samples': 20,
    'reg_alpha': 0.005437524149790202,
    'reg_lambda': 0.87445285534292,
    'min_split_gain': 0.010585021824577587,
    'scale_pos_weight_factor': 1.1575139862587251
}
FBETA = 2

# Feature set (from your script)
FORECASTING_FEATURE_SET = [
    'target_future',
    'doy1_sin', 'doy1_cos',
    'imerg_max_1h_intensity_mm', 'imerg_mean_1h_intensity_mm',
    'gee_imerg_sum_24h_mm', 'gee_imerg_max_1h_intensity_mm',
    'gee_imerg_max_3h_intensity_mm', 'gee_imerg_max_6h_intensity_mm',
    'gee_imerg_sum_3d_before_mm', 'gee_imerg_sum_7d_before_mm',
    'gee_imerg_max_daily_7d_mm', 'gee_imerg_intensity_3d_mm_per_day',
    'gee_imerg_intensity_7d_mm_per_day',
    'gee_smap_surface_soil_moisture', 'gee_smap_subsurface_soil_moisture',
    'gee_smap_soil_moisture_anomaly', 'gee_antecedent_moisture_proxy',
    'gee_flashiness_index_7d', 'gee_api_weighted_mm',
    'gee_flashiness_index', 'gee_saturation_proxy', 'gee_runoff_potential',
    'gee_aspect_mean', 'gee_elevation_mean', 'gee_slope_mean',
    'gee_gsw_occurrence_mean', 'gee_merit_upa_mean', 'gee_twi',
    'gee_ndvi_mean_30d', 'gee_ndbi_mean_30d', 'gee_ndwi_mean_30d',
    'gee_precip_x_slope',
    'glofas_forecast_control', 'glofas_forecast_mean', 'glofas_forecast_median',
    'glofas_forecast_std_dev', 'glofas_forecast_10th_percentile',
    'glofas_forecast_90th_percentile',
    'precip_soil_interaction', 'intensity_slope', 'glofas_saturation',
    'antecedent_twi', 'elevation_precip_ratio'
]

LABEL_COLUMN = "target_future"

# Zero is valid for these features
ZERO_IS_VALID = [
    'target', 'target_future', 'month_sin', 'month_cos', 'doy1_sin', 'doy1_cos',
    'imerg_max_1h_intensity_mm', 'imerg_mean_1h_intensity_mm',
    'gee_imerg_sum_24h_mm', 'gee_imerg_max_1h_intensity_mm',
    'gee_imerg_max_3h_intensity_mm', 'gee_imerg_max_6h_intensity_mm',
    'gee_imerg_sum_3d_before_mm', 'gee_imerg_sum_7d_before_mm',
    'gee_imerg_max_daily_7d_mm', 'gee_imerg_intensity_3d_mm_per_day',
    'gee_imerg_intensity_7d_mm_per_day', 'gee_slope_mean',
    'gee_aspect_mean', 'gee_elevation_mean', 'gee_merit_upa_mean',
    'gee_precip_x_slope', 'precip_soil_interaction',
    'intensity_slope', 'antecedent_twi', 'elevation_precip_ratio',
    'glofas_forecast_control', 'glofas_forecast_mean',
    'glofas_forecast_median', 'glofas_forecast_std_dev',
    'glofas_forecast_10th_percentile', 'glofas_forecast_90th_percentile',
]

# Baseline assumption for slow-changing (static) descriptors
STATIC_BASELINE_FEATURES = [
    'gee_aspect_mean',
    'gee_elevation_mean',
    'gee_slope_mean',
    'gee_gsw_occurrence_mean',
    'gee_merit_upa_mean',
    'gee_twi',
    'gee_ndvi_mean_30d',
    'gee_ndbi_mean_30d',
    'gee_ndwi_mean_30d',
    'gee_runoff_potential',
    'gee_saturation_proxy',
    'antecedent_twi',
    'elevation_precip_ratio'
]

# Hard overrides (can be edited by user)
STATIC_FORCE_FEATURES = {
    'gee_elevation_mean',
    'gee_slope_mean',
    'gee_aspect_mean',
    'gee_merit_upa_mean',
    'gee_gsw_occurrence_mean',
    'elevation_precip_ratio'
}

DYNAMIC_FORCE_FEATURES = set()

# Variability-driven partitioning thresholds (percentiles of std / CV)
STATIC_VARIANCE_PERCENTILE = 0.35
DYNAMIC_VARIANCE_PERCENTILE = 0.70

ENSEMBLE_THRESHOLD = 0.5
WEIGHT_SEARCH_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
FBETA_VALUES = np.round(np.arange(1.0, 3.01, 0.1), 1)
CALIBRATION_METHOD = 'isotonic'
CALIBRATION_FRACTION = 0.2
THRESHOLD_SEARCH_GRID = np.linspace(0.10, 0.90, 81)
ENSEMBLE_THRESHOLD_GRID = np.round(np.arange(0.10, 0.901, 0.01), 3)
PROBA_EPS = 1e-6
MIN_MODEL_WEIGHT = 0.05
MAX_MODEL_WEIGHT = 0.9

CATBOOST_PARAM_OPTIONS = {
    "iterations": [1600],
    "depth": [9],
    "learning_rate": [0.03],
    "l2_leaf_reg": [2],
    "subsample": [0.75],
    "colsample_bylevel": [0.85],
    "random_strength": [0.4],
    "bagging_temperature": [1.1],
    "border_count": [254]
}

XGBOOST_PARAM_OPTIONS = {
    "n_estimators": [500, 700, 900, 1100, 1300, 1500],
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.02, 0.035, 0.05, 0.07],
    "subsample": [0.75, 0.85, 0.95, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1.0, 2.0, 3.0, 4.0],
    "gamma": [0.0, 0.5, 1.0, 1.5],
    "reg_alpha": [0.0, 0.1, 0.2, 0.4],
    "reg_lambda": [0.8, 1.0, 1.2, 1.5],
    "scale_pos_weight_factor": [0.9, 1.0, 1.1, 1.3]
}

LIGHTGBM_PARAM_OPTIONS = {
    "n_estimators": [600, 800, 1000, 1200, 1500],
    "num_leaves": [31, 63, 95, 127],
    "learning_rate": [0.02, 0.035, 0.05, 0.07],
    "max_depth": [-1, 6, 8, 10],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_samples": [15, 25, 35, 45],
    "reg_alpha": [0.0, 0.1, 0.2, 0.4],
    "reg_lambda": [0.8, 1.0, 1.2, 1.5],
    "min_split_gain": [0.0, 0.1, 0.2],
    "scale_pos_weight_factor": [0.9, 1.0, 1.1, 1.3]
}

CATBOOST_MAX_COMBINATIONS = 1
XGBOOST_MAX_COMBINATIONS = 20
LIGHTGBM_MAX_COMBINATIONS = 20


def beta_to_key(beta: float) -> str:
    """Convert an F-beta value (float) into a safe dictionary key suffix."""
    return f"fbeta_{beta:.1f}".replace('.', '_')


def fp_fn_penalty_score(y_true, y_pred):
    """Return negative (FP + FN) so larger is better for scoring frameworks."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return -(fp + fn)


def safe_metric(metric_fn, *args, default=np.nan, **kwargs):
    """Safely compute metric, returning default when metric raises a ValueError."""
    try:
        return metric_fn(*args, **kwargs)
    except ValueError:
        return default


def compute_classification_metrics(y_true, y_pred, y_proba, beta_values, threshold):
    """Compute a rich set of classification metrics and confusion statistics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tp = int(tp)
    fp = int(fp)
    tn = int(tn)
    fn = int(fn)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    y_proba_clipped = np.clip(y_proba, PROBA_EPS, 1.0 - PROBA_EPS)

    metrics = {
        'threshold': float(threshold),
        'roc_auc': safe_metric(roc_auc_score, y_true, y_proba_clipped),
        'pr_auc': safe_metric(average_precision_score, y_true, y_proba_clipped),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'mcc': safe_metric(matthews_corrcoef, y_true, y_pred),
        'brier': brier_score_loss(y_true, y_proba_clipped),
        'log_loss': safe_metric(log_loss, y_true, y_proba_clipped),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'tpr': tpr,
        'tnr': tnr,
        'fpr': fpr,
        'fnr': fnr,
        'fp_plus_fn': fp + fn,
    }

    for beta in beta_values:
        metrics[beta_to_key(beta)] = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)

    return metrics


def total_error_objective(metrics):
    """Objective = -(FP + FN); larger value means lower total error."""
    if metrics is None:
        return -np.inf
    total_error = metrics.get('fp_plus_fn')
    if total_error is None or np.isnan(total_error):
        return -np.inf
    return -float(total_error)


def selection_objective(metrics):
    """Proxy objective using total error with MCC/Brier tie-breaks."""
    if metrics is None:
        return -np.inf
    total_err = metrics.get('fp_plus_fn')
    if total_err is None or np.isnan(total_err):
        return -np.inf
    mcc = metrics.get('mcc')
    brier = metrics.get('brier')
    # Larger score is better: minimize total error, maximize MCC, minimize Brier
    score = -float(total_err)
    if mcc is not None and not np.isnan(mcc):
        score += float(mcc)
    if brier is not None and not np.isnan(brier):
        score -= float(brier)
    return score


def compute_oof_predictions(model, X, y, cv, method='predict_proba'):
    """Generate out-of-fold predictions for ``model`` using CV splits."""
    X_array = X.to_numpy()
    y_array = y.to_numpy()
    oof_pred = np.zeros(len(y_array), dtype=float)
    for train_idx, test_idx in cv.split(X_array, y_array):
        clone_model = clone(model)
        clone_model.fit(X_array[train_idx], y_array[train_idx])
        if method == 'predict_proba':
            probs = clone_model.predict_proba(X_array[test_idx])[:, 1]
        else:
            probs = clone_model.decision_function(X_array[test_idx])
        oof_pred[test_idx] = probs
    return oof_pred


def apply_test_metrics(result_container, evaluation_output):
    """Update result dictionary with freshly computed test metrics."""
    metrics = evaluation_output['metrics']
    result_container['test_metrics'] = metrics
    result_container['test_proba'] = evaluation_output['proba']
    result_container['test_pred'] = evaluation_output['preds']
    result_container['test_roc_auc'] = metrics['roc_auc']
    result_container['test_fbeta_scores'] = {
        key: value for key, value in metrics.items() if key.startswith('fbeta_')
    }
    result_container['test_tp'] = metrics['tp']
    result_container['test_fp'] = metrics['fp']
    result_container['test_tn'] = metrics['tn']
    result_container['test_fn'] = metrics['fn']
    result_container['test_tpr'] = metrics['tpr']
    result_container['test_tnr'] = metrics['tnr']
    result_container['test_fpr'] = metrics['fpr']
    result_container['test_fnr'] = metrics['fnr']


def calibrate_model(base_model, X, y, n_folds):
    """Fit model and isotonic calibrator using OOF predictions."""
    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_raw = compute_oof_predictions(base_model, X_reset, y_reset, cv)

    unique_vals = np.unique(oof_raw)
    if unique_vals.size < 2:
        rng = np.random.RandomState(RANDOM_STATE)
        oof_raw = oof_raw + rng.normal(0.0, 1e-6, size=oof_raw.shape)

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_raw, y_reset)
    oof_calibrated = np.clip(calibrator.predict(oof_raw), PROBA_EPS, 1.0 - PROBA_EPS)

    final_model = clone(base_model)
    final_model.fit(X_reset, y_reset)

    return final_model, calibrator, oof_raw, oof_calibrated


class IsotonicCalibratedModel:
    """Wrapper exposing calibrated predict / predict_proba interface."""

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw)
        calibrated = np.clip(calibrated, PROBA_EPS, 1.0 - PROBA_EPS)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class WeightedEnsemble:
    """Helper for blending calibrated model outputs with fixed weights."""

    def __init__(
        self,
        models,
        weights,
        threshold,
        metadata=None,
        feature_sets=None,
        model_feature_roles=None
    ):
        self.models = models
        self.weights = tuple(weights)
        self.threshold = float(threshold)
        self.metadata = metadata or {}
        self.feature_sets = feature_sets or {}
        default_roles = {
            "catboost": "static",
            "xgboost": "dynamic",
            "lightgbm": "full"
        }
        self.model_feature_roles = model_feature_roles or default_roles

    def _resolve_frame(self, role, feature_frames):
        frame = feature_frames.get(role)
        if frame is None:
            raise ValueError(f"Missing feature frame for role '{role}'.")
        return frame

    def predict_proba(
        self,
        X_static=None,
        X_dynamic=None,
        X_full=None,
        feature_frames=None
    ):
        if feature_frames is None:
            feature_frames = {
                'static': X_static,
                'dynamic': X_dynamic,
                'full': X_full
            }

        p_cat = self.models["catboost"].predict_proba(
            self._resolve_frame(self.model_feature_roles.get("catboost", "static"), feature_frames)
        )[:, 1]
        p_xgb = self.models["xgboost"].predict_proba(
            self._resolve_frame(self.model_feature_roles.get("xgboost", "dynamic"), feature_frames)
        )[:, 1]
        p_lgb = self.models["lightgbm"].predict_proba(
            self._resolve_frame(self.model_feature_roles.get("lightgbm", "full"), feature_frames)
        )[:, 1]
        blended = (
            self.weights[0] * p_cat
            + self.weights[1] * p_xgb
            + self.weights[2] * p_lgb
        )
        return blended

    def predict(
        self,
        X_static=None,
        X_dynamic=None,
        X_full=None,
        feature_frames=None
    ):
        proba = self.predict_proba(X_static, X_dynamic, X_full, feature_frames)
        return (proba >= self.threshold).astype(int)


def sweep_thresholds(y_true, proba, beta_values, threshold_grid):
    """Sweep thresholds to minimize FP+FN; returns metrics per threshold."""
    best_metrics = None
    best_score = -np.inf
    all_metrics = []

    for threshold in threshold_grid:
        preds = (proba >= threshold).astype(int)
        metrics = compute_classification_metrics(y_true, preds, proba, beta_values, threshold)
        score = selection_objective(metrics)
        if score > best_score:
            best_score = score
            best_metrics = metrics
        all_metrics.append(metrics)

    return best_metrics, all_metrics


def find_best_ensemble_weights(cat_proba, xgb_proba, lgb_proba, y_true):
    """Search weight grid to minimize FP+FN; returns best weights, threshold, and metrics."""
    best_metrics = None
    best_threshold = ENSEMBLE_THRESHOLD
    best_weights = (1 / 3, 1 / 3, 1 / 3)
    best_blended = None
    weight_results = []

    for w_cat in WEIGHT_SEARCH_GRID:
        if w_cat > MAX_MODEL_WEIGHT:
            continue
        for w_xgb in WEIGHT_SEARCH_GRID:
            if w_xgb > MAX_MODEL_WEIGHT:
                continue
            w_lgb = 1.0 - w_cat - w_xgb
            if w_lgb < -1e-9:
                continue
            if w_lgb < MIN_MODEL_WEIGHT - 1e-9:
                continue
            if max(w_cat, w_xgb, w_lgb) > MAX_MODEL_WEIGHT + 1e-9:
                continue

            blended = w_cat * cat_proba + w_xgb * xgb_proba + w_lgb * lgb_proba
            metrics_best, metrics_all = sweep_thresholds(y_true, blended, FBETA_VALUES, ENSEMBLE_THRESHOLD_GRID)

            for metrics in metrics_all:
                metrics_with_weights = {
                    **metrics,
                    'weight_static': w_cat,
                    'weight_dynamic': w_xgb,
                    'weight_lightgbm': w_lgb
                }
                weight_results.append(metrics_with_weights)

                if selection_objective(metrics_with_weights) > selection_objective(best_metrics):
                    best_metrics = metrics_with_weights
                    best_threshold = metrics_with_weights['threshold']
                    best_weights = (w_cat, w_xgb, w_lgb)
                    best_blended = blended

    if best_metrics is None:
        raise RuntimeError("Weight sweep failed to produce metrics.")

    return best_weights, best_threshold, best_metrics, weight_results, best_blended


def evaluate_fixed_ensemble(cat_proba, xgb_proba, lgb_proba, y_true, weights, threshold):
    """Evaluate ensemble with fixed weights/threshold on provided probabilities."""
    blended = weights[0] * cat_proba + weights[1] * xgb_proba + weights[2] * lgb_proba
    preds = (blended >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, preds, blended, FBETA_VALUES, threshold)
    metrics['weight_static'] = weights[0]
    metrics['weight_dynamic'] = weights[1]
    metrics['weight_lightgbm'] = weights[2]
    metrics['threshold'] = threshold
    metrics['fbeta_default'] = metrics.get(beta_to_key(FBETA))
    return metrics, blended


def create_study_with_trials(study_name, base_params, iteration_schedule):
    """Create Optuna study and enqueue fixed iteration schedule around base params."""
    study = optuna.create_study(direction="maximize", study_name=study_name)
    if base_params:
        study.enqueue_trial(dict(base_params), skip_if_exists=False)
    if iteration_schedule:
        for idx, iterations in enumerate(iteration_schedule):
            if base_params and idx == 0 and base_params.get('iterations') == iterations:
                continue
            study.enqueue_trial({'iterations': iterations}, skip_if_exists=False)
    return study


def partition_features_by_variability(
    X: pd.DataFrame,
    baseline_static,
    force_static,
    force_dynamic,
    static_percentile,
    dynamic_percentile
):
    """Partition features into static/dynamic sets using variability heuristics."""

    available = [col for col in X.columns]
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # Compute coefficient of variation for numeric features (std normalized by mean)
    std_series = X[numeric_cols].std(axis=0, skipna=True)
    mean_series = X[numeric_cols].mean(axis=0, skipna=True).abs() + 1e-6
    cv_series = std_series / mean_series
    cv_series = cv_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Determine percentile cutoffs
    static_cut = cv_series.quantile(static_percentile) if len(cv_series) else 0.0
    dynamic_cut = cv_series.quantile(dynamic_percentile) if len(cv_series) else 0.0

    if np.isnan(static_cut):
        static_cut = 0.0
    if np.isnan(dynamic_cut):
        dynamic_cut = static_cut

    auto_static = set()
    auto_dynamic = set()

    baseline_static_set = set(baseline_static)
    force_static_set = set(force_static)
    force_dynamic_set = set(force_dynamic)

    for feature in available:
        if feature in force_static_set:
            auto_static.add(feature)
            continue
        if feature in force_dynamic_set:
            auto_dynamic.add(feature)
            continue

        score = cv_series.get(feature, None)
        if score is None:
            # Non-numeric or missing stats default to baseline membership
            if feature in baseline_static_set:
                auto_static.add(feature)
            else:
                auto_dynamic.add(feature)
            continue

        if score <= static_cut:
            auto_static.add(feature)
        elif score >= dynamic_cut:
            auto_dynamic.add(feature)
        else:
            if feature in baseline_static_set:
                auto_static.add(feature)
            else:
                auto_dynamic.add(feature)

    # Ensure all features assigned
    unassigned = set(available) - auto_static - auto_dynamic
    if unassigned:
        auto_dynamic.update(unassigned)

    # Provide debug output
    print("\nFeature variability partition summary:")
    print(f"  Static cutoff (CV ≤ {static_cut:.4f}), Dynamic cutoff (CV ≥ {dynamic_cut:.4f})")

    top_dynamic = cv_series[list(auto_dynamic & set(numeric_cols))].sort_values(ascending=False)
    top_static = cv_series[list(auto_static & set(numeric_cols))].sort_values()

    if not top_dynamic.empty:
        print("  Highest-variability (dynamic) features:")
        for feat, val in top_dynamic.head(5).items():
            print(f"    • {feat}: CV={val:.4f}")

    if not top_static.empty:
        print("  Lowest-variability (static) features:")
        for feat, val in top_static.head(5).items():
            print(f"    • {feat}: CV={val:.4f}")

    return sorted(auto_static), sorted(auto_dynamic)


def sample_param_grid(option_dict, max_combinations, random_state):
    """Sample parameter combinations from an option dictionary."""
    keys = list(option_dict.keys())
    value_lists = [option_dict[k] for k in keys]

    total = 1
    for values in value_lists:
        total *= len(values)

    if total <= max_combinations:
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
        return combos, total

    rng = random.Random(random_state)
    combos = []
    seen = set()
    attempts = 0
    max_attempts = max_combinations * 20

    while len(combos) < max_combinations and attempts < max_attempts:
        choice = tuple(rng.choice(values) for values in value_lists)
        if choice in seen:
            attempts += 1
            continue
        seen.add(choice)
        combos.append(dict(zip(keys, choice)))

    if len(combos) < max_combinations:
        for combo in itertools.product(*value_lists):
            if len(combos) >= max_combinations:
                break
            if combo in seen:
                continue
            combos.append(dict(zip(keys, combo)))

    return combos, total


def save_study_trials(study: optuna.study.Study, out_csv: str):
    rows = []
    for t in study.trials:
        rows.append({
            'number': t.number,
            'state': t.state.name if hasattr(t.state, 'name') else str(t.state),
            'value': t.value,
            'params': json.dumps(t.params),
            'intermediate_values': json.dumps({int(k): float(v) for k, v in t.intermediate_values.items()})
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def append_trial_metrics(csv_path: str, trial_number: int, params: dict, fold_metrics: list, objective_value: float):
    if not fold_metrics:
        return
    fp_fn_values = [m['fp_plus_fn'] for m in fold_metrics]
    mcc_values = [m['mcc'] for m in fold_metrics]
    brier_values = [m['brier'] for m in fold_metrics]

    row = {
        'trial_number': trial_number,
        'objective_value': objective_value,
        'mean_fp_plus_fn': float(np.mean(fp_fn_values)),
        'mean_mcc': float(np.nanmean(mcc_values)),
        'mean_brier': float(np.nanmean(brier_values)),
        'params': json.dumps(params)
    }

    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=header, index=False)

# ==================== DATA LOADING ====================
def load_dataset(data_dir, pattern="*.csv", exclude_2019=False):
    """Load and combine CSV files. Set ``exclude_2019`` to True to drop that year."""
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    print(f"\n{'='*60}")
    print(f"LOADING DATA")
    print(f"{'='*60}")
    print(f"Found {len(files)} CSV files")
    
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    print(f"Combined shape: {df.shape}")
    
    if exclude_2019 and 'year' in df.columns:
        before = len(df)
        df = df[df['year'] != 2019]
        print(f"Excluded 2019: {before - len(df)} samples removed")
    
    # Remove leaky features
    leak_cols = [c for c in df.columns if "event_day" in c.lower()]
    if leak_cols:
        print(f"Removing {len(leak_cols)} leaky features")
        df = df.drop(columns=leak_cols)
    
    print(f"\nClass distribution ({LABEL_COLUMN}):")
    print(df[LABEL_COLUMN].value_counts())
    
    return df

# ==================== PREPROCESSING ====================
def preprocess_data(df, holdout_fraction=HOLDOUT_FRACTION, static_feats=None, dynamic_feats=None):
    """Preprocess data with intelligent zero handling"""
    df = df.copy()
    
    # Select features
    available = [c for c in FORECASTING_FEATURE_SET 
                 if c != LABEL_COLUMN and c in df.columns]
    missing = sorted(set(FORECASTING_FEATURE_SET) - {LABEL_COLUMN} - set(available))
    
    if missing:
        print(f"\nWarning: {len(missing)} features missing")
    
    df = df[available + [LABEL_COLUMN]]
    
    # Handle infinities
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Clean label
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce')
    df[LABEL_COLUMN] = df[LABEL_COLUMN].round().clip(0, 1)
    df = df.dropna(subset=[LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]
    
    # Intelligent zero-to-NaN conversion
    print("\nApplying zero-to-NaN conversion for invalid zeros...")
    zero_replaced = 0
    for col in X.select_dtypes(include=np.number).columns:
        if col not in ZERO_IS_VALID:
            zero_mask = (X[col] == 0)
            count = zero_mask.sum()
            if count > 0:
                X.loc[zero_mask, col] = np.nan
                zero_replaced += count
    
    if zero_replaced > 0:
        print(f"Replaced {zero_replaced:,} invalid zeros with NaN")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=holdout_fraction, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train class balance: {y_train.value_counts().to_dict()}")
    
    if static_feats is None or dynamic_feats is None:
        static_feats, dynamic_feats = partition_features_by_variability(
            X_train,
            baseline_static=STATIC_BASELINE_FEATURES,
            force_static=STATIC_FORCE_FEATURES,
            force_dynamic=DYNAMIC_FORCE_FEATURES,
            static_percentile=STATIC_VARIANCE_PERCENTILE,
            dynamic_percentile=DYNAMIC_VARIANCE_PERCENTILE
        )
    else:
        static_feats = [feat for feat in static_feats if feat in X_train.columns]
        dynamic_feats = [feat for feat in dynamic_feats if feat in X_train.columns]
        missing = set(static_feats + dynamic_feats) - set(X_train.columns)
        if missing:
            print(f"Warning: {len(missing)} provided features missing from training data: {sorted(missing)}")

    print(f"Static features (auto + overrides): {len(static_feats)}")
    print(f"Dynamic features (auto + overrides): {len(dynamic_feats)}")

    if static_feats:
        print("Static feature list (with ordering preserved):")
        for idx, feat in enumerate(static_feats, start=1):
            print(f"  {idx:2d}. {feat}")
    if dynamic_feats:
        print("Dynamic feature list (with ordering preserved):")
        for idx, feat in enumerate(dynamic_feats, start=1):
            print(f"  {idx:2d}. {feat}")

    return X_train, X_test, y_train, y_test, static_feats, dynamic_feats

# ==================== PREPROCESSING HELPERS ====================
def preprocess_static(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute static features (slow-changing descriptors)."""
    return X.fillna(X.median())


def preprocess_dynamic(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute dynamic features (rapidly changing signals)."""
    return X.fillna(X.median())

# ==================== CLASS WEIGHT ====================
def compute_class_weight(y):
    """Compute scale_pos_weight for imbalanced data"""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / max(1, pos)

# ==================== MODEL TRAINING ====================
def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test,
                             feature_names=None, beta_values=FBETA_VALUES,
                             n_folds=N_FOLDS):
    """Train model and evaluate with repeated stratified CV."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")

    train_index = X_train.index.to_numpy()
    X_train_use = X_train.reset_index(drop=True)
    y_train_use = y_train.reset_index(drop=True)
    X_test_use = X_test.reset_index(drop=True)
    y_test_use = y_test.reset_index(drop=True)

    # Define scoring across requested F-beta values
    scoring = {
        'roc_auc': 'roc_auc',
        'recall': 'recall',
        'precision': 'precision',
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'mcc': make_scorer(matthews_corrcoef),
        'fp_fn_penalty': make_scorer(fp_fn_penalty_score)
    }
    for beta in beta_values:
        scoring[beta_to_key(beta)] = make_scorer(fbeta_score, beta=beta, zero_division=0)

    # Single CV (N_FOLDS folds)
    print(f"Running cross-validation ({n_folds} folds)...")
    score_history = {metric: [] for metric in scoring.keys()}

    cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    cv_results = cross_validate(
        model,
        X_train_use,
        y_train_use,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    for metric_name in score_history:
        score_history[metric_name].extend(cv_results[f'test_{metric_name}'])

    cv_summary = {
        metric: {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        }
        for metric, values in score_history.items()
    }

    # Print CV results
    print(f"\nCross-Validation Results ({n_folds} folds):")
    print(f"  ROC-AUC:   {cv_summary['roc_auc']['mean']:.4f} ± {cv_summary['roc_auc']['std']:.4f}")
    print(f"  Recall:    {cv_summary['recall']['mean']:.4f} ± {cv_summary['recall']['std']:.4f}")
    print(f"  Precision: {cv_summary['precision']['mean']:.4f} ± {cv_summary['precision']['std']:.4f}")
    print(f"  Balanced Acc: {cv_summary['balanced_accuracy']['mean']:.4f} ± {cv_summary['balanced_accuracy']['std']:.4f}")
    print(f"  MCC:       {cv_summary['mcc']['mean']:.4f} ± {cv_summary['mcc']['std']:.4f}")
    print(f"  -(FP+FN):  {cv_summary['fp_fn_penalty']['mean']:.4f} ± {cv_summary['fp_fn_penalty']['std']:.4f}")

    # Highlight select F-beta scores for readability (F1, F2, F3 if available)
    for beta in [1.0, 1.5, 2.0, 2.5, 3.0]:
        key = beta_to_key(beta)
        if key in cv_summary:
            print(f"  F{beta:.1f} Score: {cv_summary[key]['mean']:.4f} ± {cv_summary[key]['std']:.4f}")

    # Generate OOF probabilities and fit isotonic calibrator
    print("\nGenerating out-of-fold predictions and fitting isotonic calibrator...")
    final_model, calibrator, oof_raw, oof_calibrated = calibrate_model(model, X_train_use, y_train_use, n_folds)
    calibrated_model = IsotonicCalibratedModel(final_model, calibrator)

    # Test set evaluation with threshold sweep minimizing FP+FN
    test_proba_raw = final_model.predict_proba(X_test_use)[:, 1]
    test_proba = calibrator.predict(test_proba_raw)
    test_proba = np.clip(test_proba, PROBA_EPS, 1.0 - PROBA_EPS)

    best_metrics, threshold_metrics = sweep_thresholds(y_test_use, test_proba, beta_values, THRESHOLD_SEARCH_GRID)
    best_threshold = best_metrics['threshold']
    best_preds = (test_proba >= best_threshold).astype(int)

    print(f"\nTest Set Results (threshold={best_threshold:.2f} minimizing FP+FN):")
    print(f"  ROC-AUC:        {best_metrics['roc_auc']:.4f}")
    print(f"  Balanced Acc:   {best_metrics['balanced_accuracy']:.4f}")
    print(f"  MCC:            {best_metrics['mcc']:.4f}")
    print(f"  Total Error:    {best_metrics['fp_plus_fn']}")
    print(f"  Precision:      {best_metrics['precision']:.4f}  Recall: {best_metrics['recall']:.4f}")
    print(f"  Confusion Matrix (Test): TP={best_metrics['tp']}  FP={best_metrics['fp']}  TN={best_metrics['tn']}  FN={best_metrics['fn']}")
    print(f"  Rates: TPR={best_metrics['tpr']:.4f}  TNR={best_metrics['tnr']:.4f}  FPR={best_metrics['fpr']:.4f}  FNR={best_metrics['fnr']:.4f}")
    print(f"  Brier:          {best_metrics['brier']:.4f}  Log-loss: {best_metrics['log_loss']:.4f}")

    # Extract feature importance from raw (uncalibrated) model
    importances = extract_feature_importance(final_model, X_train_use, feature_names)

    result = {
        'model_name': name,
        'cv_summary': cv_summary,
        'model': calibrated_model,
        'raw_model': final_model,
        'importances': importances,
        'threshold_metrics': threshold_metrics,
        'best_threshold': best_threshold,
        'oof_proba_raw': oof_raw,
        'oof_proba_calibrated': oof_calibrated,
        'oof_indices': train_index.tolist(),
        'oof_targets': y_train_use.tolist(),
        'calibrator': calibrator,
        'test_indices': X_test.index.tolist(),
        'test_targets': y_test.tolist()
    }

    apply_test_metrics(result, {
        'metrics': best_metrics,
        'proba': test_proba,
        'preds': best_preds
    })
    result['test_proba_raw'] = test_proba_raw
    result['test_proba_calibrated'] = test_proba

    return result

# ==================== FEATURE IMPORTANCE ====================
def extract_feature_importance(pipeline, X_train, feature_names):
    """Extract feature importance from trained model"""
    try:
        model = pipeline
        
        # Extract importances based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
                # Get feature names after preprocessing
                preprocessor = pipeline.named_steps['preprocessor']
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except:
                    pass
            importances = model.get_feature_importance()
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create DataFrame
        imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return imp_df
    
    except Exception as e:
        print(f"  Warning: Could not extract feature importance: {e}")
        return None


# ==================== MAIN EXECUTION ====================
def run_stage_one(df, output_dir):
    print(f"\n{'='*60}")
    print("STAGE 1: 80/20 validation multi-cycle search")
    print(f"{'='*60}")

    X_train, X_valid, y_train, y_valid, static_feats, dynamic_feats = preprocess_data(
        df,
        holdout_fraction=STAGE1_VALIDATION_FRACTION
    )

    if not static_feats:
        raise ValueError("Stage 1: No static features available for CatBoost training.")
    if not dynamic_feats:
        raise ValueError("Stage 1: No dynamic features available for XGBoost training.")

    X_train_static = preprocess_static(X_train[static_feats])
    X_valid_static = preprocess_static(X_valid[static_feats])

    X_train_dynamic = preprocess_dynamic(X_train[dynamic_feats])
    X_valid_dynamic = preprocess_dynamic(X_valid[dynamic_feats])

    X_train_full = preprocess_dynamic(X_train)
    X_valid_full = preprocess_dynamic(X_valid)

    pos_weight = compute_class_weight(y_train)
    print(f"Class imbalance ratio (Stage 1 train): {pos_weight:.2f}")

    cat_common_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_state": RANDOM_STATE,
        "verbose": 0,
        "thread_count": -1,
    }

    fixed_cat_params = dict(BEST_CATBOOST_PARAMS)
    fixed_cat_params['iterations'] = 2050

    xgb_common_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    lgb_common_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "num_threads": -1,
        "random_state": RANDOM_STATE,
        "verbosity": -1
    }

    cycle_specs = [
        {
            'name': 'Cycle 1 (Cat:static | XGB:dynamic | LGB:full)',
            'cat_features': ('static', static_feats, X_train_static, X_valid_static),
            'xgb_features': ('dynamic', dynamic_feats, X_train_dynamic, X_valid_dynamic),
            'lgb_features': ('full', X_train.columns.tolist(), X_train_full, X_valid_full)
        },
        {
            'name': 'Cycle 2 (Cat:dynamic | XGB:full | LGB:static)',
            'cat_features': ('dynamic', dynamic_feats, X_train_dynamic, X_valid_dynamic),
            'xgb_features': ('full', X_train.columns.tolist(), X_train_full, X_valid_full),
            'lgb_features': ('static', static_feats, X_train_static, X_valid_static)
        },
        {
            'name': 'Cycle 3 (Cat:full | XGB:static | LGB:dynamic)',
            'cat_features': ('full', X_train.columns.tolist(), X_train_full, X_valid_full),
            'xgb_features': ('static', static_feats, X_train_static, X_valid_static),
            'lgb_features': ('dynamic', dynamic_feats, X_train_dynamic, X_valid_dynamic)
        }
    ]

    cycle_results = []
    weight_export_rows = []

    for cycle_idx, spec in enumerate(cycle_specs, start=1):
        print(f"\n{'-'*60}")
        print(f"Stage 1 {spec['name']}")
        print(f"{'-'*60}")

        cat_key, cat_feats, cat_train, cat_valid = spec['cat_features']
        xgb_key, xgb_feats, xgb_train, xgb_valid = spec['xgb_features']
        lgb_key, lgb_feats, lgb_train, lgb_valid = spec['lgb_features']

        print(f"CatBoost uses {cat_key} features ({len(cat_feats)} columns)")
        print(f"XGBoost uses {xgb_key} features ({len(xgb_feats)} columns)")
        print(f"LightGBM uses {lgb_key} features ({len(lgb_feats)} columns)")

        cat_model = CatBoostClassifier(**{**cat_common_params, **fixed_cat_params})
        cat_results = train_and_evaluate_model(
            f"CatBoost ({cat_key}) - stage1-cycle{cycle_idx}",
            cat_model,
            cat_train,
            y_train,
            cat_valid,
            y_valid,
            feature_names=cat_feats,
            beta_values=FBETA_VALUES,
            n_folds=STAGE1_FOLDS
        )
        cat_results['config_params'] = copy.deepcopy(fixed_cat_params)

        xgb_params = dict(BEST_XGBOOST_PARAMS)
        xgb_scale = xgb_params.pop('scale_pos_weight_factor', 1.0)
        xgb_model = XGBClassifier(**{**xgb_common_params, **xgb_params, 'scale_pos_weight': pos_weight * xgb_scale})
        xgb_results = train_and_evaluate_model(
            f"XGBoost ({xgb_key}) - stage1-cycle{cycle_idx}",
            xgb_model,
            xgb_train,
            y_train,
            xgb_valid,
            y_valid,
            feature_names=xgb_feats,
            beta_values=FBETA_VALUES,
            n_folds=STAGE1_FOLDS
        )
        xgb_params['scale_pos_weight_factor'] = xgb_scale
        xgb_results['config_params'] = xgb_params

        lgb_params = dict(BEST_LIGHTGBM_PARAMS)
        lgb_scale = lgb_params.pop('scale_pos_weight_factor', 1.0)
        lgb_model = LGBMClassifier(**{**lgb_common_params, **lgb_params, 'scale_pos_weight': pos_weight * lgb_scale})
        lgb_results = train_and_evaluate_model(
            f"LightGBM ({lgb_key}) - stage1-cycle{cycle_idx}",
            lgb_model,
            lgb_train,
            y_train,
            lgb_valid,
            y_valid,
            feature_names=lgb_feats,
            beta_values=FBETA_VALUES,
            n_folds=STAGE1_FOLDS
        )
        lgb_params['scale_pos_weight_factor'] = lgb_scale
        lgb_results['config_params'] = lgb_params

        proba_cat = cat_results['test_proba_calibrated']
        proba_xgb = xgb_results['test_proba_calibrated']
        proba_lgb = lgb_results['test_proba_calibrated']
        best_weights, best_threshold, best_metrics, weight_results, blended = find_best_ensemble_weights(
            proba_cat,
            proba_xgb,
            proba_lgb,
            y_valid.to_numpy()
        )

        print(f"\nEnsemble summary for {spec['name']}")
        print(f"  CatBoost weight: {best_weights[0]:.2f}")
        print(f"  XGBoost weight:  {best_weights[1]:.2f}")
        print(f"  LightGBM weight:{best_weights[2]:.2f}")
        print(f"  Threshold: {best_threshold:.2f}")
        print(f"  FP+FN: {best_metrics['fp_plus_fn']:.1f} | MCC: {best_metrics['mcc']:.3f} | Brier: {best_metrics['brier']:.3f}")

        for row in weight_results:
            export_row = dict(row)
            export_row['cycle'] = cycle_idx
            export_row['cycle_name'] = spec['name']
            weight_export_rows.append(export_row)

        cycle_results.append({
            'cycle_index': cycle_idx,
            'cycle_name': spec['name'],
            'cat_results': cat_results,
            'xgb_results': xgb_results,
            'lgb_results': lgb_results,
            'weights': best_weights,
            'threshold': best_threshold,
            'ensemble_metrics': best_metrics,
            'blended_proba': blended,
            'feature_roles': {
                'catboost': cat_key,
                'xgboost': xgb_key,
                'lightgbm': lgb_key
            },
            'feature_lists': {
                'static': static_feats,
                'dynamic': dynamic_feats,
                'full': X_train.columns.tolist()
            }
        })

    weight_df = pd.DataFrame(weight_export_rows)
    if not weight_df.empty:
        weight_df.sort_values(['cycle', 'fp_plus_fn'], inplace=True)
        weight_df.to_csv(
            os.path.join(output_dir, 'stage1_cycle_weight_sweep.csv'),
            index=False
        )

    best_cycle = max(
        cycle_results,
        key=lambda cr: (
            -cr['ensemble_metrics']['fp_plus_fn'],
            cr['ensemble_metrics']['mcc'],
            -cr['ensemble_metrics']['brier']
        )
    )

    best_cat_params = copy.deepcopy(best_cycle['cat_results']['config_params'])
    best_xgb_params = copy.deepcopy(best_cycle['xgb_results']['config_params'])
    best_lgb_params = copy.deepcopy(best_cycle['lgb_results']['config_params'])
    best_weights = tuple(best_cycle['weights'])
    best_threshold = float(best_cycle['threshold'])
    best_feature_roles = dict(best_cycle['feature_roles'])
    feature_lists = {
        'static': list(static_feats),
        'dynamic': list(dynamic_feats),
        'full': list(X_train.columns)
    }

    print(f"\n{'='*60}")
    print("Stage 1 best cycle selection")
    print(f"{'='*60}")
    print(f"Best cycle: {best_cycle['cycle_name']} (index {best_cycle['cycle_index']})")
    print(
        f"FP+FN={best_cycle['ensemble_metrics']['fp_plus_fn']:.1f} | "
        f"MCC={best_cycle['ensemble_metrics']['mcc']:.3f} | "
        f"Brier={best_cycle['ensemble_metrics']['brier']:.3f}"
    )

    stage1_summary_rows = []
    for cycle in cycle_results:
        metrics = cycle['ensemble_metrics']
        stage1_summary_rows.append({
            'cycle_index': cycle['cycle_index'],
            'cycle_name': cycle['cycle_name'],
            'fp_plus_fn': metrics['fp_plus_fn'],
            'mcc': metrics['mcc'],
            'brier': metrics['brier'],
            'roc_auc': metrics['roc_auc'],
            'pr_auc': metrics['pr_auc'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'threshold': metrics['threshold'],
            'weight_catboost': cycle['weights'][0],
            'weight_xgboost': cycle['weights'][1],
            'weight_lightgbm': cycle['weights'][2]
        })
    pd.DataFrame(stage1_summary_rows).to_csv(
        os.path.join(output_dir, 'stage1_cycle_summary.csv'),
        index=False
    )

    return {
        'static_feats': static_feats,
        'dynamic_feats': dynamic_feats,
        'feature_lists': feature_lists,
        'cycle_results': cycle_results,
        'best_cycle': best_cycle,
        'best_cat_params': best_cat_params,
        'best_xgb_params': best_xgb_params,
        'best_lgb_params': best_lgb_params,
        'best_weights': best_weights,
        'best_threshold': best_threshold,
        'best_feature_roles': best_feature_roles,
        'pos_weight': pos_weight
    }


def run_stage_two(df, output_dir, stage1_info):
    print(f"\n{'='*60}")
    print("STAGE 2: Final training with tiny holdout")
    print(f"{'='*60}")

    static_feats = stage1_info['static_feats']
    dynamic_feats = stage1_info['dynamic_feats']
    feature_lists = stage1_info['feature_lists']
    best_feature_roles = stage1_info['best_feature_roles']
    best_weights = tuple(stage1_info['best_weights'])
    best_threshold = float(stage1_info['best_threshold'])
    best_cat_params_stage1 = stage1_info['best_cat_params']
    best_xgb_params_stage1 = stage1_info['best_xgb_params']
    best_lgb_params_stage1 = stage1_info['best_lgb_params']
    X_train, X_holdout, y_train, y_holdout, _, _ = preprocess_data(
        df,
        holdout_fraction=HOLDOUT_FRACTION,
        static_feats=static_feats,
        dynamic_feats=dynamic_feats
    )

    X_train_static = preprocess_static(X_train[static_feats])
    X_holdout_static = preprocess_static(X_holdout[static_feats])

    X_train_dynamic = preprocess_dynamic(X_train[dynamic_feats])
    X_holdout_dynamic = preprocess_dynamic(X_holdout[dynamic_feats])

    X_train_full = preprocess_dynamic(X_train)
    X_holdout_full = preprocess_dynamic(X_holdout)

    train_frames = {
        'static': X_train_static,
        'dynamic': X_train_dynamic,
        'full': X_train_full
    }
    holdout_frames = {
        'static': X_holdout_static,
        'dynamic': X_holdout_dynamic,
        'full': X_holdout_full
    }
    feature_names_map = {
        role: frame.columns.tolist()
        for role, frame in train_frames.items()
    }

    pos_weight = compute_class_weight(y_train)
    print(f"Class imbalance ratio (Stage 2 train): {pos_weight:.2f}")

    cat_common_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_state": RANDOM_STATE,
        "verbose": 0,
        "thread_count": -1,
    }
    cat_role = best_feature_roles.get('catboost', 'static')
    cat_feature_names = feature_names_map[cat_role]
    best_cat_params = dict(best_cat_params_stage1)
    cat_model = CatBoostClassifier(**{**cat_common_params, **best_cat_params})
    cat_results = train_and_evaluate_model(
        f"CatBoost ({cat_role}) - final",
        cat_model,
        train_frames[cat_role],
        y_train,
        holdout_frames[cat_role],
        y_holdout,
        feature_names=cat_feature_names,
        beta_values=FBETA_VALUES,
        n_folds=N_FOLDS
    )
    cat_results['config_params'] = copy.deepcopy(best_cat_params)
    cat_results['feature_role'] = cat_role

    xgb_common_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    xgb_role = best_feature_roles.get('xgboost', 'dynamic')
    xgb_feature_names = feature_names_map[xgb_role]
    best_xgb_params = dict(best_xgb_params_stage1)
    xgb_scale = best_xgb_params.pop('scale_pos_weight_factor', 1.0)
    xgb_model = XGBClassifier(**{**xgb_common_params, **best_xgb_params, 'scale_pos_weight': pos_weight * xgb_scale})
    xgb_results = train_and_evaluate_model(
        f"XGBoost ({xgb_role}) - final",
        xgb_model,
        train_frames[xgb_role],
        y_train,
        holdout_frames[xgb_role],
        y_holdout,
        feature_names=xgb_feature_names,
        beta_values=FBETA_VALUES,
        n_folds=N_FOLDS
    )
    best_xgb_params['scale_pos_weight_factor'] = xgb_scale
    xgb_results['config_params'] = copy.deepcopy(best_xgb_params)
    xgb_results['feature_role'] = xgb_role

    lgb_common_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "num_threads": -1,
        "random_state": RANDOM_STATE,
        "verbosity": -1
    }
    lgb_role = best_feature_roles.get('lightgbm', 'full')
    lgb_feature_names = feature_names_map[lgb_role]
    best_lgb_params = dict(best_lgb_params_stage1)
    lgb_scale = best_lgb_params.pop('scale_pos_weight_factor', 1.0)
    lgb_model = LGBMClassifier(**{**lgb_common_params, **best_lgb_params, 'scale_pos_weight': pos_weight * lgb_scale})
    lgb_results = train_and_evaluate_model(
        f"LightGBM ({lgb_role}) - final",
        lgb_model,
        train_frames[lgb_role],
        y_train,
        holdout_frames[lgb_role],
        y_holdout,
        feature_names=lgb_feature_names,
        beta_values=FBETA_VALUES,
        n_folds=N_FOLDS
    )
    best_lgb_params['scale_pos_weight_factor'] = lgb_scale
    lgb_results['config_params'] = copy.deepcopy(best_lgb_params)
    lgb_results['feature_role'] = lgb_role

    cat_proba = cat_results['test_proba_calibrated']
    xgb_proba = xgb_results['test_proba_calibrated']
    lgb_proba = lgb_results['test_proba_calibrated']
    ensemble_metrics, blended = evaluate_fixed_ensemble(
        cat_proba,
        xgb_proba,
        lgb_proba,
        y_holdout.to_numpy(),
        best_weights,
        best_threshold
    )

    print(f"\n{'-'*60}")
    print("Stage 2 ensemble (tiny holdout)")
    print(f"Static weight (CatBoost): {ensemble_metrics['weight_static']:.2f}")
    print(f"Dynamic weight (XGBoost): {ensemble_metrics['weight_dynamic']:.2f}")
    print(f"LightGBM weight:          {ensemble_metrics['weight_lightgbm']:.2f}")
    print(f"Threshold (fixed): {best_threshold:.2f}")
    print(
        f"F{FBETA:.1f}: {ensemble_metrics['fbeta_default']:.4f}  |  "
        f"ROC-AUC: {ensemble_metrics['roc_auc']:.4f}  |  PR-AUC: {ensemble_metrics['pr_auc']:.4f}"
    )
    print(
        f"Precision: {ensemble_metrics['precision']:.4f}  |  Recall: {ensemble_metrics['recall']:.4f}  |  MCC: {ensemble_metrics['mcc']:.4f}"
    )
    print(
        f"Brier: {ensemble_metrics['brier']:.4f}  |  Log-loss: {ensemble_metrics['log_loss']:.4f}"
    )
    print(
        f"Confusion Matrix: TP={ensemble_metrics['tp']}  FP={ensemble_metrics['fp']}  "
        f"TN={ensemble_metrics['tn']}  FN={ensemble_metrics['fn']}"
    )

    summary_rows = [
        {
            'stage': 'stage2',
            'model': cat_results['model_name'],
            'feature_role': cat_role,
            'config_params': json.dumps(best_cat_params),
            'selected_threshold': cat_results['best_threshold'],
            **cat_results['test_metrics']
        },
        {
            'stage': 'stage2',
            'model': xgb_results['model_name'],
            'feature_role': xgb_role,
            'config_params': json.dumps(best_xgb_params),
            'selected_threshold': xgb_results['best_threshold'],
            **xgb_results['test_metrics']
        },
        {
            'stage': 'stage2',
            'model': lgb_results['model_name'],
            'feature_role': lgb_role,
            'config_params': json.dumps(best_lgb_params),
            'selected_threshold': lgb_results['best_threshold'],
            **lgb_results['test_metrics']
        },
        {
            'stage': 'stage2',
            'model': 'Ensemble',
            'feature_roles': json.dumps(best_feature_roles),
            'selected_threshold': best_threshold,
            **ensemble_metrics
        }
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'stage2_model_summary.csv'), index=False)

    ensemble_model = WeightedEnsemble(
        models={
            "catboost": cat_results['model'],
            "xgboost": xgb_results['model'],
            "lightgbm": lgb_results['model']
        },
        weights=best_weights,
        threshold=best_threshold,
        metadata={
            "objective": "minimize FP+FN, maximize MCC",
            "stage": "stage2",
            "fbeta": FBETA,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": dict(ensemble_metrics),
            "feature_roles": dict(best_feature_roles)
        },
        feature_sets={
            "static": static_feats,
            "dynamic": dynamic_feats,
            "full": X_train_full.columns.tolist()
        },
        model_feature_roles=best_feature_roles
    )
    ensemble_path = os.path.join(output_dir, "final_ensemble.pkl")
    joblib.dump(ensemble_model, ensemble_path)
    print(f"\n✅ Saved final ensemble model to {ensemble_path}")

    return {
        'cat_results': cat_results,
        'xgb_results': xgb_results,
        'lgb_results': lgb_results,
        'ensemble_metrics': ensemble_metrics,
        'blended_proba': blended,
        'weights': best_weights,
        'threshold': best_threshold,
        'holdout_targets': y_holdout.to_numpy(),
        'model_feature_roles': best_feature_roles
    }


def main(data_dir, output_dir="./ensemble_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("STATIC + DYNAMIC ENSEMBLE")
    print(f"{'='*60}")

    df = load_dataset(data_dir)

    stage1_info = run_stage_one(df, output_dir)
    stage2_info = run_stage_two(
        df,
        output_dir,
        stage1_info
    )

    # Persist stage 1 summary
    stage1_summary_rows = [
        {
            'stage': 'stage1',
            'model': stage1_info['cat_results']['model_name'],
            'config_params': json.dumps(stage1_info['best_cat_params']),
            'selected_threshold': stage1_info['cat_results']['best_threshold'],
            **stage1_info['cat_results']['test_metrics']
        },
        {
            'stage': 'stage1',
            'model': stage1_info['xgb_results']['model_name'],
            'config_params': json.dumps(stage1_info['best_xgb_params']),
            'selected_threshold': stage1_info['xgb_results']['best_threshold'],
            **stage1_info['xgb_results']['test_metrics']
        },
        {
            'stage': 'stage1',
            'model': stage1_info['lgb_results']['model_name'],
            'config_params': json.dumps(stage1_info['best_lgb_params']),
            'selected_threshold': stage1_info['lgb_results']['best_threshold'],
            **stage1_info['lgb_results']['test_metrics']
        },
        {
            'stage': 'stage1',
            'model': 'Ensemble',
            'selected_threshold': stage1_info['best_threshold'],
            **stage1_info['ensemble_metrics']
        }
    ]
    pd.DataFrame(stage1_summary_rows).to_csv(
        os.path.join(output_dir, 'stage1_model_summary.csv'),
        index=False
    )

    results_summary = {
        'stage1': stage1_info,
        'stage2': stage2_info
    }

    return results_summary


def _legacy_main(data_dir, output_dir="./ensemble_outputs"):
    """Train static (CatBoost) and dynamic (XGBoost) models and blend them."""

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("STATIC + DYNAMIC ENSEMBLE")
    print(f"{'='*60}")

    # Load and split data
    df = load_dataset(data_dir)
    X_train, X_test, y_train, y_test, static_feats, dynamic_feats = preprocess_data(df)

    if not static_feats:
        raise ValueError("No static features available for CatBoost training. Verify STATIC_FEATURES list.")
    if not dynamic_feats:
        raise ValueError("No dynamic features available for XGBoost training. Verify feature lists.")

    X_train_static = preprocess_static(X_train[static_feats])
    X_test_static = preprocess_static(X_test[static_feats])

    X_train_dynamic = preprocess_dynamic(X_train[dynamic_feats])
    X_test_dynamic = preprocess_dynamic(X_test[dynamic_feats])

    X_train_full = preprocess_dynamic(X_train)
    X_test_full = preprocess_dynamic(X_test)

    pos_weight = compute_class_weight(y_train)
    print(f"\nClass imbalance ratio: {pos_weight:.2f}")

    oof_store = {
        'catboost': [],
        'xgboost': [],
        'lightgbm': []
    }

    # Train static model (CatBoost) across search space
    cat_common_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_state": RANDOM_STATE,
        "verbose": 0,
        "thread_count": -1,
    }
    cat_threshold_rows = []

    optuna.logging.set_verbosity(optuna.logging.INFO)
    def cat_objective(trial: optuna.trial.Trial):
        params = {
            'iterations': trial.suggest_int('iterations', 1850, 2200),
            'depth': trial.suggest_int('depth', 9, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.04),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.8, 1.6),
            'subsample': trial.suggest_float('subsample', 0.78, 0.85),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.88, 0.95),
            'random_strength': trial.suggest_float('random_strength', 0.5, 0.8),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.95, 1.2),
            'border_count': trial.suggest_categorical('border_count', [384, 512]),
        }
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_metrics = []
        for tr_idx, va_idx in cv.split(X_train_static, y_train):
            model = CatBoostClassifier(**{**cat_common_params, **params})
            model.fit(X_train_static.iloc[tr_idx], y_train.iloc[tr_idx])
            val_proba = model.predict_proba(X_train_static.iloc[va_idx])[:, 1]
            best_metrics, _ = sweep_thresholds(y_train.iloc[va_idx].to_numpy(), val_proba, FBETA_VALUES, THRESHOLD_SEARCH_GRID)
            fold_metrics.append(best_metrics)
        objective_value = float(-np.mean([m['fp_plus_fn'] for m in fold_metrics]))
        mean_fpfn = float(np.mean([m['fp_plus_fn'] for m in fold_metrics]))
        mean_mcc = float(np.mean([m['mcc'] for m in fold_metrics]))
        mean_brier = float(np.mean([m['brier'] for m in fold_metrics]))
        print(
            f"[CatBoost][Trial {trial.number:03d}] FP+FN={mean_fpfn:.1f}, "
            f"MCC={mean_mcc:.3f}, Brier={mean_brier:.3f}, Objective={objective_value:.3f}"
        )
        append_trial_metrics(
            os.path.join(output_dir, 'catboost_trial_metrics.csv'),
            trial.number,
            params,
            fold_metrics,
            objective_value
        )
        return objective_value

    cat_study = optuna.create_study(direction="maximize", study_name="catboost_study")
    cat_study.enqueue_trial({
        'iterations': 2000,
        'depth': 10,
        'learning_rate': 0.034497224452741705,
        'l2_leaf_reg': 1.1793622591307933,
        'subsample': 0.8118543026413315,
        'colsample_bylevel': 0.9195639780831663,
        'random_strength': 0.6616953985666554,
        'bagging_temperature': 1.088585694737754,
        'border_count': 512
    })
    cat_study.optimize(cat_objective, n_trials=N_TRIALS)
    best_cat_params = cat_study.best_trial.params
    print(f"\nOptuna-selected CatBoost params: {best_cat_params}")
    save_study_trials(cat_study, os.path.join(output_dir, 'catboost_optuna_trials.csv'))

    cat_model = CatBoostClassifier(**{**cat_common_params, **best_cat_params})
    cat_results = train_and_evaluate_model(
        f"CatBoost (Static) - optuna",
        cat_model,
        X_train_static,
        y_train,
        X_test_static,
        y_test,
        feature_names=static_feats,
        beta_values=FBETA_VALUES
    )
    cat_results['config_index'] = 0
    cat_results['config_params'] = best_cat_params
    for threshold_metrics in cat_results['threshold_metrics']:
        threshold_row = {**threshold_metrics}
        threshold_row['config_index'] = 0
        threshold_row['model'] = 'CatBoost'
        cat_threshold_rows.append(threshold_row)

    oof_store['catboost'] = [{
        'config_index': cat_results['config_index'],
        'indices': cat_results['oof_indices'],
        'targets': cat_results['oof_targets'],
        'oof_raw': cat_results['oof_proba_raw'],
        'oof_calibrated': cat_results['oof_proba_calibrated']
    }]

    # Train dynamic model (XGBoost) across search space
    xgb_common_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    xgb_threshold_rows = []

    def xgb_objective(trial: optuna.trial.Trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 800, 900, step=10),
            'max_depth': trial.suggest_int('max_depth', 7, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.025),
            'subsample': trial.suggest_float('subsample', 0.88, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.82, 0.90),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.8, 2.5),
            'gamma': trial.suggest_float('gamma', 0.0, 0.4),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.1),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 1.4),
            'scale_pos_weight_factor': trial.suggest_float('scale_pos_weight_factor', 1.05, 1.25)
        }
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_metrics = []
        for tr_idx, va_idx in cv.split(X_train_dynamic, y_train):
            spw = pos_weight * params['scale_pos_weight_factor']
            full_params = {**xgb_common_params, **{k: v for k, v in params.items() if k != 'scale_pos_weight_factor'}, 'scale_pos_weight': spw}
            model = XGBClassifier(**full_params)
            model.fit(X_train_dynamic.iloc[tr_idx], y_train.iloc[tr_idx])
            val_proba = model.predict_proba(X_train_dynamic.iloc[va_idx])[:, 1]
            best_metrics, _ = sweep_thresholds(y_train.iloc[va_idx].to_numpy(), val_proba, FBETA_VALUES, THRESHOLD_SEARCH_GRID)
            fold_metrics.append(best_metrics)
        objective_value = float(-np.mean([m['fp_plus_fn'] for m in fold_metrics]))
        mean_fpfn = float(np.mean([m['fp_plus_fn'] for m in fold_metrics]))
        mean_mcc = float(np.mean([m['mcc'] for m in fold_metrics]))
        mean_brier = float(np.mean([m['brier'] for m in fold_metrics]))
        print(
            f"[XGBoost][Trial {trial.number:03d}] FP+FN={mean_fpfn:.1f}, "
            f"MCC={mean_mcc:.3f}, Brier={mean_brier:.3f}, Objective={objective_value:.3f}"
        )
        append_trial_metrics(
            os.path.join(output_dir, 'xgboost_trial_metrics.csv'),
            trial.number,
            params,
            fold_metrics,
            objective_value
        )
        return objective_value

    xgb_study = optuna.create_study(direction="maximize", study_name="xgboost_study")
    xgb_study.enqueue_trial({
        'n_estimators': 850,
        'max_depth': 8,
        'learning_rate': 0.01737332862455038,
        'subsample': 0.9030614120011207,
        'colsample_bytree': 0.8539443128274192,
        'min_child_weight': 2.082400817315821,
        'gamma': 0.21544798764830475,
        'reg_alpha': 0.0444368331223305,
        'reg_lambda': 1.1980211816720427,
        'scale_pos_weight_factor': 1.1470689789882924
    })
    xgb_study.optimize(xgb_objective, n_trials=N_TRIALS)
    best_xgb_params = xgb_study.best_trial.params
    print(f"\nOptuna-selected XGBoost params: {best_xgb_params}")
    save_study_trials(xgb_study, os.path.join(output_dir, 'xgboost_optuna_trials.csv'))

    spw = pos_weight * best_xgb_params.pop('scale_pos_weight_factor', 1.0)
    xgb_model = XGBClassifier(**{**xgb_common_params, **best_xgb_params, 'scale_pos_weight': spw})
    xgb_results = train_and_evaluate_model(
            f"XGBoost (Dynamic) - optuna",
            xgb_model,
            X_train_dynamic,
            y_train,
            X_test_dynamic,
            y_test,
            feature_names=dynamic_feats,
            beta_values=FBETA_VALUES
    )
    xgb_results['config_index'] = 0
    xgb_results['config_params'] = best_xgb_params
    for threshold_metrics in xgb_results['threshold_metrics']:
        threshold_row = {**threshold_metrics}
        threshold_row['config_index'] = 0
        threshold_row['model'] = 'XGBoost'
        xgb_threshold_rows.append(threshold_row)

    oof_store['xgboost'] = [{
        'config_index': xgb_results['config_index'],
        'indices': xgb_results['oof_indices'],
        'targets': xgb_results['oof_targets'],
        'oof_raw': xgb_results['oof_proba_raw'],
        'oof_calibrated': xgb_results['oof_proba_calibrated']
    }]

    # Train LightGBM on full feature set
    lgb_common_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "num_threads": -1,
        "random_state": RANDOM_STATE,
        "verbosity": -1
    }
    lgb_threshold_rows = []

    def lgb_objective(trial: optuna.trial.Trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1250, 1450, step=25),
            'num_leaves': trial.suggest_categorical('num_leaves', [80, 95, 110]),
            'learning_rate': trial.suggest_float('learning_rate', 0.06, 0.09),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 8, 10]),
            'subsample': trial.suggest_float('subsample', 0.90, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.85, 0.92),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 30, step=5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.05),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.7, 1.1),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),
            'scale_pos_weight_factor': trial.suggest_float('scale_pos_weight_factor', 1.05, 1.25)
        }
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_metrics = []
        for tr_idx, va_idx in cv.split(X_train_full, y_train):
            spw = pos_weight * params['scale_pos_weight_factor']
            full_params = {**lgb_common_params, **{k: v for k, v in params.items() if k != 'scale_pos_weight_factor'}, 'scale_pos_weight': spw}
            model = LGBMClassifier(**full_params)
            model.fit(X_train_full.iloc[tr_idx], y_train.iloc[tr_idx])
            val_proba = model.predict_proba(X_train_full.iloc[va_idx])[:, 1]
            best_metrics, _ = sweep_thresholds(y_train.iloc[va_idx].to_numpy(), val_proba, FBETA_VALUES, THRESHOLD_SEARCH_GRID)
            fold_metrics.append(best_metrics)
        objective_value = float(-np.mean([m['fp_plus_fn'] for m in fold_metrics]))
        mean_fpfn = float(np.mean([m['fp_plus_fn'] for m in fold_metrics]))
        mean_mcc = float(np.mean([m['mcc'] for m in fold_metrics]))
        mean_brier = float(np.mean([m['brier'] for m in fold_metrics]))
        print(
            f"[LightGBM][Trial {trial.number:03d}] FP+FN={mean_fpfn:.1f}, "
            f"MCC={mean_mcc:.3f}, Brier={mean_brier:.3f}, Objective={objective_value:.3f}"
        )
        append_trial_metrics(
            os.path.join(output_dir, 'lightgbm_trial_metrics.csv'),
            trial.number,
            params,
            fold_metrics,
            objective_value
        )
        return objective_value

    lgb_study = optuna.create_study(direction="maximize", study_name="lightgbm_study")
    lgb_study.enqueue_trial({
        'n_estimators': 1350,
        'num_leaves': 95,
        'learning_rate': 0.07879855138490763,
        'max_depth': -1,
        'subsample': 0.929574246178696,
        'colsample_bytree': 0.8812337309009289,
        'min_child_samples': 20,
        'reg_alpha': 0.005437524149790202,
        'reg_lambda': 0.87445285534292,
        'min_split_gain': 0.010585021824577587,
        'scale_pos_weight_factor': 1.1575139862587251
    })
    lgb_study.optimize(lgb_objective, n_trials=N_TRIALS)
    best_lgb_params = lgb_study.best_trial.params
    print(f"\nOptuna-selected LightGBM params: {best_lgb_params}")
    save_study_trials(lgb_study, os.path.join(output_dir, 'lightgbm_optuna_trials.csv'))

    spw_lgb = pos_weight * best_lgb_params.pop('scale_pos_weight_factor', 1.0)
    lgb_model = LGBMClassifier(**{**lgb_common_params, **best_lgb_params, 'scale_pos_weight': spw_lgb})
    lgb_results = train_and_evaluate_model(
            f"LightGBM (Full) - optuna",
            lgb_model,
            X_train_full,
            y_train,
            X_test_full,
            y_test,
            feature_names=X_train_full.columns.tolist(),
            beta_values=FBETA_VALUES
    )
    lgb_results['config_index'] = 0
    lgb_results['config_params'] = best_lgb_params
    for threshold_metrics in lgb_results['threshold_metrics']:
        threshold_row = {**threshold_metrics}
        threshold_row['config_index'] = 0
        threshold_row['model'] = 'LightGBM'
        lgb_threshold_rows.append(threshold_row)

    oof_store['lightgbm'] = [{
        'config_index': lgb_results['config_index'],
        'indices': lgb_results['oof_indices'],
        'targets': lgb_results['oof_targets'],
        'oof_raw': lgb_results['oof_proba_raw'],
        'oof_calibrated': lgb_results['oof_proba_calibrated']
    }]

    # Persist threshold sweep summaries and Optuna trials
    if cat_threshold_rows:
        pd.DataFrame(cat_threshold_rows).to_csv(
            os.path.join(output_dir, 'catboost_threshold_sweep.csv'),
            index=False
        )
    if xgb_threshold_rows:
        pd.DataFrame(xgb_threshold_rows).to_csv(
            os.path.join(output_dir, 'xgboost_threshold_sweep.csv'),
            index=False
        )
    if lgb_threshold_rows:
        pd.DataFrame(lgb_threshold_rows).to_csv(
            os.path.join(output_dir, 'lightgbm_threshold_sweep.csv'),
            index=False
        )

    # Blend predictions via weight search (CatBoost + XGBoost + LightGBM)
    cat_proba = cat_results['test_proba_calibrated']
    xgb_proba = xgb_results['test_proba_calibrated']
    lgb_proba = lgb_results['test_proba_calibrated']

    best_metrics = None
    best_threshold = ENSEMBLE_THRESHOLD
    best_blended = None
    best_weights = (1/3, 1/3, 1/3)
    weight_results = []

    for w_cat in WEIGHT_SEARCH_GRID:
        for w_xgb in WEIGHT_SEARCH_GRID:
            w_lgb = 1.0 - w_cat - w_xgb
            if w_lgb < 0:
                continue
            if min(w_cat, w_xgb, w_lgb) < MIN_MODEL_WEIGHT:
                continue

            blended = w_cat * cat_proba + w_xgb * xgb_proba + w_lgb * lgb_proba
            metrics_best, metrics_all = sweep_thresholds(y_test.to_numpy(), blended, FBETA_VALUES, ENSEMBLE_THRESHOLD_GRID)

            for metrics in metrics_all:
                metrics_with_weights = {
                    **metrics,
                    'weight_static': w_cat,
                    'weight_dynamic': w_xgb,
                    'weight_lightgbm': w_lgb
                }
                weight_results.append(metrics_with_weights)

                score = selection_objective(metrics_with_weights)
                if score > selection_objective(best_metrics):
                    best_metrics = metrics_with_weights
                    best_threshold = metrics_with_weights['threshold']
                    best_blended = blended
                    best_weights = (w_cat, w_xgb, w_lgb)

    if best_metrics is None:
        raise RuntimeError("Weight sweep failed to produce metrics.")

    ensemble_preds = (best_blended >= best_threshold).astype(int)
    ensemble_metrics = {**best_metrics}
    ensemble_metrics['weight_static'] = best_weights[0]
    ensemble_metrics['weight_dynamic'] = best_weights[1]
    ensemble_metrics['weight_lightgbm'] = best_weights[2]
    ensemble_metrics['fbeta_default'] = ensemble_metrics.get(beta_to_key(FBETA))

    print(f"\n{'='*60}")
    print("ENSEMBLE SUMMARY")
    print(f"{'='*60}")
    print(f"Static weight (CatBoost): {ensemble_metrics['weight_static']:.2f}")
    print(f"Dynamic weight (XGBoost): {ensemble_metrics['weight_dynamic']:.2f}")
    print(f"LightGBM weight:          {ensemble_metrics['weight_lightgbm']:.2f}")
    print(f"Threshold (min FP+FN): {best_threshold:.2f}")
    print(f"F{FBETA:.1f}: {ensemble_metrics['fbeta_default']:.4f}  |  ROC-AUC: {ensemble_metrics['roc_auc']:.4f}  |  PR-AUC: {ensemble_metrics['pr_auc']:.4f}")
    print(f"Precision: {ensemble_metrics['precision']:.4f}  |  Recall: {ensemble_metrics['recall']:.4f}  |  MCC: {ensemble_metrics['mcc']:.4f}")
    print(f"Brier: {ensemble_metrics['brier']:.4f}  |  Log-loss: {ensemble_metrics['log_loss']:.4f}")
    print(f"Confusion Matrix: TP={ensemble_metrics['tp']}  FP={ensemble_metrics['fp']}  TN={ensemble_metrics['tn']}  FN={ensemble_metrics['fn']}")

    # Collate metrics for export
    summary_rows = [
        {
            'model': cat_results['model_name'],
            'config_index': cat_results['config_index'],
            'config_params': json.dumps(cat_results['config_params']),
            'selected_threshold': cat_results['best_threshold'],
            **cat_results['test_metrics']
        },
        {
            'model': xgb_results['model_name'],
            'config_index': xgb_results['config_index'],
            'config_params': json.dumps(xgb_results['config_params']),
            'selected_threshold': xgb_results['best_threshold'],
            **xgb_results['test_metrics']
        },
        {
            'model': lgb_results['model_name'],
            'config_index': lgb_results['config_index'],
            'config_params': json.dumps(lgb_results['config_params']),
            'selected_threshold': lgb_results['best_threshold'],
            **lgb_results['test_metrics']
        },
        {
            'model': 'Ensemble',
            'selected_threshold': best_threshold,
            **ensemble_metrics
        }
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)

    weight_df = pd.DataFrame(weight_results)
    weight_df['total_error'] = weight_df['fp_plus_fn']
    weight_df.sort_values('total_error', inplace=True)
    weight_df.to_csv(
        os.path.join(output_dir, 'ensemble_weight_sweep.csv'),
        index=False
    )

    top10 = weight_df.head(10)
    print(f"\nTop 10 ensemble combinations minimizing FP+FN:")
    for idx, row in top10.iterrows():
        print(
            f"  #{idx}: w_cat={row['weight_static']:.2f}, w_xgb={row['weight_dynamic']:.2f}, w_lgb={row['weight_lightgbm']:.2f}, "
            f"thr={row['threshold']:.2f}, FP={row['fp']}, FN={row['fn']}, Total={row['total_error']}, "
            f"MCC={row['mcc']:.4f}, F2={row.get(beta_to_key(2.0), np.nan):.4f}, "
            f"Precision={row['precision']:.4f}, Recall={row['recall']:.4f}, ROC-AUC={row['roc_auc']:.4f}, Brier={row['brier']:.4f}"
        )

    # Export calibrated OOF predictions for downstream analysis
    oof_rows = []
    for model_key, trials in oof_store.items():
        for trial in trials:
            for idx, proba, target in zip(trial['indices'], trial['oof_calibrated'], trial['targets']):
                oof_rows.append({
                    'model': model_key,
                    'config_index': trial['config_index'],
                    'sample_index': idx,
                    'calibrated_proba': proba,
                    'target': target
                })
    if oof_rows:
        pd.DataFrame(oof_rows).to_csv(
            os.path.join(output_dir, 'oof_calibrated_predictions.csv'),
            index=False
        )

    # Optional feature importance dumps
    for result in (cat_results, xgb_results, lgb_results):
        if result['importances'] is not None:
            print(f"\nFull feature ranking for {result['model_name']}:")
            for rank, (_, row) in enumerate(result['importances'].iterrows(), start=1):
                print(f"  {rank:3d}. {row['feature']:35s} {row['importance']:.6f}")
            result['importances'].to_csv(
                os.path.join(output_dir, f"{result['model_name'].replace(' ', '_')}_importance.csv"),
                index=False
            )

    ensemble_metadata = {
        "objective": "minimize FP+FN, maximize MCC",
        "fbeta": FBETA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": dict(ensemble_metrics)
    }
    feature_sets = {
        "static": static_feats,
        "dynamic": dynamic_feats,
        "full": X_train_full.columns.tolist()
    }
    ensemble_model = WeightedEnsemble(
        models={
            "catboost": cat_results['model'],
            "xgboost": xgb_results['model'],
            "lightgbm": lgb_results['model']
        },
        weights=best_weights,
        threshold=best_threshold,
        metadata=ensemble_metadata,
        feature_sets=feature_sets
    )
    ensemble_path = os.path.join(output_dir, "final_ensemble.pkl")
    joblib.dump(ensemble_model, ensemble_path)
    print(f"\n✅ Saved final ensemble model to {ensemble_path}")

    results_summary = {
        'catboost': cat_results,
        'xgboost': xgb_results,
        'lightgbm': lgb_results,
        'ensemble': ensemble_metrics
    }

    return results_summary

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <data_directory>")
        print("Example: python script.py /path/to/enriched_csvs")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    results = main(data_dir)