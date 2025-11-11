# -----------------------------------------------------------------------------
# GeoRiskAI - Final Analysis Engine
#
# CRITICAL OVERHAUL (Final Investor Mandate): This module is now a lean and
# robust component of the production pipeline.
# 1. SAMPLING: Robust error handling for GEE sampling.
# 2. ZONATION: Uses HDBSCAN for intelligent, density-based spatial clustering.
# 3. SUMMARIZATION: Extracts global SHAP feature importance for reporting.
# -----------------------------------------------------------------------------

import logging
import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans # <<< FIX: Added missing import
import ee
import time
import io
import json
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth import default

def get_drive_service():
    """Builds and returns a Google Drive service object."""
    creds, _ = default()
    return build('drive', 'v3', credentials=creds)

def sample_to_dataframe(image, roi, num_pixels, random_state=42):
    """
    Samples a GEE image to a pandas DataFrame using a robust, asynchronous export to Google Drive.
    """
    scale = 30 # Use a fixed scale to avoid issues with coarse projections
    logging.info(f"Starting asynchronous export for {num_pixels} pixels at resolution ({scale:.2f} m).")

    try:
        sample = image.sample(
            region=roi,
            scale=scale,
            numPixels=num_pixels,
            seed=random_state,
            geometries=True,
            dropNulls=True
        )

        task_description = f'GeoRiskAI_Sample_Export_{int(time.time())}'
        task = ee.batch.Export.table.toDrive(
            collection=sample,
            description=task_description,
            folder='GeoRiskAI_Exports',
            fileNamePrefix=task_description,
            fileFormat='CSV'
        )
        task.start()

        logging.info(f"GEE Export task started: {task_description} (id: {task.id}). Monitoring for completion...")
        while task.active():
            time.sleep(10)
            status = task.status()
            logging.info(f"Task status: {status['state']}...")

        status = task.status()
        if status['state'] != 'COMPLETED':
            logging.error(f"GEE export task failed. Final state: {status['state']}. Error: {status.get('error_message', 'No error message.')}")
            return None, {"requested": num_pixels, "returned": 0, "coverage_ratio": 0.0, "sample_scale_m": scale}

        logging.info("GEE export task completed. Retrieving data from Google Drive...")
        
        service = get_drive_service()
        file_name = f"{task_description}.csv"
        
        response = service.files().list(
            q=f"name='{file_name}' and mimeType='text/csv'",
            spaces='drive',
            fields='files(id, name)').execute()
        
        files = response.get('files', [])
        if not files:
            logging.error(f"Could not find exported file '{file_name}' in Google Drive.")
            return None, {"requested": num_pixels, "returned": 0, "coverage_ratio": 0.0, "sample_scale_m": scale}
        
        file_id = files[0].get('id')
        request = service.files().get_media(fileId=file_id)
        
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logging.info(f"Download {int(status.progress() * 100)}%.")

        fh.seek(0)
        df = pd.read_csv(fh)
        
        df['.geo'] = df['.geo'].apply(lambda x: json.loads(x)['coordinates'])
        df['longitude'] = df['.geo'].apply(lambda x: x[0])
        df['latitude'] = df['.geo'].apply(lambda x: x[1])
        df = df.drop(columns=['.geo', 'system:index'])

        logging.info(f"Successfully downloaded and parsed {len(df)} pixels.")
        coverage = {
            "requested": int(num_pixels),
            "returned": int(len(df)),
            "coverage_ratio": float(len(df) / max(1, int(num_pixels))),
            "sample_scale_m": float(scale),
        }
        return df, coverage

    except Exception as e:
        logging.error(f"Error during data sampling or download: {e}", exc_info=True)
        return None, {"requested": int(num_pixels), "returned": 0, "coverage_ratio": 0.0, "sample_scale_m": float(scale)}


def clean_and_prepare_data(df):
    """Cleans and prepares the sampled data."""
    logging.info("Cleaning and preparing sampled data...")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    land_cover_mapping = {
        10: 'Trees', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland',
        50: 'Built-up', 60: 'Bare/Sparse vegetation', 70: 'Snow/Ice',
        80: 'Water', 90: 'Wetland', 95: 'Mangroves', 100: 'Moss/Lichen'
    }
    if 'Land_Cover' in df.columns:
        df['Land_Cover_Type'] = df['Land_Cover'].map(land_cover_mapping).fillna('Unknown')
    else:
        df['Land_Cover_Type'] = 'Unknown'
        
    logging.info("Data preparation complete.")
    return df

def tune_hdbscan_params(X, min_cluster_range=(10, 200), min_samples_factor=(1, 5)):
    """
    Quick heuristic tuner for HDBSCAN parameters.
    """
    import hdbscan
    best_params, best_score = None, -1
    for mcs in range(min_cluster_range[0], min_cluster_range[1]+1, 10):
        for f in range(min_samples_factor[0], min_samples_factor[1]+1):
            min_samples = max(1, f*X.shape[1])
            cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=min_samples)
            lab = cl.fit_predict(X)
            noise = (lab == -1).mean()
            n_clusters = len(set(lab)) - (1 if -1 in lab else 0)
            score = n_clusters - (noise * 5)
            if score > best_score:
                best_params, best_score = (mcs, min_samples), score
    return {"min_cluster_size": best_params[0], "min_samples": best_params[1]}

def detect_risk_zones(df):
    """Performs intelligent, density-based spatial clustering using HDBSCAN."""
    logging.info("Detecting risk zones using advanced HDBSCAN clustering...")
    
    if len(df) < 20:
        df['Risk_Cluster'] = -1
        df['Risk_Zone'] = "Uncategorized"
        logging.warning("Not enough data points to perform meaningful clustering.")
        return df, 0

    coords = df[['latitude', 'longitude', 'Final_Risk_Score']].copy()
    coords_scaled = StandardScaler().fit_transform(coords)
    
    try:
        params = tune_hdbscan_params(coords_scaled, min_cluster_range=(10, min(100, len(df)//2)))
        logging.info(f"Tuned HDBSCAN params: {params}")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'], min_samples=params['min_samples'])
    except Exception as e:
        logging.warning(f"HDBSCAN tuning failed ({e}), falling back to defaults.")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)

    cluster_labels = clusterer.fit_predict(coords_scaled)
    df['Risk_Cluster'] = cluster_labels
    
    median_risk_per_cluster = df[df['Risk_Cluster'] != -1].groupby('Risk_Cluster')['Final_Risk_Score'].median()
    
    if not median_risk_per_cluster.empty:
        quantiles = median_risk_per_cluster.quantile([0.33, 0.66]).to_dict()
        def assign_zone_name(cluster_id):
            if cluster_id == -1: return "Outlier/Noise"
            median_risk = median_risk_per_cluster.get(cluster_id)
            if median_risk is None: return "Uncategorized"
            if median_risk >= quantiles[0.66]: return "High-Risk Zone"
            if median_risk >= quantiles[0.33]: return "Moderate-Risk Zone"
            return "Low-Risk Zone"
        df['Risk_Zone'] = df['Risk_Cluster'].apply(assign_zone_name)
    else:
        df['Risk_Zone'] = "Uncategorized"

    n_zones = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logging.info(f"Risk zonation complete. Found {n_zones} significant zones.")
    return df, n_zones

def assign_spatial_clusters(df: pd.DataFrame, min_cluster_size: int = 20, min_samples: int = 5, fallback_k: int = 10) -> pd.Series:
    """
    Assign spatial clusters for cross-validation to mitigate spatial autocorrelation.
    """
    if not {'latitude', 'longitude'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'latitude' and 'longitude' columns for spatial clustering.")

    coords = df[['latitude', 'longitude']].to_numpy(dtype=float)
    coords_scaled = StandardScaler().fit_transform(coords)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(coords_scaled)

    if (labels == -1).all():
        logging.info("Assigning spatial clusters using KMeans...")
        k = min(fallback_k, max(2, int(np.sqrt(len(df) / 25))))
        labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(coords_scaled)

    unique_labels = {lab: i for i, lab in enumerate(np.unique(labels))}
    norm_labels = np.array([unique_labels[lab] for lab in labels], dtype=int)
    return pd.Series(norm_labels, index=df.index, name='spatial_cluster')

def assign_operational_priority(df: pd.DataFrame, probability_col: str = 'Final_Risk_Score', uncertainty_col: str = 'Uncertainty', threshold: float = 0.5) -> pd.DataFrame:
    """
    Convert probabilistic risk and uncertainty into actionable categories.
    """
    p = df[probability_col].astype(float)
    u = df[uncertainty_col].astype(float)

    if {'Risk_Lower_90', 'Risk_Upper_90'}.issubset(df.columns):
        width = (df['Risk_Upper_90'] - df['Risk_Lower_90']).astype(float)
    else:
        width = u.astype(float)
    
    if len(width.dropna()) > 1:
        low_t, high_t = np.quantile(width.dropna(), [1/3, 2/3])
    else:
        low_t, high_t = 0.1, 0.2

    conditions = [
        (p >= threshold) & (width < low_t),
        (p >= threshold) & (width >= low_t),
        (p < threshold) & (width >= high_t),
    ]
    choices = [
        'High risk (confident)',
        'High risk (uncertain)',
        'Investigate (uncertain)',
    ]
    df['Operational_Priority'] = np.select(conditions, choices, default='Low risk (confident)')
    try:
        import config as _cfg
        triggers = getattr(_cfg, 'ACTION_TRIGGERS', [])
        def map_action(prob, wid):
            for t in triggers:
                if prob >= t.get('min_prob', 1.0) and wid <= t.get('max_uncertainty', 1.0):
                    return t.get('label', 'Action')
            return 'Monitor'
        df['Action_Trigger'] = [map_action(pp, ww) for pp, ww in zip(p.values, width.values)]
    except Exception:
        df['Action_Trigger'] = 'Monitor'
    return df

def get_global_shap_summary(df, feature_columns=None):
    """Aggregates per-pixel SHAP values to find global feature importance."""
    if 'SHAP_Values' not in df.columns or df['SHAP_Values'].isnull().all():
        return {}
    if feature_columns is None:
        try:
            width = len(df['SHAP_Values'].dropna().iloc[0])
            feature_columns = [f'f{i}' for i in range(width)]
        except Exception:
            return {}
    shap_matrix = np.vstack(df['SHAP_Values'].to_numpy())
    shap_df = pd.DataFrame(shap_matrix, columns=feature_columns)
    global_importance = shap_df.abs().mean().sort_values(ascending=False)
    return global_importance.to_dict()

def generate_report_summary(df):
    """Generates a comprehensive summary dictionary for the final HTML report."""
    logging.info("Generating final report summary...")
    avg_lower = df['Risk_Lower_90'].mean() if 'Risk_Lower_90' in df.columns else np.nan
    avg_upper = df['Risk_Upper_90'].mean() if 'Risk_Upper_90' in df.columns else np.nan
    avg_uncertainty = (df['Risk_Upper_90'] - df['Risk_Lower_90']).mean() if {'Risk_Upper_90','Risk_Lower_90'}.issubset(df.columns) else df.get('Uncertainty', pd.Series([np.nan]*len(df))).mean()
    
    summary = {
        'avg_risk_score': df['Final_Risk_Score'].mean(),
        'avg_risk_lower_90': avg_lower,
        'avg_risk_upper_90': avg_upper,
        'max_risk_score': df['Final_Risk_Score'].max(),
        'avg_uncertainty': float(avg_uncertainty) if not pd.isna(avg_uncertainty) else np.nan,
        'risk_zone_distribution': df['Risk_Zone'].value_counts(normalize=True).to_dict() if 'Risk_Zone' in df.columns else {},
        'high_risk_area_pct': np.nan,
        'dominant_land_cover': df['Land_Cover_Type'].mode()[0] if 'Land_Cover_Type' in df.columns and not df['Land_Cover_Type'].empty else 'N/A',
        'operational_priority_distribution': df['Operational_Priority'].value_counts(normalize=True).to_dict() if 'Operational_Priority' in df.columns else {}
    }
    return summary