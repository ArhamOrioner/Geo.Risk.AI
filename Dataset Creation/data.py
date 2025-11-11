import os
import sys
import time
import logging
import random
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import shapely.geometry
from shapely.ops import transform, unary_union
import pyproj

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# try to import georisk module (your georisk_enrich_export_v5)
import importlib.util

# ============================================================================
# FLOOD DATA ENRICHMENT PIPELINE v3.0
# Near-Real-Time Sources Only (No CHIRPS/ERA5/GloFAS)
#
# Data Sources & Latencies:
# - IMERG Early Run: 4 hours (event day precipitation intensity)
# - IMERG Late Run: 14 hours (historical precipitation accumulations)
# - SMAP Soil Moisture: 2-3 days (surface and subsurface moisture)
# - NOAA GFS: 3-6 hours (atmospheric conditions)
# - Static datasets: DEM, MERIT Hydro, HydroRIVERS, MCD12Q1 land cover
#
# Expected Features: ~39 (down from 109)
# Max Latency: 14 hours (operational capable)
# ============================================================================

# ----------------- CONFIG -----------------
EMDAT_PATH = "1_emdat.xlsx"
OUT_DIR = "/content"

# IMPROVED: Much larger buffers for flood phenomena
AOI_BUFFER_METERS = 25000  # 25km buffer for better watershed coverage
WATERSHED_BUFFER_METERS = 50000  # 50km for riverine floods


# Negative generation
NEGATIVE_RATIO = 1.0
NEG_EXCLUSION_RADIUS_METERS = 75000  # Larger exclusion radius
NEGATIVE_STRATEGY = "time_offset" 
NEG_OFFSET_RANGE_DAYS = (5, 14)  # One-per-positive: random 1–7 days before
NEGATIVE_ONE_PER_POSITIVE = True
NEGATIVE_DAYS_BEFORE = 7  # Not used when NEGATIVE_ONE_PER_POSITIVE=True

# Output settings
SINGLE_OUTPUT = True
OUTPUT_FILE = os.path.join(OUT_DIR, "enriched_events_imerg_smap_gfs.csv")
DROP_ZERO_NAN_COLUMNS = True
DROP_SPARSE_COLUMNS = True
ZERO_FRACTION_THRESHOLD = 0.95
NAN_FRACTION_THRESHOLD = 0.95

# Dataset temporal coverage (CRITICAL for version selection)
IMERG_GLOBAL_START = pd.Timestamp("2000-06-01")
IMERG_EARLY_START = pd.Timestamp("2019-06-01")
SMAP_V008_START = pd.Timestamp("2015-03-31")

# VALIDATION SETTINGS
ENABLE_VALIDATION = True
# Validation thresholds (updated for IMERG half-hourly data)
MIN_INTENSITY_FLASH_FLOOD_MM_HR = 10.0  # Min 1-hour max intensity for flash floods
MIN_PRECIP_RIVERINE_FLOOD_7D_MM = 25.0  # Min 7-day accumulation for riverine floods
SOIL_MOISTURE_RANGE = (0.0, 1.0)        # Valid SMAP soil moisture range
TEMPERATURE_RANGE_C = (-50.0, 60.0)     # Physically plausible temperature range

# Flood types that may have zero immediate precipitation
NON_PRECIP_FLOOD_TYPES = [
    "snowmelt",
    "coastal",
    "storm surge",
    "dam",
    "levee",
    "ice jam",
]

MAX_EVENTS_DEV = None
try:
    MAX_GEE_WORKERS = max(1, int(os.getenv("MAX_GEE_WORKERS", "5")))
except ValueError:
    MAX_GEE_WORKERS = 5
GEORISK_MODULE_PATH = "georisk_enrich_export_v5.py"

os.makedirs(OUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUT_DIR, "enrichment.log"))
    ]
)

# ---------------- Earth Engine init ----------------
try:
    import ee
    try:
        ee.Initialize()
        logging.info("Earth Engine initialized")
    except Exception:
        logging.info("Authenticating Earth Engine...")
        ee.Authenticate()
        ee.Initialize()
        logging.info("Earth Engine authenticated and initialized")
except Exception as e:
    raise RuntimeError(f"earthengine-api is required and must be authenticated. Error: {e}")

# Updated dataset IDs
MCD12Q1_ID = "MODIS/061/MCD12Q1"
MOD44W_ID = "MODIS/006/MOD44W"
DEM_ID = "USGS/SRTMGL1_003"
JRC_GSW_ID = "JRC/GSW1_4/GlobalSurfaceWater"
MERIT_HYDRO_ID = "MERIT/Hydro/v1_0_1"
HYDRO_RIVERS_ID = "WWF/HydroSHEDS/HydroRIVERS/v1/Global"

# ============= NEAR REAL-TIME DATA SOURCES =============
# IMERG Precipitation (single unified collection)
IMERG_ID        = "NASA/GPM_L3/IMERG_V07"
IMERG_BAND      = "precipitation"  # mm/hr
IMERG_GLOBAL_START = pd.Timestamp("2000-06-01")
IMERG_EARLY_START  = pd.Timestamp("2019-06-01")

# SMAP Soil Moisture (version-aware SPL4SMGP products)
SMAP_V007_ID = "NASA/SMAP/SPL4SMGP/007"
SMAP_V008_ID = "NASA/SMAP/SPL4SMGP/008"
SMAP_V008_START = pd.Timestamp("2015-03-31")
SMAP_SURFACE_BAND = "sm_surface"
SMAP_ROOTZONE_BAND = "sm_rootzone"

# NOAA GFS Atmospheric Data
NOAA_GFS_ID = "NOAA/GFS0P25"  # 3-6 hour latency

# Configuration
USE_IMERG_EARLY_FOR_EVENT_DAY = True  # Event day = Early Run (4h)
USE_IMERG_LATE_FOR_HISTORICAL = True  # History = Late Run (14h)

# (No legacy precip lag/rolling configuration in v3.0)

# Initialize GEE collections
dem = ee.Image(DEM_ID)
slope = ee.Terrain.slope(dem)
aspect = ee.Terrain.aspect(dem)

# Initialize IMERG collection
try:
    imerg_collection = ee.ImageCollection(IMERG_ID).select(IMERG_BAND)
    imerg_early = imerg_collection
    imerg_late = imerg_collection
    logging.info("✓ IMERG V07 initialized:")
    logging.info("   - Coverage: 2000-06-01 onwards")
    logging.info("   - Early Run data: 2019-06-01 onwards (4h latency)")
    logging.info("   - Late/Final Run data: 2000-06-01 onwards (14h latency)")
except Exception as e:
    logging.warning(f"IMERG initialization failed: {e}")
    imerg_collection = None
    imerg_early = None
    imerg_late = None

# Initialize SMAP v007
try:
    smap_v007 = ee.ImageCollection(SMAP_V007_ID).select([SMAP_SURFACE_BAND, SMAP_ROOTZONE_BAND])
    logging.info("✓ SMAP v007 initialized (legacy fallback) - DEPRECATED")
except Exception as e:
    logging.warning(f"SMAP v007 not available: {e}")
    smap_v007 = None

# Initialize SMAP v008
try:
    smap_v008 = ee.ImageCollection(SMAP_V008_ID).select([SMAP_SURFACE_BAND, SMAP_ROOTZONE_BAND])
    logging.info("✓ SMAP v008 initialized (2015-03-31 onwards) - RECOMMENDED")
except Exception as e:
    logging.warning(f"SMAP v008 not available: {e}")
    smap_v008 = None

# Initialize GFS
try:
    gfs_collection = ee.ImageCollection(NOAA_GFS_ID)
    logging.info("✓ NOAA GFS initialized (3-6h latency)")
except Exception as e:
    logging.warning(f"GFS not available: {e}")
    gfs_collection = None

# Dataset temporal coverage summary
logging.info("\n" + "=" * 60)
logging.info("DATASET TEMPORAL COVERAGE SUMMARY:")
logging.info("=" * 60)
logging.info(f"IMERG Global Start: {IMERG_GLOBAL_START.date()}")
logging.info(f"IMERG Early Run Start: {IMERG_EARLY_START.date()}")
logging.info(f"SMAP v008 Start: {SMAP_V008_START.date()}")
logging.info("Events before 2000-06-01: NO IMERG DATA")
logging.info("Events 2000-06-01 to 2019-05-31: LATE/FINAL RUN ONLY")
logging.info("Events 2019-06-01 onwards: EARLY RUN AVAILABLE")
logging.info("=" * 60 + "\n")

# NDVI acceptable with 3-8 day MODIS lag (slow-changing index)
MODIS_NDVI_ID = "MODIS/061/MOD13Q1"
try:
    modis_ndvi = ee.ImageCollection(MODIS_NDVI_ID).select("NDVI")
    logging.info("✓ MODIS NDVI initialized (3-8 day lag acceptable)")
except Exception as e:
    logging.warning(f"MODIS NDVI not available: {e}")
    modis_ndvi = None


try:
    jrc_gsw = ee.Image(JRC_GSW_ID).select("occurrence")
except Exception:
    jrc_gsw = None
    logging.warning("JRC Global Surface Water not available")

try:
    merit_hydro = ee.Image(MERIT_HYDRO_ID).select("upa")
except Exception:
    merit_hydro = None
    logging.warning("MERIT Hydro not available")

try:
    hydro_rivers = ee.FeatureCollection(HYDRO_RIVERS_ID)
except Exception:
    hydro_rivers = None
    logging.warning("HydroRIVERS not available")

# ---------------- Load georisk module if present -----------------
georisk = None
if os.path.exists(GEORISK_MODULE_PATH):
    try:
        spec = importlib.util.spec_from_file_location("georisk", GEORISK_MODULE_PATH)
        georisk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(georisk)
        logging.info(f"Imported {GEORISK_MODULE_PATH}")
    except Exception as e:
        logging.warning(f"Failed to import {GEORISK_MODULE_PATH}: {e}")

# ---------------- Utility functions ----------------

def load_emdat(path):
    """Enhanced EMDAT loader with better error handling."""
    logging.info(f"Loading EMDAT data from {path}")
    
    try:
        if path.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Could not read EMDAT file {path}: {e}")
    
    logging.info(f"Loaded {len(df)} raw records from EMDAT")
    
    # Enhanced date parsing
    if any(c.lower() == "start_date" for c in df.columns):
        col = next(c for c in df.columns if c.lower() == "start_date")
        df["flood_start_date"] = pd.to_datetime(df[col], errors="coerce")
    elif all(c in df.columns for c in ("Start Year","Start Month","Start Day")):
        df["flood_start_date"] = pd.to_datetime(
            df["Start Year"].astype(str) + "-" + 
            df["Start Month"].astype(str) + "-" + 
            df["Start Day"].astype(str), 
            errors="coerce"
        )
    else:
        # Auto-detect date columns
        found = False
        for c in df.columns:
            if any(keyword in c.lower() for keyword in ['date', 'start', 'begin']):
                tmp = pd.to_datetime(df[c], errors="coerce")
                if tmp.notna().sum() > len(df) * 0.5:  # At least 50% valid dates
                    df["flood_start_date"] = tmp
                    found = True
                    logging.info(f"Using column '{c}' as start date")
                    break
        if not found:
            raise ValueError("Could not detect start date column in EMDAT file")
    
    # Enhanced coordinate parsing
    lat_cols = [c for c in df.columns if c.lower() in ("latitude", "lat", "y", "coord_y")]
    lon_cols = [c for c in df.columns if c.lower() in ("longitude", "lon", "long", "x", "coord_x")]
    
    if not lat_cols or not lon_cols:
        raise ValueError("Latitude/Longitude columns not found in EMDAT file")
    
    df["latitude"] = pd.to_numeric(df[lat_cols[0]], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_cols[0]], errors="coerce")
    
    # Validate coordinates
    coord_mask = (
        df["latitude"].between(-90, 90) & 
        df["longitude"].between(-180, 180) &
        df["flood_start_date"].notna()
    )
    
    invalid_count = len(df) - coord_mask.sum()
    if invalid_count > 0:
        logging.warning(f"Removing {invalid_count} records with invalid coordinates or dates")
    
    df = df[coord_mask].reset_index(drop=True)
    
    # Add flood type if not present
    if "flood_type" not in df.columns:
        df["flood_type"] = "Unknown"
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df.copy(), 
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), 
        crs="EPSG:4326"
    )
    gdf["target"] = 1
    
    logging.info(f"Successfully processed {len(gdf)} valid flood events")
    return gdf

def get_optimal_buffer(flood_type, base_buffer=AOI_BUFFER_METERS):
    """Return optimal buffer size based on flood type."""
    flood_type_lower = str(flood_type).lower()
    
    if any(term in flood_type_lower for term in ['flash', 'urban', 'local']):
        return base_buffer  # 25km for flash floods
    elif any(term in flood_type_lower for term in ['riverine', 'river', 'coastal']):
        return WATERSHED_BUFFER_METERS  # 50km for riverine floods
    else:
        return base_buffer


def bbox_from_point(lat, lon, buffer_m=AOI_BUFFER_METERS):
    """Enhanced bbox calculation with better projection handling."""
    try:
        # Use Azimuthal Equidistant projection centered on the point
        wgs84 = pyproj.CRS("EPSG:4326")
        proj_string = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
        aeqd_crs = pyproj.CRS.from_proj4(proj_string)
        
        to_aeqd = pyproj.Transformer.from_crs(wgs84, aeqd_crs, always_xy=True).transform
        to_wgs = pyproj.Transformer.from_crs(aeqd_crs, wgs84, always_xy=True).transform
        
        pt = shapely.geometry.Point(lon, lat)
        pt_proj = transform(to_aeqd, pt)
        buf_proj = pt_proj.buffer(buffer_m)
        buf_wgs = transform(to_wgs, buf_proj)
        
        lon_min, lat_min, lon_max, lat_max = buf_wgs.bounds
        return float(lon_min), float(lon_max), float(lat_min), float(lat_max)
    except Exception as e:
        logging.debug(f"Projection failed, using approximate degree buffer: {e}")
        # Fallback: approximate degree buffer
        dlon = buffer_m / (111320.0 * np.cos(np.radians(lat)))
        dlat = buffer_m / 110540.0
        return float(lon - dlon), float(lon + dlon), float(lat - dlat), float(lat + dlat)

def gee_reduce_safe(img, reducer, geometry, scale, description="", max_pixels=1e9, max_attempts=3):
    """Enhanced safe GEE reduction with retry + larger tile scale for complex ops."""
    backoff_seconds = 1.0
    for attempt in range(max_attempts):
        try:
            result = img.reduceRegion(
                reducer=reducer,
                geometry=geometry,
                scale=scale,
                maxPixels=max_pixels,
                bestEffort=True,
                tileScale=4
            ).getInfo()

            if result is None or len(result) == 0:
                logging.debug(f"Empty result for {description}")
                return None

            # Check for valid values
            valid_result = {}
            for k, v in result.items():
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    valid_result[k] = v

            return valid_result if valid_result else None

        except Exception as e:
            if attempt < max_attempts - 1:
                wait_time = backoff_seconds * (attempt + 1)
                time.sleep(wait_time)
                continue
            logging.debug(f"GEE reduction failed for {description}: {e}")
            return None

# ---------------- Near-Real-Time feature extractors ----------------

def extract_imerg_precipitation_features(aoi, event_date, flood_type):
    """
    Extract precipitation using IMERG Early (event day) + Late (historical).
    Returns IMERG intensity and accumulation features.
    """
    features = {}
    try:
        if imerg_early is None and imerg_late is None:
            return features

        event_dt = pd.to_datetime(event_date)

        if event_dt < IMERG_GLOBAL_START:
            logging.warning(
                f"IMERG unavailable before {IMERG_GLOBAL_START.date()} (event: {event_dt.date()}); setting zeros."
            )
            features.update({
                "gee_imerg_sum_24h_mm": 0.0,
                "gee_imerg_max_1h_intensity_mm": 0.0,
                "gee_imerg_max_3h_intensity_mm": 0.0,
                "gee_imerg_max_6h_intensity_mm": 0.0,
                "gee_imerg_sum_3d_before_mm": 0.0,
                "gee_imerg_sum_7d_before_mm": 0.0,
                "gee_imerg_max_daily_7d_mm": 0.0,
                "gee_imerg_intensity_3d_mm_per_day": 0.0,
                "gee_imerg_intensity_7d_mm_per_day": 0.0,
                "imerg_max_1h_intensity_mm": 0.0,
                "imerg_mean_1h_intensity_mm": 0.0,
                "imerg_data_missing": 1,
                "imerg_source": "none",
                "imerg_coverage_note": "pre_imerg_era",
            })
            return features

        def _first_value(result_dict):
            if not result_dict:
                return np.nan
            try:
                return float(next(iter(result_dict.values())))
            except Exception:
                return np.nan

        day_start = event_dt.normalize()
        day_end   = day_start + pd.Timedelta(days=1)

        # Event day: IMERG Early
        features.setdefault("imerg_max_1h_intensity_mm", np.nan)
        features.setdefault("imerg_mean_1h_intensity_mm", np.nan)
        features["imerg_data_missing"] = 1
        features["imerg_extraction_failed"] = 1
        features["imerg_source"] = "unavailable"
        features.setdefault("imerg_coverage_note", "")

        use_early = (
            USE_IMERG_EARLY_FOR_EVENT_DAY
            and imerg_early is not None
            and event_dt >= IMERG_EARLY_START
        )

        event_day_coll = None
        event_day_source = None
        if use_early:
            event_day_coll = imerg_early.filterDate(day_start.strftime("%Y-%m-%d"), day_end.strftime("%Y-%m-%d"))
            event_day_source = "early"
        elif imerg_late is not None:
            event_day_coll = imerg_late.filterDate(day_start.strftime("%Y-%m-%d"), day_end.strftime("%Y-%m-%d"))
            event_day_source = "late"

        extraction_success = False
        if event_day_coll is not None:
            for attempt in range(2):
                try:
                    size = event_day_coll.size().getInfo()
                    if size == 0:
                        raise ValueError(
                            f"No IMERG data available for {event_dt.date()} (collection size: 0, source={event_day_source})"
                        )

                    logging.info(
                        f"[IMERG] Event-day extraction attempt {attempt + 1}/2 using {event_day_source.upper()} run for {event_dt.strftime('%Y-%m-%d')}"
                    )
                    sum_24h = event_day_coll.sum().multiply(0.5)
                    max_rate = event_day_coll.max()
                    mean_rate = event_day_coll.mean()

                    r24 = gee_reduce_safe(sum_24h, ee.Reducer.mean(), aoi, 10000, "imerg_sum_24h_event")
                    r_max = gee_reduce_safe(max_rate, ee.Reducer.mean(), aoi, 10000, "imerg_max_1h_event")
                    r_mean = gee_reduce_safe(mean_rate, ee.Reducer.mean(), aoi, 10000, "imerg_mean_1h_event")

                    if r24 is None or r_max is None:
                        raise ValueError("Empty IMERG event-day reducer output")

                    sum_24h_val = _first_value(r24)
                    max_1h_val = _first_value(r_max)
                    mean_1h_val = _first_value(r_mean)

                    if np.isnan(sum_24h_val) or np.isnan(max_1h_val):
                        raise ValueError("Invalid IMERG event-day statistics")

                    features["gee_imerg_sum_24h_mm"] = sum_24h_val
                    features["gee_imerg_max_1h_intensity_mm"] = max_1h_val
                    features["imerg_max_1h_intensity_mm"] = max_1h_val
                    features["imerg_mean_1h_intensity_mm"] = mean_1h_val if not np.isnan(mean_1h_val) else np.nan

                    features["gee_imerg_max_3h_intensity_mm"] = float(max_1h_val) * 1.5
                    features["gee_imerg_max_6h_intensity_mm"] = float(max_1h_val) * 2.0

                    features["imerg_data_missing"] = 0
                    features["imerg_extraction_failed"] = 0
                    features["imerg_source"] = event_day_source
                    extraction_success = True
                    break
                except Exception as e:
                    logging.warning(
                        f"[IMERG] Event-day extraction attempt {attempt + 1}/2 ({event_day_source}) for {event_dt.strftime('%Y-%m-%d')} failed: {e}"
                    )
                    if (
                        event_day_source == "early"
                        and attempt == 0
                        and imerg_late is not None
                    ):
                        logging.info(
                            f"[IMERG] Falling back to IMERG Late/Final Run for {event_dt.strftime('%Y-%m-%d')}"
                        )
                        event_day_coll = imerg_late.filterDate(
                            day_start.strftime("%Y-%m-%d"),
                            day_end.strftime("%Y-%m-%d")
                        )
                        event_day_source = "late_final"
                        time.sleep(2)
                        continue
                    if attempt < 1:
                        time.sleep(5)
                        continue

        if not extraction_success:
            features["gee_imerg_sum_24h_mm"] = 0.0
            features["gee_imerg_max_1h_intensity_mm"] = 0.0
            features["gee_imerg_max_3h_intensity_mm"] = 0.0
            features["gee_imerg_max_6h_intensity_mm"] = 0.0
            features["imerg_max_1h_intensity_mm"] = 0.0
            features["imerg_mean_1h_intensity_mm"] = 0.0
            features["imerg_data_missing"] = 1
            features["imerg_extraction_failed"] = 1
            if event_day_source:
                features["imerg_source"] = event_day_source
            features.setdefault("imerg_coverage_note", "")
        else:
            if event_dt < IMERG_GLOBAL_START:
                features["imerg_coverage_note"] = "pre_imerg_era"
            else:
                features.setdefault("imerg_coverage_note", "")

        # History: IMERG Late
        if imerg_late is not None:
            sum3_start = (event_dt - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            sum7_start = (event_dt - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            end_str    = event_dt.strftime("%Y-%m-%d")

            for attempt in range(2):
                try:
                    historical_coll = imerg_late.filterDate(sum7_start, end_str)
                    hist_size = historical_coll.size().getInfo()
                    if hist_size == 0:
                        raise ValueError(f"No IMERG historical data for window ending {end_str}")

                    sum3_img = imerg_late.filterDate(sum3_start, end_str).sum().multiply(0.5)
                    sum7_img = imerg_late.filterDate(sum7_start, end_str).sum().multiply(0.5)

                    r3b = gee_reduce_safe(sum3_img, ee.Reducer.mean(), aoi, 5000, "imerg_sum_3d_before")
                    r7b = gee_reduce_safe(sum7_img, ee.Reducer.mean(), aoi, 5000, "imerg_sum_7d_before")

                    features["gee_imerg_sum_3d_before_mm"] = _first_value(r3b)
                    features["gee_imerg_sum_7d_before_mm"] = _first_value(r7b)

                    def daily_sum(day_offset):
                        d0 = event_dt - pd.Timedelta(days=day_offset)
                        d1 = d0 + pd.Timedelta(days=1)
                        return imerg_late.filterDate(d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d")).sum().multiply(0.5).set({'d0': d0.strftime("%Y-%m-%d")})

                    daily_imgs = ee.ImageCollection([daily_sum(k) for k in range(1, 8)])
                    rmaxd = gee_reduce_safe(daily_imgs.max(), ee.Reducer.mean(), aoi, 5000, "imerg_max_daily_7d")
                    features["gee_imerg_max_daily_7d_mm"] = _first_value(rmaxd)
                    break
                except Exception as e:
                    logging.warning(
                        f"IMERG historical extraction failed (attempt {attempt + 1}/2) for {event_date}: {e}"
                    )
                    if attempt < 1:
                        time.sleep(5)
                        continue

                    features.setdefault("gee_imerg_sum_3d_before_mm", 0.0)
                    features.setdefault("gee_imerg_sum_7d_before_mm", 0.0)
                    features.setdefault("gee_imerg_max_daily_7d_mm", 0.0)

        # Derived intensities
        try:
            s3 = features.get("gee_imerg_sum_3d_before_mm")
            features["gee_imerg_intensity_3d_mm_per_day"] = float(s3) / 3.0 if s3 not in (None, np.nan) else np.nan
        except Exception:
            features["gee_imerg_intensity_3d_mm_per_day"] = np.nan
        try:
            s7 = features.get("gee_imerg_sum_7d_before_mm")
            features["gee_imerg_intensity_7d_mm_per_day"] = float(s7) / 7.0 if s7 not in (None, np.nan) else np.nan
        except Exception:
            features["gee_imerg_intensity_7d_mm_per_day"] = np.nan

    except Exception:
        pass

    return features


def extract_smap_features(aoi, event_date):
    """Extract SMAP soil moisture with 3-day lookback and 30-60d anomaly."""
    features = {}
    if smap_v007 is None and smap_v008 is None:
        return features

    event_dt = pd.to_datetime(event_date)
    use_v008 = smap_v008 is not None and event_dt >= SMAP_V008_START

    collection = None
    smap_version_label = None

    if use_v008:
        collection = smap_v008
        smap_version_label = "SPL4SMGP_v008"
    elif smap_v007 is not None:
        collection = smap_v007
        smap_version_label = "SPL4SMGP_v007"
    elif smap_v008 is not None:
        # Fallback: v008 available but event date before launch
        collection = smap_v008
        smap_version_label = "SPL4SMGP_v008"

    if collection is None:
        return features

    logging.info(
        f"SMAP extraction using {smap_version_label} for event on {event_dt.strftime('%Y-%m-%d')}"
    )
    features["gee_smap_version"] = smap_version_label
    if "v008" in smap_version_label.lower():
        features["smap_version_used"] = "v008"
    elif "v007" in smap_version_label.lower():
        features["smap_version_used"] = "v007"
    else:
        features["smap_version_used"] = "unknown"
    features["smap_data_missing"] = 0

    try:
        cur_start = (event_dt - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        cur_end   = (event_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        cur_coll  = collection.filterDate(cur_start, cur_end)

        try:
            ssm_val = gee_reduce_safe(cur_coll.select(SMAP_SURFACE_BAND).mean(), ee.Reducer.mean(), aoi, 10000, "sm_surface_only")
            features["gee_smap_surface_soil_moisture"] = float(next(iter(ssm_val.values()))) if ssm_val else np.nan
        except Exception:
            features["gee_smap_surface_soil_moisture"] = np.nan

        try:
            susm_val = gee_reduce_safe(cur_coll.select(SMAP_ROOTZONE_BAND).mean(), ee.Reducer.mean(), aoi, 10000, "sm_rootzone_only")
            features["gee_smap_subsurface_soil_moisture"] = float(next(iter(susm_val.values()))) if susm_val else np.nan
        except Exception:
            features["gee_smap_subsurface_soil_moisture"] = np.nan
    except Exception:
        features["gee_smap_surface_soil_moisture"] = np.nan
        features["gee_smap_subsurface_soil_moisture"] = np.nan
        features["smap_data_missing"] = 1

    try:
        clim_start = (event_dt - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        clim_end   = (event_dt - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        clim_coll  = collection.filterDate(clim_start, clim_end)
        ssm_clim   = clim_coll.select(SMAP_SURFACE_BAND).mean()
        ssm_cur    = collection.filterDate(cur_start, cur_end).select(SMAP_SURFACE_BAND).mean()
        r_cur_ssm  = gee_reduce_safe(ssm_cur, ee.Reducer.mean(), aoi, 10000, "ssm_current")
        r_clim_ssm = gee_reduce_safe(ssm_clim, ee.Reducer.mean(), aoi, 10000, "ssm_climatology")
        if r_cur_ssm and r_clim_ssm:
            cur_val  = float(next(iter(r_cur_ssm.values())))
            clim_val = float(next(iter(r_clim_ssm.values())))
            features["gee_smap_soil_moisture_anomaly"] = cur_val - clim_val
        else:
            features["gee_smap_soil_moisture_anomaly"] = np.nan
    except Exception:
        features["gee_smap_soil_moisture_anomaly"] = np.nan

    return features


def extract_gfs_features(aoi, event_date):
    """Extract atmospheric conditions from NOAA GFS (3-6 hour latency)."""
    features = {}
    if gfs_collection is None:
        return features

    event_dt = pd.to_datetime(event_date)
    try:
        start = (event_dt - pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S")
        end   = event_dt.strftime("%Y-%m-%dT%H:%M:%S")
        gfs   = gfs_collection.filterDate(start, end).mean()

        try:
            temp_c = gfs.select("temperature_2m_above_ground").subtract(273.15)
            r_t = gee_reduce_safe(temp_c, ee.Reducer.mean(), aoi, 25000, "gfs_temp2m")
            features["gee_gfs_temp2m_celsius"] = float(next(iter(r_t.values()))) if r_t else np.nan
        except Exception:
            features["gee_gfs_temp2m_celsius"] = np.nan

        try:
            rh = gfs.select("relative_humidity_2m_above_ground")
            r_rh = gee_reduce_safe(rh, ee.Reducer.mean(), aoi, 25000, "gfs_rh2m")
            features["gee_gfs_relative_humidity_pct"] = float(next(iter(r_rh.values()))) if r_rh else np.nan
        except Exception:
            features["gee_gfs_relative_humidity_pct"] = np.nan

        try:
            ps = gfs.select("pressure_surface")
            r_ps = gee_reduce_safe(ps, ee.Reducer.mean(), aoi, 25000, "gfs_surface_pressure")
            features["gee_gfs_surface_pressure_pa"] = float(next(iter(r_ps.values()))) if r_ps else np.nan
        except Exception:
            features["gee_gfs_surface_pressure_pa"] = np.nan

        try:
            pw = gfs.select("precipitable_water_entire_atmosphere")
            r_pw = gee_reduce_safe(pw, ee.Reducer.mean(), aoi, 25000, "gfs_pw")
            features["gee_gfs_precipitable_water_kg_m2"] = float(next(iter(r_pw.values()))) if r_pw else np.nan
        except Exception:
            features["gee_gfs_precipitable_water_kg_m2"] = np.nan

        try:
            u10 = gfs.select("u_component_of_wind_10m_above_ground")
            v10 = gfs.select("v_component_of_wind_10m_above_ground")
            ws  = u10.pow(2).add(v10.pow(2)).sqrt()
            r_ws = gee_reduce_safe(ws, ee.Reducer.mean(), aoi, 25000, "gfs_wind10m")
            features["gee_gfs_wind_speed_ms"] = float(next(iter(r_ws.values()))) if r_ws else np.nan
        except Exception:
            features["gee_gfs_wind_speed_ms"] = np.nan

    except Exception:
        pass

    return features


def compute_engineered_features(features):
    """Compute hydrologic indices from extracted features."""
    engineered = {}

    def safe_get(key, default=np.nan):
        val = features.get(key, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except Exception:
            return default

    try:
        p24 = safe_get("gee_imerg_sum_24h_mm")
        p3  = safe_get("gee_imerg_sum_3d_before_mm")
        p7  = safe_get("gee_imerg_sum_7d_before_mm")
        max1h = safe_get("gee_imerg_max_1h_intensity_mm")
        slope_deg = safe_get("gee_slope_mean")
        slope_rad = np.radians(slope_deg) if not np.isnan(slope_deg) else np.nan
        ssm  = safe_get("gee_smap_surface_soil_moisture")
        upa  = safe_get("gee_merit_upa_mean")
        ws   = safe_get("gee_gfs_wind_speed_ms")
        pw   = safe_get("gee_gfs_precipitable_water_kg_m2")

        engineered["gee_api_weighted_mm"] = (0.5 * p24 + 0.3 * p3 + 0.2 * p7) if not any(np.isnan([p24, p3, p7])) else np.nan

        denom = (p24 / 24.0) if not np.isnan(p24) and p24 != 0 else np.nan
        engineered["gee_flashiness_index"] = (max1h / denom) if (not np.isnan(max1h) and not np.isnan(denom) and denom != 0) else np.nan

        engineered["gee_saturation_proxy"] = (ssm / (p7 + 1.0)) if not np.isnan(ssm) and not np.isnan(p7) else np.nan

        engineered["gee_runoff_potential"] = ((p24 * slope_deg) / (ssm + 0.01)) if not any(np.isnan([p24, slope_deg, ssm])) else np.nan

        try:
            tan_s = np.tan((slope_rad if not np.isnan(slope_rad) else 0.0) + 0.001)
            engineered["gee_twi"] = np.log(((upa + 1.0) / tan_s)) if not np.isnan(upa) and tan_s != 0 else np.nan
        except Exception:
            engineered["gee_twi"] = np.nan

        engineered["gee_moisture_flux"] = (pw * ws) if not any(np.isnan([pw, ws])) else np.nan

    except Exception:
        pass

    return engineered

def extract_gee_features(lat, lon, event_date, flood_type="Unknown"):
    """Near-real-time GEE feature extraction (IMERG/SMAP/GFS + static)."""
    features = {}
    event_dt = pd.to_datetime(event_date)
    
    # Get appropriate buffer size
    buffer_size = get_optimal_buffer(flood_type)
    
    try:
        lon_min, lon_max, lat_min, lat_max = bbox_from_point(lat, lon, buffer_size)
        aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        logging.debug(f"Using {buffer_size}m buffer for flood type: {flood_type}")
    except Exception as e:
        logging.warning(f"AOI creation failed: {e}")
        return features

    # Use georisk module if available
    if georisk and hasattr(georisk, "extract_geerisk_features_for_point"):
        try:
            gr = georisk.extract_geerisk_features_for_point(
                lat, lon, event_dt, buffer_size
            )
            if isinstance(gr, dict):
                features.update(gr)
                logging.debug(f"Georisk features extracted: {len(gr)}")
        except Exception as e:
            logging.debug(f"Georisk extraction failed: {e}")

    # 1) IMERG precipitation (Early for event day; Late for history)
    try:
        precip_features = extract_imerg_precipitation_features(aoi, event_dt, flood_type)
        features.update(precip_features)
        logging.debug(f"Extracted {len(precip_features)} IMERG features")
    except Exception as e:
        logging.warning(f"IMERG extraction failed: {e}")

    # 2) SMAP soil moisture
    try:
        smap_features = extract_smap_features(aoi, event_dt)
        features.update(smap_features)
        logging.debug(f"Extracted {len(smap_features)} SMAP features")
    except Exception as e:
        logging.warning(f"SMAP extraction failed: {e}")

    # 3) GFS atmospheric
    try:
        gfs_features = extract_gfs_features(aoi, event_dt)
        features.update(gfs_features)
        logging.debug(f"Extracted {len(gfs_features)} GFS features")
    except Exception as e:
        logging.warning(f"GFS extraction failed: {e}")

    # Topographic features
    try:
        topo = dem.addBands(slope).addBands(aspect)
        topo_stats = gee_reduce_safe(topo, ee.Reducer.mean(), aoi, 90, "topography")
        if topo_stats:
            for k, v in topo_stats.items():
                features[f"gee_{k}_mean"] = float(v) if v is not None else np.nan
    except Exception as e:
        logging.debug(f"Topography extraction failed: {e}")

    # JRC Global Surface Water: occurrence only (mean)
    try:
        gsw_img_full = ee.Image(JRC_GSW_ID).select("occurrence")
        bstats = gee_reduce_safe(gsw_img_full, ee.Reducer.mean(), aoi, 30, "gsw_occurrence")
        if bstats:
            features["gee_gsw_occurrence_mean"] = float(next(iter(bstats.values())))
    except Exception as e:
        logging.debug(f"GSW occurrence extraction failed: {e}")

    # MERIT Hydro upstream area (upa) mean
    if merit_hydro is not None:
        try:
            upa_stats = gee_reduce_safe(merit_hydro, ee.Reducer.mean(), aoi, 90, "merit_upa")
            if upa_stats:
                features["gee_merit_upa_mean"] = float(next(iter(upa_stats.values())))
        except Exception as e:
            logging.debug(f"MERIT upa extraction failed: {e}")

    # (Vegetation indices removed in v3.0)

    # Landcover mode (MCD12Q1)
    try:
        lc_img = ee.Image(MCD12Q1_ID).select("LC_Type1")
        lc_stat = gee_reduce_safe(lc_img, ee.Reducer.mode(), aoi, 500, "landcover_mode")
        if lc_stat:
            features["gee_landcover_mode"] = float(next(iter(lc_stat.values())))
    except Exception as e:
        logging.debug(f"Landcover extraction failed: {e}")

    # (Evapotranspiration removed in v3.0)

    

    # (Distance to water removed in v3.0)

    # NDBI (Normalized Difference Built-up Index) and engineered features
    # - NDBI: Landsat 8 C2 L2 (SR_B6=SWIR1, SR_B5=NIR), mean over 30 days before event
    # - Engineered: antecedent_moisture_proxy, flashiness_index_7d, precip_x_slope
    try:
        # NDVI/NDBI/NDWI: 15-30 day lag is ACCEPTABLE (slow-changing indices)
        try:
            if modis_ndvi is not None:
                ndvi_start = (event_dt - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                ndvi_end = event_dt.strftime("%Y-%m-%d")
                ndvi_img = modis_ndvi.filterDate(ndvi_start, ndvi_end).mean().multiply(0.0001)
                ndvi_stats = gee_reduce_safe(ndvi_img, ee.Reducer.mean(), aoi, 250, "NDVI_30d_mean")
                features["gee_ndvi_mean_30d"] = float(next(iter(ndvi_stats.values()))) if ndvi_stats else np.nan
            else:
                features["gee_ndvi_mean_30d"] = np.nan
        except Exception as e_ndvi:
            logging.debug(f"NDVI extraction failed: {e_ndvi}")
            features["gee_ndvi_mean_30d"] = np.nan

        try:
            l8_start = (event_dt - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            l8_end = event_dt.strftime("%Y-%m-%d")
            l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(l8_start, l8_end)
            # Compute NDBI per image with scale/offset applied per L2 SR docs
            def _ndbi(img):
                scaled = img.select(["SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
                nir = scaled.select("SR_B5")
                swir1 = scaled.select("SR_B6")
                ndbi = swir1.subtract(nir).divide(swir1.add(nir).add(1e-6)).rename("ndbi")
                return ndbi
            ndbi_coll = l8.map(_ndbi)
            ndbi_mean = ee.ImageCollection(ndbi_coll).mean()
            ndbi_stat = gee_reduce_safe(ndbi_mean, ee.Reducer.mean(), aoi, 30, "NDBI_30d_mean")
            if ndbi_stat:
                try:
                    features["gee_ndbi_mean_30d"] = float(next(iter(ndbi_stat.values())))
                except Exception:
                    features["gee_ndbi_mean_30d"] = np.nan
            else:
                features["gee_ndbi_mean_30d"] = np.nan
        except Exception as e_ndbi:
            logging.debug(f"NDBI extraction failed: {e_ndbi}")
            features["gee_ndbi_mean_30d"] = np.nan

        # Engineered features using IMERG precipitation and slope metrics
        def _num_or_nan(val):
            try:
                if val is None:
                    return np.nan
                x = float(val)
                # Keep NaNs as NaN
                return x if not (isinstance(x, float) and np.isnan(x)) else np.nan
            except Exception:
                return np.nan

        s3 = _num_or_nan(features.get("gee_imerg_sum_3d_before_mm"))
        s7 = _num_or_nan(features.get("gee_imerg_sum_7d_before_mm"))
        pmax1h = _num_or_nan(features.get("gee_imerg_max_1h_intensity_mm"))
        slope_mean = _num_or_nan(features.get("gee_slope_mean"))

        # Antecedent moisture proxy
        if not (np.isnan(s3) or np.isnan(s7)):
            features["gee_antecedent_moisture_proxy"] = 0.6 * s3 + 0.4 * s7
        else:
            features["gee_antecedent_moisture_proxy"] = np.nan

        # Flashiness index (7d)
        if not (np.isnan(pmax1h) or np.isnan(s7)) and s7 > 0:
            features["gee_flashiness_index_7d"] = pmax1h / (s7 / 7.0 + 1e-6)
        else:
            features["gee_flashiness_index_7d"] = np.nan

        # Precip x slope interaction
        if not (np.isnan(s7) or np.isnan(slope_mean)):
            features["gee_precip_x_slope"] = s7 * slope_mean
        else:
            features["gee_precip_x_slope"] = np.nan
    except Exception as e:
        logging.debug(f"Engineered GEE metrics failed: {e}")
        features.setdefault("gee_ndbi_mean_30d", np.nan)
        features.setdefault("gee_antecedent_moisture_proxy", np.nan)
        features.setdefault("gee_flashiness_index_7d", np.nan)
        features.setdefault("gee_precip_x_slope", np.nan)

    # NDWI (Normalized Difference Water Index) - Sentinel-2 or Landsat 8
    try:
        ndwi_start = (event_dt - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        ndwi_end = event_dt.strftime("%Y-%m-%d")
        
        # Use Landsat 8 (SR_B5=NIR, SR_B3=Green)
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(ndwi_start, ndwi_end)
        
        def _ndwi(img):
            scaled = img.select(["SR_B3", "SR_B5"]).multiply(0.0000275).add(-0.2)
            green = scaled.select("SR_B3")
            nir = scaled.select("SR_B5")
            ndwi = green.subtract(nir).divide(green.add(nir).add(1e-6)).rename("ndwi")
            return ndwi
        
        ndwi_coll = l8.map(_ndwi)
        ndwi_mean = ndwi_coll.mean()
        ndwi_stat = gee_reduce_safe(ndwi_mean, ee.Reducer.mean(), aoi, 30, "NDWI_30d")
        features["gee_ndwi_mean_30d"] = float(next(iter(ndwi_stat.values()))) if ndwi_stat else np.nan
    except Exception as e:
        logging.debug(f"NDWI extraction failed: {e}")
        features["gee_ndwi_mean_30d"] = np.nan

    # Impervious Surface Fraction (use NLCD or approximation from Landsat urban mask)
    try:
        # Ensure landcover image is available
        if 'lc_img' not in locals():
            lc_img = ee.Image(MCD12Q1_ID).select("LC_Type1")
        # Simple urban fraction from landcover (LC_Type1: 13=urban)
        urban_frac = (lc_img.eq(13)).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=500,
            maxPixels=1e9
        ).getInfo()
        features["gee_impervious_fraction"] = float(urban_frac.get('LC_Type1', 0))
    except Exception:
        features["gee_impervious_fraction"] = np.nan

    

    # Engineered features
    try:
        engineered = compute_engineered_features(features)
        features.update(engineered)
        logging.debug(f"Computed {len(engineered)} engineered features")
    except Exception as e:
        logging.warning(f"Engineered feature computation failed: {e}")

    logging.info(f"Total extracted features: {len(features)}")
    return features

 

# ---------------- Enhanced validation functions ----------------

def validate_flood_features(features, flood_type, event_date, lat, lon):
    """Validate features for physical consistency and data availability."""
    issues = []
    warnings = []

    flood_type_lower = str(flood_type).lower()

    # Check 1 - IMERG sub-daily intensity exists
    imerg_intensity_keys = [
        "gee_imerg_max_1h_intensity_mm",
        "gee_imerg_max_3h_intensity_mm",
        "gee_imerg_max_6h_intensity_mm",
    ]
    if not any(features.get(k) not in (None, np.nan) for k in imerg_intensity_keys):
        issues.append("missing_imerg_intensity_features")

    # Check 2 - Validate precipitation thresholds (warnings only)
    min_flash = globals().get("MIN_INTENSITY_FLASH_FLOOD_MM_HR", 10.0)
    min_riv7d = globals().get("MIN_PRECIP_RIVERINE_FLOOD_7D_MM", 25.0)

    def _safe_float(val):
        try:
            v = float(val)
            if isinstance(v, float) and np.isnan(v):
                return np.nan
            return v
        except Exception:
            return np.nan

    max1h = _safe_float(features.get("gee_imerg_max_1h_intensity_mm"))
    sum7 = _safe_float(features.get("gee_imerg_sum_7d_before_mm"))

    imerg_missing_flag = features.get("imerg_data_missing")
    imerg_missing = False
    if imerg_missing_flag is not None:
        try:
            imerg_missing = bool(int(imerg_missing_flag))
        except Exception:
            imerg_missing = bool(imerg_missing_flag)

    if imerg_missing or (np.isnan(max1h) and np.isnan(sum7)):
        warnings.append("imerg_data_unavailable")
    else:
        if any(t in flood_type_lower for t in ["flash", "urban", "pluvial"]):
            if max1h == 0:
                warnings.append("zero_precipitation_flash_flood")
            elif not np.isnan(max1h) and max1h < min_flash:
                warnings.append(f"low_intensity_flash_flood_{max1h:.1f}mm_hr")
        elif any(t in flood_type_lower for t in ["riverine", "river"]):
            if sum7 == 0:
                warnings.append("zero_precipitation_riverine_flood")
            elif not np.isnan(sum7) and sum7 < min_riv7d:
                warnings.append(f"low_precip_riverine_flood_{sum7:.1f}mm_7d")
        elif any(t in flood_type_lower for t in ["snowmelt", "coastal", "storm surge", "dam", "levee"]):
            pass  # These floods may legitimately have zero immediate precipitation
        else:
            if max1h == 0 or sum7 == 0:
                warnings.append(f"zero_precipitation_{flood_type_lower or 'unknown'}")

    # Check 3 - SMAP availability
    if features.get("gee_smap_surface_soil_moisture") in (None, np.nan):
        warnings.append("smap_surface_soil_moisture_missing")

    # Check 4 - GFS availability
    if (
        features.get("gee_gfs_temp2m_celsius") in (None, np.nan)
        and features.get("gee_gfs_relative_humidity_pct") in (None, np.nan)
    ):
        warnings.append("gfs_atmospheric_missing")

    # Check 5 - Temporal validation
    try:
        if pd.notna(event_date):
            event_dt = pd.to_datetime(event_date)
            now = pd.Timestamp.now()
            if event_dt > now:
                issues.append(f"future_event_date_{event_dt.date()}")
            if event_dt < pd.Timestamp("2000-06-01"):
                warnings.append(f"pre_imerg_era_{event_dt.date()}")
    except Exception:
        pass

    # Check 6 - Spatial validation
    try:
        if not (-90 <= float(lat) <= 90):
            issues.append(f"invalid_latitude_{lat}")
        if not (-180 <= float(lon) <= 180):
            issues.append(f"invalid_longitude_{lon}")
    except Exception:
        pass

    # Check 7 - Range validation
    soil_range = globals().get("SOIL_MOISTURE_RANGE", (0.0, 1.0))
    temp_range = globals().get("TEMPERATURE_RANGE_C", (-50.0, 60.0))
    ssm = features.get("gee_smap_surface_soil_moisture")
    if ssm is not None and not (isinstance(ssm, float) and np.isnan(ssm)):
        try:
            ssmf = float(ssm)
            if not (soil_range[0] <= ssmf <= soil_range[1]):
                warnings.append(f"soil_moisture_out_of_range_{ssmf:.3f}")
        except Exception:
            pass
    t2m = features.get("gee_gfs_temp2m_celsius")
    if t2m is not None and not (isinstance(t2m, float) and np.isnan(t2m)):
        try:
            t2 = float(t2m)
            if not (temp_range[0] <= t2 <= temp_range[1]):
                warnings.append(f"temperature_out_of_range_{t2:.1f}C")
        except Exception:
            pass
    for k, v in features.items():
        if "imerg" in k and isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) and v < 0:
            warnings.append(f"negative_precip_value_{k}_{v}")

    return issues, warnings

def validate_and_clean_features(features: dict) -> dict:
    """Validate feature ranges and mark suspicious values as NaN."""
    validation_rules = {
        # Precipitation-related (non-negative)
        'imerg': (0, 10000),  # mm or mm/day proxies
        'gee_elevation_mean': (-500, 9000),  # meters
        'gee_slope_mean': (0, 90),  # degrees
        'gee_aspect_mean': (0, 360),
        'gee_gsw_occurrence_mean': (0, 100),
        'month_sin': (-1, 1),
        'month_cos': (-1, 1),
        'doy_sin': (-1, 1),
        'doy_cos': (-1, 1),
        'gee_distance_to_river_min_m': (0, 1e6),  # meters
        'gee_merit_upa_mean': (0, 1e10),  # upstream area in km²
        'gee_gfs_temp2m_celsius': (-50, 60),
        'gee_gfs_relative_humidity_pct': (0, 100),
        'gee_gfs_wind_speed_ms': (0, 100),
        'gee_gfs_surface_pressure_pa': (3e4, 1.1e5),
        'gee_gfs_precipitable_water_kg_m2': (0, 200),
    }

    cleaned = features.copy()

    for key, value in features.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        try:
            val = float(value)
            # Check each rule
            for pattern, (min_val, max_val) in validation_rules.items():
                if pattern in key.lower():
                    if val < min_val or val > max_val:
                        logging.warning(f"Invalid value for {key}: {val} (expected {min_val} to {max_val}). Setting to NaN.")
                        cleaned[key] = np.nan
                        break
        except Exception:
            continue

    return cleaned


def validate_and_fix_derived_features(features: dict, attempt: int = 1) -> dict:
    """
    Validate derived features using LOGICAL constraints only.

    Constraint types:
    - BOUNDED: Must be within exact mathematical range (e.g., sin/cos in [-1,1])
    - NON_NEGATIVE: Must be >= 0 (e.g., precipitation, distances)
    - FRACTION: Must be in [0, 1] (e.g., soil moisture volume fraction)
    - PERCENTAGE: Must be in [0, 100]
    - ANGLE_DEG: Must be in [0, 360] degrees
    - SLOPE_DEG: Must be in [0, 90] degrees
    - TEMPERATURE_C: Physically plausible Earth surface temps [-90, 60]
    - UNBOUNDED_POSITIVE: Any positive value allowed (e.g., upstream area)

    Args:
        features: Dictionary of extracted features
        attempt: Current attempt number (for logging)

    Returns:
        Dictionary with validated/fixed features
    """
    fixed = features.copy()

    # Define LOGICAL constraints (not arbitrary caps!)
    validation_rules = {
        # BOUNDED: Exact mathematical limits
        'month_sin': ('bounded', -1.0, 1.0),
        'month_cos': ('bounded', -1.0, 1.0),
        'doy_sin': ('bounded', -1.0, 1.0),
        'doy_cos': ('bounded', -1.0, 1.0),
        'gee_ndvi_mean_30d': ('bounded', -1.0, 1.0),
        'gee_ndbi_mean_30d': ('bounded', -1.0, 1.0),
        'gee_ndwi_mean_30d': ('bounded', -1.0, 1.0),

        # FRACTION: Volume fractions, proportions
        'gee_smap_surface_soil_moisture': ('fraction', 0.0, 1.0),
        'gee_smap_subsurface_soil_moisture': ('fraction', 0.0, 1.0),
        'gee_impervious_fraction': ('fraction', 0.0, 1.0),
        'land_fraction_saturated': ('fraction', 0.0, 1.0),
        'land_fraction_unsaturated': ('fraction', 0.0, 1.0),
        'land_fraction_wilting': ('fraction', 0.0, 1.0),
        'land_fraction_snow_covered': ('fraction', 0.0, 1.0),

        # PERCENTAGE: 0-100 scale
        'gee_gfs_relative_humidity_pct': ('percentage', 0.0, 100.0),
        'gee_gsw_occurrence_mean': ('percentage', 0.0, 100.0),

        # ANGLE: Degrees [0, 360)
        'gee_aspect_mean': ('angle_deg', 0.0, 360.0),

        # SLOPE: Degrees [0, 90]
        'gee_slope_mean': ('slope_deg', 0.0, 90.0),

        # TEMPERATURE: Physically plausible Earth surface
        'gee_gfs_temp2m_celsius': ('temperature_c', -90.0, 60.0),
        'surface_temp': ('temperature_k', 180.0, 350.0),

        # NON_NEGATIVE: Precipitation (no upper limit - global dataset!)
        'gee_imerg_sum_24h_mm': ('non_negative', 0.0, None),
        'gee_imerg_max_1h_intensity_mm': ('non_negative', 0.0, None),
        'gee_imerg_max_3h_intensity_mm': ('non_negative', 0.0, None),
        'gee_imerg_max_6h_intensity_mm': ('non_negative', 0.0, None),
        'gee_imerg_sum_3d_before_mm': ('non_negative', 0.0, None),
        'gee_imerg_sum_7d_before_mm': ('non_negative', 0.0, None),
        'gee_imerg_max_daily_7d_mm': ('non_negative', 0.0, None),
        'imerg_max_1h_intensity_mm': ('non_negative', 0.0, None),
        'imerg_mean_1h_intensity_mm': ('non_negative', 0.0, None),
        'gee_imerg_intensity_3d_mm_per_day': ('non_negative', 0.0, None),
        'gee_imerg_intensity_7d_mm_per_day': ('non_negative', 0.0, None),

        # NON_NEGATIVE: Distances, areas, pressures
        'gee_distance_to_river_min_m': ('non_negative', 0.0, None),
        'gee_elevation_mean': ('allow_negative', -500.0, None),  # Below sea level OK

        # UNBOUNDED_POSITIVE: Can be arbitrarily large (global scale)
        'gee_merit_upa_mean': ('unbounded_positive', 0.0, None),  # Upstream area
        'gee_gfs_surface_pressure_pa': ('pressure_pa', 30000.0, 110000.0),  # Physical atmosphere
        'gee_gfs_precipitable_water_kg_m2': ('non_negative', 0.0, None),
        'gee_gfs_wind_speed_ms': ('non_negative', 0.0, None),

        # ENGINEERED: Non-negative, no caps (depend on input scales)
        'gee_flashiness_index': ('non_negative', 0.0, None),
        'gee_saturation_proxy': ('non_negative', 0.0, None),
        'gee_runoff_potential': ('non_negative', 0.0, None),
        'gee_api_weighted_mm': ('non_negative', 0.0, None),
        'gee_flashiness_index_7d': ('non_negative', 0.0, None),
        'gee_precip_x_slope': ('non_negative', 0.0, None),
        'gee_antecedent_moisture_proxy': ('non_negative', 0.0, None),
        'gee_moisture_flux': ('allow_any', None, None),  # Can be any sign
        'gee_twi': ('allow_any', None, None),  # Topographic wetness index

        # ANOMALIES: Can be positive or negative
        'gee_smap_soil_moisture_anomaly': ('allow_any', None, None),
    }

    issues_found = []

    for feature_name, rule in validation_rules.items():
        if feature_name not in fixed:
            continue

        constraint_type = rule[0]
        min_val = rule[1]
        max_val = rule[2] if len(rule) > 2 else None

        value = fixed[feature_name]

        # Skip NaN/None (those are valid "missing data")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue

        try:
            val = float(value)
            should_fix = False
            fixed_val = np.nan

            # Apply constraint based on type
            if constraint_type == 'bounded':
                # STRICT: Must be in [min, max] - clamp tiny FP errors
                if abs(val) < 1e-10:  # Essentially zero
                    fixed_val = 0.0
                    should_fix = True
                elif val < min_val:
                    if val > min_val - 0.01:  # FP error near boundary
                        fixed_val = min_val
                        should_fix = True
                    else:  # Serious violation
                        fixed_val = np.nan
                        should_fix = True
                        issues_found.append(f"{feature_name}={val:.6f}<{min_val}")
                elif val > max_val:
                    if val < max_val + 0.01:  # FP error near boundary
                        fixed_val = max_val
                        should_fix = True
                    else:  # Serious violation
                        fixed_val = np.nan
                        should_fix = True
                        issues_found.append(f"{feature_name}={val:.6f}>{max_val}")

            elif constraint_type in ['fraction', 'percentage', 'angle_deg', 'slope_deg', 'temperature_c', 'temperature_k', 'pressure_pa']:
                # STRICT: Must be in [min, max]
                if val < min_val or val > max_val:
                    fixed_val = np.nan
                    should_fix = True
                    issues_found.append(f"{feature_name}={val:.2f} outside [{min_val},{max_val}]")

            elif constraint_type == 'non_negative':
                # Must be >= 0, no upper limit
                if val < 0:
                    if val > -1e-10:  # Tiny FP error
                        fixed_val = 0.0
                        should_fix = True
                    else:  # Actual negative value
                        fixed_val = np.nan
                        should_fix = True
                        issues_found.append(f"{feature_name}={val:.6f}<0")

            elif constraint_type == 'unbounded_positive':
                # Must be >= 0, can be arbitrarily large
                if val < 0:
                    fixed_val = np.nan
                    should_fix = True
                    issues_found.append(f"{feature_name}={val:.6f}<0")

            elif constraint_type == 'allow_negative':
                # Can be negative (e.g., elevation below sea level), but has lower bound
                if val < min_val:
                    fixed_val = np.nan
                    should_fix = True
                    issues_found.append(f"{feature_name}={val:.2f}<{min_val}")

            elif constraint_type == 'allow_any':
                # No constraints (e.g., anomalies, indices)
                pass

            # Apply fix if needed
            if should_fix:
                if attempt == 1 and not np.isnan(fixed_val):
                    logging.debug(f"Clamped {feature_name} from {val:.10f} to {fixed_val}")
                fixed[feature_name] = fixed_val

        except (ValueError, TypeError) as e:
            logging.debug(f"Could not validate {feature_name}: {e}")
            continue

    if issues_found and attempt == 1:
        logging.warning(
            f"Fixed {len(issues_found)} features with invalid values. "
            f"Examples: {'; '.join(issues_found[:3])}"
        )

    return fixed

def create_validation_summary(df):
    """Create validation summary statistics."""
    if df.empty:
        return {}
    
    summary = {
        'total_events': len(df),
        'valid_events': (df.get('data_quality_valid', True)).sum() if 'data_quality_valid' in df.columns else len(df),
        'invalid_events': len(df) - ((df.get('data_quality_valid', True)).sum() if 'data_quality_valid' in df.columns else 0),
    }
    
    # Count issue types
    if 'data_quality_issues' in df.columns:
        all_issues = []
        for issues_str in df['data_quality_issues'].dropna():
            if issues_str:
                all_issues.extend(issues_str.split('|'))
        
        issue_counts = pd.Series(all_issues).value_counts()
        summary['common_issues'] = issue_counts.head(10).to_dict()
    
    # Flood type breakdown
    if 'flood_type' in df.columns:
        summary['flood_type_counts'] = df['flood_type'].value_counts().to_dict()
    
    if 'imerg_data_missing' in df.columns:
        try:
            missing_pct = pd.to_numeric(df['imerg_data_missing'], errors='coerce').fillna(0).mean() * 100
            logging.info(f"IMERG data missing for {missing_pct:.1f}% of events")
        except Exception as e:
            logging.debug(f"Failed to compute IMERG missing percentage: {e}")

    return summary

# ---------------- Enhanced negative generation ----------------

def generate_negatives_time_offset(positives_gdf, ratio=NEGATIVE_RATIO, 
                                 offset_range=NEG_OFFSET_RANGE_DAYS):
    """Generate time-offset negatives per positive at fixed offsets.

    Offsets (days before flood_start_date): 1, 3, 7, 15, 30.
    - Same lat/lon as parent
    - target=0
    - flood_start_date set to offset date
    - Preserve other attributes (e.g., flood_type)
    - event_id: neg_{original_event_id}_d{offset}
    - Skip any negative whose offset date is before 1980-01-01
    """
    if len(positives_gdf) == 0:
        return gpd.GeoDataFrame(columns=list(positives_gdf.columns), crs=positives_gdf.crs)

    FIXED_OFFSETS = [1, 3, 7, 15, 30]
    MIN_DATE = pd.Timestamp("1980-01-01")

    rows = []
    anchors = positives_gdf.copy()

    for idx, row in anchors.iterrows():
        flood_date = row.get("flood_start_date")
        if pd.isna(flood_date):
            continue

        base_date = pd.to_datetime(flood_date)
        orig_event_id = row.get("event_id")
        if pd.isna(orig_event_id) or orig_event_id in (None, ""):
            orig_event_id = str(idx)

        for off in FIXED_OFFSETS:
            try:
                neg_date = base_date - pd.Timedelta(days=int(off))
                if pd.isna(neg_date) or neg_date < MIN_DATE:
                    continue  # Skip invalid/too-early dates

                neg_row = row.drop(labels=["geometry"], errors='ignore').to_dict()
                neg_row["target"] = 0
                # --- Soft label & near-miss flags ---
                try:
                    off_int = int(off)
                except Exception:
                    off_int = None
                if off_int == 1:
                    neg_row["soft_label"] = 0.9
                    neg_row["near_miss"] = 1
                elif off_int == 3:
                    neg_row["soft_label"] = 0.3
                    neg_row["near_miss"] = 1
                else:
                    neg_row["soft_label"] = 0.0
                    neg_row["near_miss"] = 0
                # ------------------------------------
                neg_row["flood_start_date"] = neg_date
                try:
                    neg_row["year"] = int(neg_date.year)
                except Exception:
                    pass
                neg_row["event_id"] = f"neg_{orig_event_id}_d{int(off)}"

                rows.append(neg_row)
            except Exception as e:
                logging.debug(f"Failed to create negative (offset {off}) for row {idx}: {e}")
                continue

    if not rows:
        return gpd.GeoDataFrame(columns=list(positives_gdf.columns), crs="EPSG:4326")

    df_neg = pd.DataFrame(rows)
    gdf_neg = gpd.GeoDataFrame(
        df_neg,
        geometry=gpd.points_from_xy(df_neg["longitude"], df_neg["latitude"]),
        crs="EPSG:4326",
    )

    logging.info(
        f"Generated {len(gdf_neg)} time-offset negatives (fixed offsets: {', '.join(str(o) for o in FIXED_OFFSETS)} days before)"
    )
    return gdf_neg.reset_index(drop=True)

# ---------------- Spatial negative generation (Natural Earth) ----------------

def load_naturalearth_countries_via_zip_https():
    """
    Load Natural Earth countries either via zip+https VSICURL or by downloading
    and extracting to a temp directory. Returns GeoDataFrame in EPSG:4326.
    """
    url_zip = "zip+https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    try:
        world = gpd.read_file(url_zip)
        logging.info("[NE] Loaded Natural Earth via zip+https")
        return world.to_crs("EPSG:4326")
    except Exception as e:
        logging.warning(f"[NE] zip+https read failed (falling back to download): {e}")
    try:
        import requests, zipfile, io, tempfile
        r = requests.get("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip", stream=True, timeout=60)
        r.raise_for_status()
        tmpdir = tempfile.mkdtemp(prefix="ne_")
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        zf.extractall(tmpdir)
        shp = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")), None)
        if shp is None:
            raise RuntimeError("No .shp found in Natural Earth zip")
        world = gpd.read_file(shp)
        logging.info("[NE] Loaded Natural Earth via download fallback")
        return world.to_crs("EPSG:4326")
    except Exception as e:
        logging.error(f"[NE] Could not fetch Natural Earth countries: {e}")
        raise

def generate_negatives_spatial(positives_gdf, ratio=NEGATIVE_RATIO, exclusion_m=NEG_EXCLUSION_RADIUS_METERS):
    """
    Generate spatial negatives:
      - Sample area-weighted inside Natural Earth country polygons (prefer countries present in EMDAT)
      - Exclude any point within exclusion_m meters of any positive
      - Return GeoDataFrame with columns latitude, longitude, flood_start_date=NaT, target=0
    """
    num_required = int(len(positives_gdf) * ratio)
    if num_required <= 0:
        return gpd.GeoDataFrame(columns=["geometry","latitude","longitude","flood_start_date","target"], crs="EPSG:4326")

    world = load_naturalearth_countries_via_zip_https()

    # Restrict candidate polygons to the set of countries in positives (if available)
    candidate_polys = world
    if "country_code" in positives_gdf.columns:
        codes = positives_gdf["country_code"].dropna().astype(str).str.upper().unique().tolist()
        if len(codes) > 0:
            matched = None
            for col in ["ISO_A2","ISO_A3","iso_a2","iso_a3","ADM0_A3","adm0_a3","gu_a3","iso_a2"]:
                if col in world.columns:
                    try:
                        mask = world[col].astype(str).str.strip().str.upper().isin(codes)
                        if mask.any():
                            matched = world[mask]
                            break
                    except Exception:
                        continue
            if matched is not None and len(matched) > 0:
                candidate_polys = matched
                logging.info("[NEG] Sampling negatives in these countries: %s", ", ".join(map(str, codes)))
            else:
                logging.info("[NEG] No Natural Earth country matched the EMDAT country codes; sampling globally.")

    # Compute area weights in projected CRS (EPSG:3857) for sampling
    try:
        candidate_proj = candidate_polys.to_crs("EPSG:3857")
        areas = candidate_proj.geometry.area.values
        areas_sum = areas.sum()
        if areas_sum <= 0:
            weights = None
        else:
            weights = areas / areas_sum
    except Exception:
        weights = None

    # Build exclusion union (project positives to 3857 and buffer)
    try:
        pos_proj = positives_gdf.to_crs("EPSG:3857")
        buffers = [geom.buffer(exclusion_m) for geom in pos_proj.geometry]
        exclusion_union = unary_union(buffers) if len(buffers) > 0 else None
    except Exception:
        exclusion_union = None

    chosen_points = []
    max_attempts = max(10000, num_required * 500)
    attempts = 0

    polys_list = list(candidate_polys.geometry.values)
    if len(polys_list) == 0:
        raise RuntimeError("No candidate polygons found in Natural Earth to sample negatives.")

    while len(chosen_points) < num_required and attempts < max_attempts:
        attempts += 1
        # select polygon index (weighted if possible)
        if weights is not None:
            try:
                idx = np.random.choice(len(polys_list), p=weights)
            except Exception:
                idx = random.randrange(len(polys_list))
        else:
            idx = random.randrange(len(polys_list))
        poly = polys_list[idx]
        minx, miny, maxx, maxy = poly.bounds
        if minx == maxx or miny == maxy:
            continue
        # sample random point in bbox, reject if outside polygon
        rx = random.uniform(minx, maxx)
        ry = random.uniform(miny, maxy)
        p = Point(rx, ry)
        if not poly.contains(p):
            continue
        # project and test exclusion buffer
        try:
            p_proj = transform(pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, p)
            if exclusion_union is not None and exclusion_union.contains(p_proj):
                continue
        except Exception:
            pass
        chosen_points.append(p)

    # Fallback global sampling if not enough points
    if len(chosen_points) < num_required:
        logging.warning("[NEG] Could only sample %d/%d required negatives after %d attempts; trying global fallback", len(chosen_points), num_required, attempts)
        world_polys = list(world.geometry.values)
        attempts2 = 0
        while len(chosen_points) < num_required and attempts2 < max_attempts:
            attempts2 += 1
            poly = random.choice(world_polys)
            minx, miny, maxx, maxy = poly.bounds
            rx = random.uniform(minx, maxx); ry = random.uniform(miny, maxy)
            p = Point(rx, ry)
            if not poly.contains(p):
                continue
            try:
                p_proj = transform(pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform, p)
                if exclusion_union is not None and exclusion_union.contains(p_proj):
                    continue
            except Exception:
                pass
            chosen_points.append(p)

    chosen_points = chosen_points[:num_required]
    neg_gdf = gpd.GeoDataFrame({"geometry": chosen_points}, crs="EPSG:4326")
    neg_gdf["latitude"] = neg_gdf.geometry.y
    neg_gdf["longitude"] = neg_gdf.geometry.x
    neg_gdf["flood_start_date"] = pd.NaT
    neg_gdf["target"] = 0
    logging.info("[NEG] Generated %d negative samples (requested %d).", len(neg_gdf), num_required)
    return neg_gdf.reset_index(drop=True)

# ---------------- Positive event augmentation (duration-based) ----------------

def augment_positive_events(positives_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Augment positive events based on flood duration.

    Mapping (total positives produced per event):
      - 1-2 days  -> 1 event (original only)
      - 3-4 days  -> 2 events (original + day 2)
      - 5-8 days  -> 3 events (original + days 2,3)
      - 9-12 days -> 4 events (original + days 2,3,4)
      - 13+ days  -> 5 events (original + days 2,3,4,5)

    Rules:
      - Only duplicates positives (target=1)
      - Additional events share same lat/lon and attributes
      - flood_start_date = original + offset_days
      - If duration missing/NaN, treat as 1-day (no augmentation)
      - No special ID handling; event_id is preserved (duplicates allowed)
    """
    if positives_gdf is None or len(positives_gdf) == 0:
        return positives_gdf

    def offsets_for_duration(dur_val) -> list:
        try:
            if pd.isna(dur_val):
                return []
            d = int(dur_val)
        except Exception:
            return []
        if d <= 2:
            return []
        if d <= 4:
            return [1]
        if d <= 8:
            return [1, 2]
        if d <= 12:
            return [1, 2, 3]
        return [1, 2, 3, 4]

    extras = []
    for idx, row in positives_gdf.iterrows():
        # Only augment positives (target == 1)
        try:
            if int(row.get("target", 1)) != 1:
                continue
        except Exception:
            # Assume positive if target missing
            pass

        base_date = row.get("flood_start_date")
        if pd.isna(base_date):
            continue
        base_date = pd.to_datetime(base_date)

        offs = offsets_for_duration(row.get("flood_duration_days", np.nan))
        if not offs:
            continue

        for off in offs:
            try:
                new_date = base_date + pd.Timedelta(days=int(off))
                dup = row.drop(labels=["geometry"], errors="ignore").to_dict()
                dup["flood_start_date"] = new_date
                dup["target"] = 1
                # Optionally set year now; pipeline will recompute anyway
                try:
                    dup["year"] = int(new_date.year)
                except Exception:
                    pass
                extras.append(dup)
            except Exception as e:
                logging.debug(f"Augment failed for row {idx} offset {off}: {e}")
                continue

    if not extras:
        logging.info("[AUG] No positives augmented by duration rules")
        return positives_gdf

    df_extras = pd.DataFrame(extras)
    gdf_extras = gpd.GeoDataFrame(
        df_extras,
        geometry=gpd.points_from_xy(df_extras["longitude"], df_extras["latitude"]),
        crs=positives_gdf.crs or "EPSG:4326",
    )

    out = pd.concat([positives_gdf, gdf_extras], ignore_index=True).reset_index(drop=True)
    logging.info("[AUG] Added %d augmented positives; new positive count: %d", len(gdf_extras), len(out))
    return out

# ---------------- Main processing pipeline ----------------

def process_monthly_batch(month_events: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Process all events for a single month."""

    enriched_rows = []
    pre_imerg_count = 0
    pre_smap_count = 0

    total_events = len(month_events)
    if total_events == 0:
        logging.info(f"No events to process for {year}-{month:02d}")
        return pd.DataFrame()

    event_contexts = []
    futures = []
    future_to_index = {}

    max_workers = max(1, MAX_GEE_WORKERS)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for event_idx, (row_idx, row) in enumerate(month_events.iterrows()):
                lat = row["latitude"]
                lon = row["longitude"]
                event_date = row["flood_start_date"]
                flood_type = row.get("flood_type", "Unknown")
                target = row.get("target", 1)

                event_dt = pd.to_datetime(event_date) if pd.notna(event_date) else pd.NaT

                base_record = row.drop(labels=["geometry"], errors='ignore').to_dict()
                try:
                    if pd.notna(event_dt):
                        m = int(event_dt.month)
                        doy = int(event_dt.dayofyear)
                        base_record["month"] = m
                        base_record["day_of_year"] = doy
                        # Round to limit floating point drift (e.g., -2.45e-16)
                        # Round to 10 decimals to avoid FP errors like -2.45e-16
                        base_record["month_sin"] = np.round(np.sin(2 * np.pi * (m / 12.0)), 10)
                        base_record["month_cos"] = np.round(np.cos(2 * np.pi * (m / 12.0)), 10)
                        base_record["doy_sin"] = np.round(np.sin(2 * np.pi * (doy / 365.0)), 10)
                        base_record["doy_cos"] = np.round(np.cos(2 * np.pi * (doy / 365.0)), 10)
                    else:
                        base_record["month"] = -1
                        base_record["day_of_year"] = -1
                        base_record["month_sin"] = np.nan
                        base_record["month_cos"] = np.nan
                        base_record["doy_sin"] = np.nan
                        base_record["doy_cos"] = np.nan
                except Exception:
                    pass

                if pd.notna(event_dt):
                    if event_dt < pd.Timestamp("2000-06-01"):
                        pre_imerg_count += 1
                        logging.info(
                            f"Event {event_idx + 1} ({event_dt.date()}) is pre-IMERG era - limited features available"
                        )
                    elif event_dt < pd.Timestamp("2015-03-31"):
                        pre_smap_count += 1
                        logging.info(
                            f"Event {event_idx + 1} ({event_dt.date()}) is pre-SMAP era - no soil moisture data"
                        )

                context = {
                    "base_record": base_record,
                    "lat": lat,
                    "lon": lon,
                    "event_dt": event_dt,
                    "flood_type": flood_type,
                    "target": target,
                    "event_number": event_idx + 1,
                    "row_index": row_idx,
                }
                event_contexts.append(context)

                if pd.notna(event_dt):
                    future = executor.submit(
                        extract_gee_features,
                        lat,
                        lon,
                        event_dt,
                        flood_type,
                    )
                    futures.append(future)
                    future_to_index[future] = event_idx
                else:
                    futures.append(None)
    except KeyboardInterrupt:
        logging.warning("GEE extraction interrupted")
        raise

    gee_results = [{} for _ in event_contexts]

    try:
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                if isinstance(result, dict):
                    gee_results[idx] = result
                else:
                    logging.debug(f"Unexpected GEE result type for event {idx + 1}: {type(result)}")
                    gee_results[idx] = {}
            except KeyboardInterrupt:
                logging.warning("Parallel GEE extraction interrupted")
                raise
            except Exception as e:
                logging.warning(f"GEE extraction failed: {e}")
                gee_results[idx] = {}
    except KeyboardInterrupt:
        raise

    for idx, context in enumerate(event_contexts):
        gee_features = gee_results[idx] if isinstance(gee_results[idx], dict) else {}
        if not gee_features:
            if pd.isna(context["event_dt"]):
                logging.warning("Skipping GEE extraction for NaT event date")

        all_features = {**context["base_record"], **gee_features}

        # First pass: Fix derived features with logical constraints
        try:
            all_features = validate_and_fix_derived_features(all_features, attempt=1)
        except Exception as e:
            logging.debug(f"Derived feature validation failed: {e}")

        # Second pass: Legacy range validation (can be removed if above covers all)
        try:
            all_features = validate_and_clean_features(all_features)
        except Exception as e:
            logging.debug(f"Range validation failed: {e}")

        # Add metadata AFTER validation
        all_features["data_extraction_timestamp"] = pd.Timestamp.now()
        all_features["pipeline_version"] = "v3.0_temporal_fix"

        # Validation
        if ENABLE_VALIDATION:
            try:
                issues, warnings = validate_flood_features(
                    all_features,
                    context["flood_type"],
                    context["event_dt"],
                    context["lat"],
                    context["lon"],
                )
                all_features["data_quality_issues"] = "|".join(issues) if issues else ""
                all_features["data_quality_warnings"] = "|".join(warnings) if warnings else ""
                all_features["data_quality_valid"] = len(issues) == 0
                if issues:
                    logging.warning(f"Event {context['event_number']} validation issues: {issues}")
            except Exception as e:
                logging.debug(f"Validation failed for event {context['event_number']}: {e}")
                all_features["data_quality_issues"] = "validation_failed"
                all_features["data_quality_warnings"] = ""
                all_features["data_quality_valid"] = False

        enriched_rows.append(all_features)

        if context["event_number"] % 10 == 0:
            logging.info(
                f"Processed {context['event_number']}/{total_events} events in {year}-{month:02d}"
            )

    df_enriched = pd.DataFrame(enriched_rows)

    logging.info(
        f"Pre-check summary for {year}-{month:02d}: {pre_imerg_count} events before IMERG coverage, {pre_smap_count} events before SMAP v008"
    )

    # Post-processing (column cleanup)
    if not df_enriched.empty and DROP_ZERO_NAN_COLUMNS:
        original_cols = len(df_enriched.columns)
        cols_to_drop = []
        protected_cols = {
            "latitude", "longitude", "flood_start_date", "target", "year",
            "flood_type", "confidence_score", "data_quality_issues", 
            "data_quality_warnings", "data_quality_valid",
            "gee_river_name",
            "river_name", "river_name_source",
        }
        protected_suffixes = ("_reduction_method",)
        for col in df_enriched.columns:
            if col in protected_cols or any(col.endswith(suf) for suf in protected_suffixes):
                continue
            try:
                numeric_series = pd.to_numeric(df_enriched[col], errors='coerce')
                if numeric_series.notna().sum() == 0:
                    cols_to_drop.append(col)
                    continue
                non_nan_values = numeric_series.dropna()
                if len(non_nan_values) > 0 and (non_nan_values.abs() < 1e-12).all():
                    cols_to_drop.append(col)
                    continue
                total_rows = len(df_enriched)
                if total_rows > 0:
                    nan_fraction = numeric_series.isna().sum() / total_rows
                    zero_fraction = (numeric_series.fillna(0).abs() < 1e-12).sum() / total_rows
                    if (nan_fraction > NAN_FRACTION_THRESHOLD or 
                        zero_fraction > ZERO_FRACTION_THRESHOLD):
                        cols_to_drop.append(col)
            except Exception:
                continue
        if cols_to_drop:
            df_enriched = df_enriched.drop(columns=cols_to_drop)
            logging.info(
                f"Dropped {len(cols_to_drop)} sparse/zero columns for {year}-{month:02d}. Remaining: {len(df_enriched.columns)} (was {original_cols})"
            )

    # Add metadata
    if not df_enriched.empty:
        df_enriched["processing_date"] = pd.Timestamp.now()
        df_enriched["processing_version"] = "v3.0_imerg_smap_gfs"
        df_enriched["data_sources"] = "IMERG_Early|IMERG_Late|SMAP|GFS|Static"
        df_enriched["max_latency_hours"] = 14

    return df_enriched

def enrich_and_write(gdf, continue_from_month=1):
    """Process events in monthly batches with optional continuation."""
    logging.info(f"Starting enrichment for {len(gdf)} positive events")

    # Generate negatives
    if NEGATIVE_STRATEGY == "time_offset":
        negs = generate_negatives_time_offset(gdf, ratio=NEGATIVE_RATIO, offset_range=NEG_OFFSET_RANGE_DAYS)
    else:
        try:
            negs = generate_negatives_spatial(gdf, ratio=NEGATIVE_RATIO, exclusion_m=NEG_EXCLUSION_RADIUS_METERS)
        except Exception as e:
            logging.warning(f"Spatial negative generation failed: {e}")
            negs = gpd.GeoDataFrame(columns=list(gdf.columns), crs=gdf.crs)

    # Augment positives AFTER negatives
    try:
        gdf_augmented = augment_positive_events(gdf)
    except Exception as e:
        logging.warning(f"Positive augmentation failed, proceeding without: {e}")
        gdf_augmented = gdf

    # Combine all events
    full_gdf = pd.concat([gdf_augmented, negs], ignore_index=True).reset_index(drop=True)
    full_gdf["year"] = full_gdf["flood_start_date"].dt.year.fillna(-1).astype(int)
    full_gdf["month"] = full_gdf["flood_start_date"].dt.month.fillna(-1).astype(int)

    if MAX_EVENTS_DEV:
        full_gdf = full_gdf.head(MAX_EVENTS_DEV)
        logging.info(f"Limited to {MAX_EVENTS_DEV} events for development")

    # Group by year-month
    full_gdf['year_month'] = full_gdf['flood_start_date'].dt.to_period('M')
    grouped = list(full_gdf.groupby('year_month', dropna=False))

    all_enriched = []
    if continue_from_month > 1:
        logging.info(f"Continuation enabled. Starting from month {continue_from_month}.")
        # Determine primary processing year: the minimum year that actually has the continue_from_month
        try:
            valid_periods = [p for (p, _) in grouped if not pd.isna(p)]
            exact_month_periods = [p for p in valid_periods if int(p.month) == continue_from_month]
            if exact_month_periods:
                primary_year = int(min(exact_month_periods).year)
            else:
                # Fallback: next year after the absolute minimum year
                min_year_overall = int(min(valid_periods).year) if valid_periods else None
                primary_year = (min_year_overall + 1) if min_year_overall is not None else None
        except Exception:
            primary_year = None

        # Load existing files only for the primary year months < continue_from_month
        if primary_year is not None:
            for m in range(1, continue_from_month):
                monthly_file = os.path.join(OUT_DIR, f"enriched_{primary_year}-{m:02d}.csv")
                if os.path.exists(monthly_file):
                    try:
                        existing_df = pd.read_csv(monthly_file)
                        all_enriched.append(existing_df)
                        logging.info(f"Loaded existing: {monthly_file}")
                    except Exception as e:
                        logging.warning(f"Failed to load {monthly_file}: {e}")

        # Filter groups:
        # - Skip any period with year < primary_year (e.g., previous December)
        # - For primary_year, keep only months >= continue_from_month
        # - For later years, keep all months
        monthly_groups = []
        for (p, ev) in grouped:
            if pd.isna(p):
                continue
            y = int(p.year)
            m = int(p.month)
            if primary_year is not None:
                if y < primary_year:
                    continue
                if y == primary_year and m < continue_from_month:
                    continue
            monthly_groups.append((p, ev))

        logging.info(f"Processing {len(monthly_groups)} monthly batches (from month {continue_from_month})")
    else:
        monthly_groups = grouped
        logging.info(f"Processing {len(monthly_groups)} monthly batches")

    # Process filtered months
    for period, month_events in monthly_groups:
        if pd.isna(period):
            logging.warning("Skipping events with NaT dates")
            continue
        year = int(period.year)
        month = int(period.month)

        logging.info("\n" + "="*60)
        logging.info(f"Processing {year}-{month:02d}: {len(month_events)} events")
        logging.info("="*60)

        enriched_month = process_monthly_batch(month_events, year, month)
        if enriched_month is not None and not enriched_month.empty:
            monthly_output = os.path.join(OUT_DIR, f"enriched_{year}-{month:02d}.csv")
            os.makedirs(os.path.dirname(monthly_output), exist_ok=True)
            enriched_month.to_csv(monthly_output, index=False)
            logging.info(f"✅ Saved monthly file: {monthly_output}")
            all_enriched.append(enriched_month)

        

    # Combine all monthly files into final output
    if all_enriched:
        df_final = pd.concat(all_enriched, ignore_index=True)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_final.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"✅ Saved combined file: {OUTPUT_FILE}")
    else:
        logging.error("No data to save - enrichment failed")
        raise RuntimeError("Enrichment pipeline produced no output data")

# ------------------- Main execution -------------------

def main():
    """Main execution function with comprehensive error handling and continuation support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Flood Data Enrichment Pipeline v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--continue-from', '-c',
        type=int,
        default=1,
        metavar='MONTH',
        help='Skip months 1 to (MONTH-1) and start processing from MONTH (1-12). Default: 1'
    )
    parser.add_argument(
        '--emdat-path',
        type=str,
        default=EMDAT_PATH,
        help=f'Path to EMDAT file (default: {EMDAT_PATH})'
    )
    args = parser.parse_args()

    # Validate continuation month
    if not 1 <= args.continue_from <= 12:
        logging.error(f"Invalid --continue-from value: {args.continue_from}. Must be 1-12.")
        sys.exit(1)

    try:
        logging.info("="*60)
        logging.info("FLOOD DATA ENRICHMENT PIPELINE v3.0")
        if args.continue_from > 1:
            logging.info(f"CONTINUATION MODE: Starting from month {args.continue_from}")
        logging.info("="*60)
        
        # Log configuration
        logging.info("Configuration:")
        logging.info(f"  EMDAT Path: {args.emdat_path}")
        logging.info(f"  Output Directory: {OUT_DIR}")
        logging.info(f"  AOI Buffer: {AOI_BUFFER_METERS}m (Flash), {WATERSHED_BUFFER_METERS}m (Riverine)")
        logging.info(f"  Validation Enabled: {ENABLE_VALIDATION}")
        logging.info(f"  Negative Strategy: {NEGATIVE_STRATEGY}")
        logging.info(f"  Negative Ratio: {NEGATIVE_RATIO}")
        
        # Check prerequisites
        if not os.path.exists(args.emdat_path):
            raise FileNotFoundError(f"EMDAT file not found: {args.emdat_path}")
        
        # Load and validate EMDAT data
        logging.info("-" * 40)
        logging.info("Loading EMDAT data...")
        gdf = load_emdat(args.emdat_path)
        
        if len(gdf) == 0:
            raise ValueError("No valid events found in EMDAT data")
        
        # Log data summary
        logging.info(f"Loaded {len(gdf)} flood events:")
        if "flood_type" in gdf.columns:
            type_counts = gdf["flood_type"].value_counts()
            for flood_type, count in type_counts.head(10).items():
                logging.info(f"  {flood_type}: {count}")
        
        date_range = gdf["flood_start_date"].dt.date
        logging.info(f"Date range: {date_range.min()} to {date_range.max()}")
        
        # Geographic distribution
        lat_range = f"{gdf['latitude'].min():.2f} to {gdf['latitude'].max():.2f}"
        lon_range = f"{gdf['longitude'].min():.2f} to {gdf['longitude'].max():.2f}"
        logging.info(f"Geographic range: Lat {lat_range}, Lon {lon_range}")
        
        # Feature breakdown logging
        logging.info("\n\U0001F4CA Expected Feature Breakdown:")
        expected_features = {
            'Precipitation (IMERG)': 9,
            'Soil Moisture (SMAP)': 3,
            'Atmospheric (GFS)': 5,
            'Topographic (Static)': 3,
            'Land Cover': 4,
            'Engineered': 6,
            'Temporal': 3,
            'Metadata': 3,
            'Target': 1,
        }
        for category, count in expected_features.items():
            logging.info(f"  • {category}: {count}")
        logging.info(f"  TOTAL: ~{sum(expected_features.values())} features\n")

        logging.info("\U0001F5D1\uFE0F  Removed from v2.0:")
        logging.info("  • All CHIRPS features (~20)")
        logging.info("  • All ERA5-Land features (3)")
        logging.info("  • All GloFAS features (~15)")
        logging.info("  • NDVI/EVI/NDBI/NDWI (delayed)")
        logging.info("  • Lag/rolling features (replaced)")
        logging.info("  • Total removed: ~70 features\n")

        # Start enrichment
        logging.info("-" * 40)
        logging.info("Starting enrichment pipeline...")
        start_time = time.time()
        
        enrich_and_write(gdf, continue_from_month=args.continue_from)
        
        # Final summary
        elapsed_time = time.time() - start_time
        logging.info("-" * 40)
        logging.info(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed_time/60:.1f} minutes")
        logging.info(f"Output saved to: {OUTPUT_FILE}")
        logging.info("="*60)
        
    except KeyboardInterrupt:
        logging.error("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}")
        logging.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()