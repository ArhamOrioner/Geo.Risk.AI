# -----------------------------------------------------------------------------
# GeoRiskAI - Final Configuration File
#
# CRITICAL OVERHAUL (Final Investor Mandate):
# 1. SECURITY: Uses .env for the API key.
# 2. CLIMATE SCIENCE: Uses a 1-year lookback for recent conditions, with a
#    note on using 30-year normals for a true climate baseline.
# 3. SCALABILITY: NUM_PIXELS is set higher for more robust analysis.
# -----------------------------------------------------------------------------

import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Core Project Settings ---
PROJECT_NAME = "GeoRiskAI"
AUTHOR = "Investor-Grade Architecture"
DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUTS_DIR = 'georiskai_outputs'

# --- Pretrained Model Integration ---
# Default locations can be overridden via environment variables to support
# alternative deployment layouts without touching code.
PRETRAINED_MODEL_PATH = os.getenv("PRETRAINED_MODEL_PATH", "final_ensemble.pkl")
PRETRAINED_SUPPORT_PATH = os.getenv(
    "PRETRAINED_SUPPORT_PATH",
    str(Path(__file__).resolve().parent / "1"),
)

# --- Region of Interest (ROI) ---
# Example: A flood-prone region in Uttarakhand, India.
ROI_BOUNDS = [78.65, 30.90, 78.95, 31.20]

# --- Analysis Time Frame ---
# Use a 1-year lookback for recent conditions.
# For a true climate baseline, a 30-year normal (e.g., from ERA5) is recommended.
END_DATE_ANALYSIS = datetime.now()
START_DATE_ANALYSIS = END_DATE_ANALYSIS - timedelta(days=365)

# --- Gemini AI Integration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY not found. Please create a .env file.")

# --- Data & Analysis Parameters ---
# Use a higher number of pixels for more robust, production-like analysis.
# In a real large-scale run, this would be determined dynamically.
NUM_PIXELS_FOR_ANALYSIS = 10000
# Explicit resampling to a common analysis resolution before sampling/combining.
# Set to a value like 100.0 to force all inputs to 100 m resolution.
# When None, native resolutions are preserved and combined at source scales.
COMMON_RESOLUTION_METERS = None

# --- Uncertainty & Hydrology Parameters ---
# Target miscoverage for conformal prediction intervals (e.g., 0.1 => 90% coverage)
CONFORMAL_ALPHA = 0.1

# Decay constant for decaying Antecedent Precipitation Index (typical 0.85–0.98)
API_DECAY_K = 0.9

# --- Backtesting ---
# Use all available events for backtesting (may be slow) instead of sampling
BACKTEST_USE_ALL_EVENTS = True

# Acceptable deviation between target and empirical coverage before recalibration
UQ_COVERAGE_TOLERANCE = 0.05

# --- External Benchmarking ---
# Optional path to FloodCastBench (2025) events CSV for external backtesting.
# Expected to include columns for lat/lon and dates, plus a binary label column.
FLOODCASTBENCH_EVENTS_CSV = 'data/floodcastbench_events.csv'  # Set to real CSV path

# --- Near Real-Time (NRT) data integration (optional) ---
# If enabled, attempt to include additional realtime signals where available.
USE_IMERG_NRT = True
USE_GLOFAS_NRT = True  # Enable GloFAS NRT
USE_VIIRS_WATER_NRT = True  # Enable VIIRS NRT

# --- K-Means Clustering for Risk Zonation ---
# HDBSCAN is now used, which determines clusters automatically.
# These are fallback parameters if we revert to a simpler method.
RISK_CLUSTERS = 'auto'
RISK_CLUSTER_RANGE = range(3, 7)

# --- Spatial Clustering (HDBSCAN) parameters ---
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 10
HDBSCAN_FALLBACK_K = 10

# --- Visualization ---
MAP_SAMPLE_POINTS = 1000

# --- Hydrology Feature Thresholds ---
CHANNEL_FLOWACC_THRESHOLD = 1000.0

# --- Safety thresholds for uncertainty
UNCERTAINTY_HIGH_THRESHOLD = 0.6
MAX_UNCERTAIN_FRACTION = 0.5

# --- Logging Configuration ---
LOG_FILE = os.path.join(OUTPUTS_DIR, 'georiskai_analysis.log')
LOG_LEVEL = 'INFO'

# --- Modeling ---
# Cross-validation folds for threshold/tuning
CV_N_SPLITS = 5

# Use Optuna to tune XGBoost hyperparameters
USE_OPTUNA_TUNING = True
OPTUNA_TRIALS = 25

# Enable optional ordinal severity head (requires labeled y_ordinal)
ENABLE_ORDINAL = True

# Enable optional DL sequence model for precipitation (experimental)
ENABLE_DL_LSTM = True

# --- Safety Gates ---
SAFETY_MODE = True
MIN_COVERAGE_RATIO = 0.8
MIN_POD_FOR_DEPLOY = 0.75
MIN_CSI_FOR_DEPLOY = 0.5

# --- NRT Windows ---
LOOKBACK_VIIRS_DAYS = 7

# --- Action Triggers (probability thresholds mapped to actions) ---
ACTION_TRIGGERS = [
    {"label": "Immediate mitigation", "min_prob": 0.8, "max_uncertainty": 0.2},
    {"label": "Rapid verification", "min_prob": 0.6, "max_uncertainty": 0.4},
    {"label": "Investigate (uncertain)", "min_prob": 0.4, "max_uncertainty": 0.7},
]
