#!/usr/bin/env python3
"""
Refactored Main Pipeline Orchestrator
Uses asyncio for download orchestration with proper resource management.
"""

import os
import sys
import argparse
import logging
import asyncio
import aiofiles
import tempfile
import zipfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta, date
from calendar import monthrange
from typing import List, Dict, Optional, Tuple, Any
from contextlib import ExitStack, asynccontextmanager, contextmanager
from collections import deque

import pandas as pd
import numpy as np
import xarray as xr
from pyproj import Transformer

# Import configuration
from config import config
from async_downloader_refactored import AsyncGloFASDownloader

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# Reduce ecCodes/cfgrib logging noise (in addition to stderr suppression)
os.environ.setdefault("ECCODES_LOG", "ERROR")
os.environ.setdefault("ECMWF_LOGLEVEL", "error")
# Quiet cfgrib/eccodes Python loggers to suppress 'missingValue' warnings
logging.getLogger('cfgrib').setLevel(logging.ERROR)
logging.getLogger('cfgrib.messages').setLevel(logging.ERROR)
logging.getLogger('eccodes').setLevel(logging.ERROR)

@contextmanager
def suppress_stderr():
    import os, sys
    try:
        fd = sys.stderr.fileno()
    except Exception:
        # If no real file descriptor (e.g., redirected), just yield
        yield
        return
    saved = os.dup(fd)
    try:
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        try:
            os.dup2(saved, fd)
        finally:
            os.close(saved)

def _open_cfgrib_with_suppressed_logs(grib_file: Path, backend_kwargs: dict):
    with suppress_stderr():
        return xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs=backend_kwargs)


def _normalize_dashes_series(s: pd.Series) -> pd.Series:
    """Replace common unicode dashes with ASCII hyphen and strip whitespace."""
    if s is None:
        return s
    return (
        s.astype(str)
        .str.strip()
        .str.replace('\u2013', '-', regex=False)  # EN DASH
        .str.replace('\u2014', '-', regex=False)  # EM DASH
        .str.replace('\u2212', '-', regex=False)  # MINUS SIGN
    )


def _parse_dates_robust_series(s: pd.Series) -> pd.Series:
    """Parse dates robustly handling ISO (YYYY-MM-DD), DD-MM-YYYY, and common variants.
    Also strips time components if present and handles Excel month/day variants.
    """
    if s is None or len(s) == 0:
        return pd.to_datetime(pd.Series([], dtype=object), errors='coerce')

    s = _normalize_dashes_series(s)
    # Remove time part if present (e.g., '2020-01-04 00:00:00' or '2020-01-04T00:00:00')
    s_no_time = s.str.replace(r"[T\s].*$", '', regex=True)

    # 1) Exact ISO first
    iso_mask = s_no_time.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    parsed = pd.to_datetime(s_no_time.where(iso_mask), format='%Y-%m-%d', errors='coerce')

    # 2) Remaining with dayfirst=True (handles DD-MM-YYYY, DD/MM/YYYY)
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(s_no_time[remaining], errors='coerce', dayfirst=True)

    # 3) Remaining without dayfirst (handles MM/DD/YYYY, etc.)
    remaining = parsed.isna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(s_no_time[remaining], errors='coerce', dayfirst=False)

    # 4) As a last resort, handle pure numeric Excel serials (e.g., 43831)
    remaining = parsed.isna()
    if remaining.any():
        num_mask = s_no_time[remaining].str.match(r'^\d+(\.\d+)?$', na=False)
        if num_mask.any():
            try:
                serials = s_no_time[remaining][num_mask].astype(float)
                base = pd.Timestamp('1899-12-30')
                parsed.loc[remaining][num_mask] = base + pd.to_timedelta(serials, unit='D')
            except Exception:
                pass

    return parsed


class PipelineOrchestrator:
    """Main pipeline orchestrator with async download management."""
    
    def __init__(self, events_dir: str, output_dir: str):
        """Initialize the orchestrator."""
        self.events_dir = Path(events_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temp directory (download staging)
        if config.download_dir:
            self.temp_dir = Path(config.download_dir)
            self._temp_dir_is_external = True
        else:
            colab_root = Path("/content")
            output_on_drive = str(self.output_dir).startswith("/content/drive/")
            if output_on_drive and colab_root.exists():
                self.temp_dir = colab_root / config.temp_dir_name
            else:
                self.temp_dir = self.output_dir / config.temp_dir_name
            self._temp_dir_is_external = False
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()
        
        # Track processing state
        self.processed_chunks = set()
        self.failed_chunks = set()

        # Shared downloader instance for global rate limiting across prefetch/current.
        # Scale concurrency to number of configured API keys to keep each key busy.
        try:
            key_count = max(1, len(config.get_api_credentials()))
        except Exception:
            key_count = max(1, config.max_concurrent_downloads)
        max_conc = max(config.max_concurrent_downloads, key_count)
        self.downloader = AsyncGloFASDownloader(str(self.temp_dir), max_concurrent=max_conc)
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / config.log_file_name
        
        # Create formatter
        formatter = logging.Formatter(config.log_format)
        
        # File handler with context manager support
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, config.log_level))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def filter_bad_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out events that fall within known bad data periods.
        Expects a 'flood_start_date' column; will coerce to datetime if needed using DD-MM-YYYY.
        """
        if events_df.empty:
            return events_df
        
        # Ensure datetime dtype (robust to ISO YYYY-MM-DD and DD-MM-YYYY)
        if 'flood_start_date' in events_df.columns:
            try:
                if not np.issubdtype(events_df['flood_start_date'].dtype, np.datetime64):
                    events_df['flood_start_date'] = _parse_dates_robust_series(events_df['flood_start_date'])
            except Exception:
                events_df['flood_start_date'] = _parse_dates_robust_series(events_df['flood_start_date'])
        else:
            # Backward compatibility: fall back to 'start_date' if present
            if 'start_date' in events_df.columns:
                events_df['flood_start_date'] = _parse_dates_robust_series(events_df['start_date'])
            else:
                self.logger.error("No 'flood_start_date' or 'start_date' column found in events DataFrame")
                return events_df
        
        original_count = len(events_df)
        
        # Check each event's flood_start_date using config
        bad_event_mask = events_df['flood_start_date'].apply(
            lambda x: config.is_bad_date(x.date()) if isinstance(x, (pd.Timestamp, datetime)) else False
        )
        bad_events = events_df[bad_event_mask]
        
        if len(bad_events) > 0:
            self.logger.warning(f"Found {len(bad_events)} events in known bad data periods:")
            for idx, event in bad_events.iterrows():
                event_date = event['flood_start_date']
                try:
                    event_date_str = event_date.strftime('%Y-%m-%d')
                except Exception:
                    event_date_str = str(event_date)
                self.logger.warning(f"  Skipping event on {event_date_str}")
        
        # Filter out bad events
        filtered_df = events_df[~bad_event_mask].copy()
        
        removed_count = original_count - len(filtered_df)
        if removed_count > 0:
            self.logger.info(f"Filtered {removed_count} bad events. {len(filtered_df)} events remaining.")
        
        return filtered_df
    
    def check_progress(self, year: int, month: int, part: int) -> bool:
        """Check if a chunk has already been processed."""
        filename = f"enriched_{year}-{month:02d}_part{part}.csv"
        filepath = self.output_dir / filename
        
        if filepath.exists():
            self.logger.info(f"✅ Chunk already exists: {filename}")
            return True
        return False
    
    def get_event_dates_for_month(self, events_file: Path) -> set:
        """Get all event dates for a month (excluding bad data periods)."""
        with open(events_file, 'r') as f:
            df = pd.read_csv(f)
        
        # Parse flood_start_date (robust to ISO and DD-MM-YYYY)
        if 'flood_start_date' in df.columns:
            df['flood_start_date'] = _parse_dates_robust_series(df['flood_start_date'])
        elif 'start_date' in df.columns:
            # Backward compatibility
            df['flood_start_date'] = _parse_dates_robust_series(df['start_date'])
        else:
            self.logger.error("Events file missing 'flood_start_date' column")
            return set()
        
        # Drop invalid dates
        df = df.dropna(subset=['flood_start_date'])
        
        # Filter out bad events
        df = self.filter_bad_events(df)
        
        if df.empty:
            return set()
        
        return set(df['flood_start_date'].dt.day)

    def get_chunk_dates(self, year: int, month: int, start_day: int, end_day: int) -> List[str]:
        """Get ISO date strings for a chunk (skips invalid days)."""
        dates: List[str] = []
        for day in range(start_day, end_day + 1):
            try:
                d = datetime(year, month, day)
                dates.append(d.strftime('%Y-%m-%d'))
            except ValueError:
                continue
        return dates

    def chunk_has_forecast(self, year: int, month: int, start_day: int, end_day: int) -> bool:
        """Check if a chunk should have forecast data based on cutoff date."""
        for day in range(start_day, end_day + 1):
            try:
                if datetime(year, month, day).date() >= config.glofas_forecast_start_date:
                    return True
            except ValueError:
                continue
        return False
    
    def determine_spatial_strategy(self, flood_type: str) -> str:
        """Determine the best spatial strategy based on flood type."""
        try:
            ft = str(flood_type or "").lower()
        except Exception:
            ft = ""
        
        # Terms indicating flash/urban floods (use nearest point)
        nearest_terms = ("flash", "urban", "pluvial", "wadi", "coastal")
        # Terms indicating riverine floods (use mean over bbox)
        mean_terms = ("riverine", "river", "snowmelt", "el nino", "elnino", "storm surge", "storm_surge")
        
        # Check if it's a mixed flood type
        is_mixed = (
            '/' in ft or 
            (any(t in ft for t in mean_terms) and 
             any(t in ft for t in ("flash", "urban", "pluvial", "landslide", "dam"))) or 
            "mixed" in ft
        )
        
        if is_mixed:
            return "hybrid"
        if any(t in ft for t in nearest_terms):
            return "nearest"
        if any(t in ft for t in mean_terms):
            return "mean_bbox"
        
        return "nearest"  # Default for unknown or unclassified types
    
    def get_glofas_buffer(self, flood_type: str) -> int:
        """Return buffer in meters for GloFAS mean-bbox selection."""
        try:
            ft = str(flood_type or "").lower()
        except Exception:
            ft = ""
        
        riverine_terms = ['riverine', 'river', 'snowmelt', 'el nino', 'elnino', 'storm surge', 'storm_surge']
        if any(term in ft for term in riverine_terms):
            return config.glofas_riverine_bbox_meters
        
        return config.aoi_buffer_meters
    
    def bbox_from_point(self, lat: float, lon: float, buffer_m: int = None) -> Tuple[float, float, float, float]:
        """Calculate bounding box from a point and buffer distance."""
        if buffer_m is None:
            buffer_m = config.aoi_buffer_meters
        
        try:
            # Use Azimuthal Equidistant projection centered on the point
            proj_string = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
            transformer_to_aeqd = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
            transformer_from_aeqd = Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)
            
            # Transform center point to projected coordinates
            x_center, y_center = transformer_to_aeqd.transform(lon, lat)
            
            # Calculate bbox corners in projected coordinates
            x_min = x_center - buffer_m
            x_max = x_center + buffer_m
            y_min = y_center - buffer_m
            y_max = y_center + buffer_m
            
            # Transform corners back to lat/lon
            lon_min, lat_min = transformer_from_aeqd.transform(x_min, y_min)
            lon_max, lat_max = transformer_from_aeqd.transform(x_max, y_max)
            
            return lon_min, lon_max, lat_min, lat_max
        
        except Exception as e:
            self.logger.warning(f"Projection-based bbox calculation failed: {e}. Using simple approximation.")
            # Fallback to simple approximation
            lat_buffer = buffer_m / 111000.0
            lon_buffer = buffer_m / (111000.0 * np.cos(np.radians(lat)))
            return lon - lon_buffer, lon + lon_buffer, lat - lat_buffer, lat + lat_buffer

    def _normalize_lon_for_ds(self, lon_value: float, ds: xr.Dataset, lon_coord: str) -> float:
        """Normalize a longitude value to match the dataset's longitude domain."""
        try:
            lons = ds[lon_coord].values
            min_lon = float(np.nanmin(lons))
            max_lon = float(np.nanmax(lons))
        except Exception:
            return lon_value
        # Dataset uses 0..360 domain
        if min_lon >= 0.0 and max_lon <= 360.0:
            return (lon_value + 360.0) % 360.0
        # Otherwise assume -180..180 domain and wrap into it
        wrapped = ((lon_value + 180.0) % 360.0) - 180.0
        return wrapped

    def _normalize_lon_pair_for_ds(self, lon_min: float, lon_max: float, ds: xr.Dataset, lon_coord: str) -> Tuple[float, float]:
        """Normalize a pair of longitude bounds to match dataset domain."""
        return (
            self._normalize_lon_for_ds(lon_min, ds, lon_coord),
            self._normalize_lon_for_ds(lon_max, ds, lon_coord),
        )

    def _slice_for_coord(self, ds: xr.Dataset, coord_name: str, vmin: float, vmax: float):
        """Return a slice respecting the coordinate order (ascending/descending)."""
        values = ds[coord_name].values
        try:
            ascending = bool(values[1] > values[0])
        except Exception:
            ascending = True
        return slice(vmin, vmax) if ascending else slice(vmax, vmin)
    
    def _select_primary_var(self, obj: xr.Dataset, preferred: Optional[List[str]] = None) -> Optional[xr.DataArray]:
        """Return a primary DataArray from a Dataset or pass-through a DataArray.
        Preference order tries common GloFAS discharge names, then first data_var.
        """
        if isinstance(obj, xr.DataArray):
            return obj
        if not isinstance(obj, xr.Dataset):
            return None
        preferred = preferred or [
            'dis24', 'dis', 'river_discharge', 'streamflow', 'sf', 'Q', 'Qw'
        ]
        try:
            for name in preferred:
                if name in obj.data_vars:
                    return obj[name]
            # Fallback: first data variable
            first = next(iter(obj.data_vars))
            return obj[first]
        except Exception:
            return None
    
    @contextmanager
    def open_grib_dataset(self, grib_file: Path, **kwargs):
        """Context manager for opening GRIB files with proper resource management."""
        ds = None
        last_exc: Optional[Exception] = None
        try:
            raw_kwargs = dict(kwargs.get('backend_kwargs', {}))
            candidates = raw_kwargs.pop('filter_by_keys_candidates', None)
            base_kwargs = {"indexpath": ""}
            base_kwargs.update(raw_kwargs)

            def _open(backend_kwargs: Dict[str, Any]) -> xr.Dataset:
                with suppress_stderr():
                    return xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs=backend_kwargs)

            if candidates:
                for option in candidates:
                    attempt_kwargs = base_kwargs.copy()
                    attempt_kwargs.update(option)
                    try:
                        ds = _open(attempt_kwargs)
                        self.logger.debug(f"Opened {grib_file.name} with {attempt_kwargs}")
                        yield ds
                        return
                    except Exception as exc:
                        last_exc = exc
                        self.logger.debug(f"cfgrib open failed for {grib_file.name} with {attempt_kwargs}: {exc}")
                        continue
                if last_exc:
                    raise last_exc
                raise RuntimeError(f"Unable to open {grib_file} with provided candidate filters")
            else:
                ds = _open(base_kwargs)
                yield ds
        finally:
            if ds is not None:
                ds.close()
    
    def _cfgrib_candidate_kwargs(self, data_type: str) -> List[Dict[str, Dict[str, Any]]]:
        """Return ordered cfgrib backend kwargs candidates for a data type."""
        return [
            {'filter_by_keys': {'dataType': data_type, 'numberOfPoints': 5400000}},
            {'filter_by_keys': {'dataType': data_type}},
            {'filter_by_keys': {'dataType': data_type, 'numberOfPoints': 21600000}},
        ]

    def _enter_cfgrib_with_variants(self, stack: ExitStack, grib_file: Path, candidates: List[Dict[str, Dict[str, Any]]]):
        """Try opening cfgrib dataset with disambiguation fallbacks."""
        lat_keys = {'latitude', 'lat', 'y', 'grid_latitude', 'latitude_0', 'lat_0'}
        lon_keys = {'longitude', 'lon', 'x', 'grid_longitude', 'longitude_0', 'lon_0'}

        for backend_kwargs in candidates:
            try:
                # Probe dataset to ensure it contains latitude/longitude before committing
                with self.open_grib_dataset(grib_file, backend_kwargs=backend_kwargs) as probe:
                    coord_names = set(getattr(probe, 'coords', {}))
                    dim_names = set(getattr(probe, 'dims', {}))
                    has_lat = bool(lat_keys & coord_names) or bool(lat_keys & dim_names)
                    has_lon = bool(lon_keys & coord_names) or bool(lon_keys & dim_names)

                if not (has_lat and has_lon):
                    self.logger.debug(
                        f"cfgrib probe missing lat/lon for {grib_file.name} with {backend_kwargs}; skipping"
                    )
                    continue

                ds = stack.enter_context(self.open_grib_dataset(grib_file, backend_kwargs=backend_kwargs))
                self.logger.debug(f"Opened {grib_file.name} with {backend_kwargs}")
                return ds
            except Exception as e:
                self.logger.debug(f"cfgrib open failed for {grib_file.name} with {backend_kwargs}: {e}")
                continue
        return None
    
    @asynccontextmanager
    async def open_grib_dataset_async(self, grib_file: Path, **kwargs):
        """Async context manager for opening GRIB files."""
        ds = None
        loop = asyncio.get_event_loop()
        candidates: Optional[List[Dict[str, Any]]] = None
        last_exc: Optional[Exception] = None
        try:
            raw_kwargs = dict(kwargs.get('backend_kwargs', {}))
            candidates = raw_kwargs.pop('filter_by_keys_candidates', None)
            base_kwargs = {"indexpath": ""}
            base_kwargs.update(raw_kwargs)

            def _blocking_open(open_kwargs: Dict[str, Any]) -> xr.Dataset:
                with suppress_stderr():
                    return xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs=open_kwargs)

            if candidates:
                for option in candidates:
                    attempt_kwargs = base_kwargs.copy()
                    attempt_kwargs.update(option)
                    try:
                        ds = await loop.run_in_executor(None, lambda ow=attempt_kwargs: _blocking_open(ow))
                        self.logger.debug(f"Opened {grib_file.name} with {attempt_kwargs} (async)")
                        yield ds
                        return
                    except Exception as exc:
                        last_exc = exc
                        self.logger.debug(f"cfgrib async open failed for {grib_file.name} with {attempt_kwargs}: {exc}")
                        continue
                if last_exc:
                    raise last_exc
                raise RuntimeError(f"Unable to open {grib_file} with provided candidates")
            else:
                ds = await loop.run_in_executor(None, lambda: _open_cfgrib_with_suppressed_logs(grib_file, base_kwargs))
                yield ds
        finally:
            if ds is not None:
                await loop.run_in_executor(None, ds.close)
    def extract_grib_value(self, grib_file: Path, lat: float, lon: float, event_date: datetime,
                           spatial_strategy: str = 'nearest', flood_type: str = '',
                           backend_kwargs: Optional[dict] = None,
                           dataset: Optional[xr.Dataset] = None) -> Optional[float]:
        """Extract a value from a GRIB file with spatial strategy and resource management.

        When a pre-opened dataset is provided, it is reused to avoid repeated file I/O.
        """

        backend_kwargs = backend_kwargs or {}

        def _extract_from_dataset(ds: xr.Dataset) -> Optional[float]:
            # Handle different coordinate names
            lat_coord = next((c for c in ['latitude', 'lat', 'y'] if c in ds.coords), None)
            lon_coord = next((c for c in ['longitude', 'lon', 'x'] if c in ds.coords), None)

            if not lat_coord or not lon_coord:
                self.logger.warning("Could not find lat/lon coordinates in GRIB file")
                return None
            var = self._select_primary_var(ds)
            if var is None:
                self.logger.warning("No data variable found in GRIB dataset")
                return None
            # Determine time dimensions for selection/indexing
            has_time_dim = 'time' in getattr(var, 'dims', ())
            has_time_coord = 'time' in getattr(var, 'coords', {})
            has_vtime_dim = 'valid_time' in getattr(var, 'dims', ())
            has_vtime_coord = 'valid_time' in getattr(var, 'coords', {})
            time_label_dim = 'time' if (has_time_dim or has_time_coord) else ('valid_time' if (has_vtime_dim or has_vtime_coord) else None)
            time_index_dim = 'time' if has_time_dim else ('valid_time' if has_vtime_dim else None)

            # Apply spatial strategy
            if spatial_strategy in ['mean_bbox', 'hybrid']:
                # Calculate mean over bounding box
                buffer_size = self.get_glofas_buffer(flood_type)
                lon_min, lon_max, lat_min, lat_max = self.bbox_from_point(lat, lon, buffer_size)
                # Normalize longitudes to dataset domain
                lon_min, lon_max = self._normalize_lon_pair_for_ds(lon_min, lon_max, ds, lon_coord)
                lon_norm = self._normalize_lon_for_ds(lon, ds, lon_coord)

                try:
                    # Select data within bounding box
                    lat_slice = self._slice_for_coord(ds, lat_coord, float(lat_min), float(lat_max))
                    lon_slice = self._slice_for_coord(ds, lon_coord, float(lon_min), float(lon_max))
                    subset = var.sel({lat_coord: lat_slice, lon_coord: lon_slice}, method='nearest')
                    if time_label_dim and (time_label_dim in subset.dims or time_label_dim in subset.coords):
                        subset = subset.sel({time_label_dim: event_date}, method='nearest')
                    # Calculate mean over spatial dimensions (scalar or time-indexed)
                    value = subset.mean(dim=[lat_coord, lon_coord])

                except Exception as e:
                    self.logger.debug(f"Mean bbox extraction failed, using nearest: {e}")
                    # Fallback to nearest point with sel
                    try:
                        value = var.sel({lat_coord: lat, lon_coord: lon_norm}, method='nearest')
                        if time_label_dim and (time_label_dim in value.dims or time_label_dim in value.coords):
                            value = value.sel({time_label_dim: event_date}, method='nearest')
                    except Exception as e2:
                        # Manual index-based fallback
                        self.logger.debug(f"Nearest sel failed, using manual isel: {e2}")
                        lat_idx = int(np.argmin(np.abs(ds[lat_coord].values - lat)))
                        lon_idx = int(np.argmin(np.abs(ds[lon_coord].values - lon)))
                        if time_index_dim:
                            coord_source = var[time_label_dim].values if time_label_dim else var[time_index_dim].values
                            time_idx = int(np.argmin(np.abs(coord_source - np.datetime64(event_date))))
                            value = var.isel({lat_coord: lat_idx, lon_coord: lon_idx, time_index_dim: time_idx})
                        else:
                            value = var.isel({lat_coord: lat_idx, lon_coord: lon_idx})
            else:
                # Use nearest point strategy
                try:
                    lon_norm = self._normalize_lon_for_ds(lon, ds, lon_coord)
                    value = var.sel({lat_coord: lat, lon_coord: lon_norm}, method='nearest')
                    if time_label_dim and (time_label_dim in value.dims or time_label_dim in value.coords):
                        value = value.sel({time_label_dim: event_date}, method='nearest')
                except Exception as e:
                    # Manual index-based fallback
                    self.logger.debug(f"Nearest selection failed, using manual isel: {e}")
                    lat_idx = int(np.argmin(np.abs(ds[lat_coord].values - lat)))
                    lon_idx = int(np.argmin(np.abs(ds[lon_coord].values - lon)))
                    if time_index_dim:
                        coord_source = var[time_label_dim].values if time_label_dim else var[time_index_dim].values
                        time_idx = int(np.argmin(np.abs(coord_source - np.datetime64(event_date))))
                        value = var.isel({lat_coord: lat_idx, lon_coord: lon_idx, time_index_dim: time_idx})
                    else:
                        value = var.isel({lat_coord: lat_idx, lon_coord: lon_idx})

            # Extract the actual scalar value
            try:
                value = value.squeeze()
                result = float(value.values.item() if hasattr(value.values, 'item') else value.values)
            except Exception:
                result = float(value.values) if hasattr(value, 'values') else None
            return result if (result is not None and not np.isnan(result)) else None

        try:
            if dataset is not None:
                return _extract_from_dataset(dataset)
            with self.open_grib_dataset(grib_file, backend_kwargs=backend_kwargs) as ds:
                return _extract_from_dataset(ds)
        except Exception as e:
            self.logger.warning(f"GRIB extraction failed (primary): {e}")
            # Fallback: try cfgrib.open_datasets and merge
            try:
                from cfgrib import xarray_store
                combined_backend_kwargs = {"indexpath": ""}
                combined_backend_kwargs.update(backend_kwargs)
                with suppress_stderr():
                    ds_list = xarray_store.open_datasets(grib_file, backend_kwargs=combined_backend_kwargs)
                try:
                    ds = xr.merge(ds_list)
                    return _extract_from_dataset(ds)
                finally:
                    # Close all component datasets if they expose close()
                    try:
                        for d in ds_list:
                            close_fn = getattr(d, 'close', None)
                            if callable(close_fn):
                                close_fn()
                    except Exception:
                        pass
            except Exception as e2:
                self.logger.warning(f"GRIB extraction failed (fallback): {e2}")
                return None
    
    def extract_ensemble_statistics(self, grib_file: Path, lat: float, lon: float, event_date: datetime,
                                   spatial_strategy: str = 'nearest', flood_type: str = '',
                                   backend_kwargs: Optional[dict] = None,
                                   dataset: Optional[xr.Dataset] = None) -> Optional[Dict]:
        """Extract ensemble statistics from GRIB file with resource management."""

        backend_kwargs = backend_kwargs or {}

        def _extract_from_dataset(ds: xr.Dataset) -> Optional[Dict]:
            lat_candidates = ['latitude', 'lat', 'y', 'grid_latitude', 'latitude_0', 'lat_0']
            lon_candidates = ['longitude', 'lon', 'x', 'grid_longitude', 'longitude_0', 'lon_0']

            lat_coord = next((c for c in lat_candidates if c in ds.coords), None)
            lon_coord = next((c for c in lon_candidates if c in ds.coords), None)

            var = self._select_primary_var(ds)
            if var is None:
                self.logger.warning("No data variable found for ensemble extraction")
                return None

            var_dims = list(getattr(var, 'dims', ()) or [])
            if not lat_coord:
                lat_coord = next((c for c in lat_candidates if c in getattr(var, 'coords', {})), None)
            if not lat_coord:
                lat_coord = next((c for c in lat_candidates if c in var_dims), None)
            if not lon_coord:
                lon_coord = next((c for c in lon_candidates if c in getattr(var, 'coords', {})), None)
            if not lon_coord:
                lon_coord = next((c for c in lon_candidates if c in var_dims), None)

            if not lat_coord or not lon_coord:
                self.logger.warning(
                    "Could not find lat/lon coordinates; coords=%s dims=%s",
                    list(ds.coords.keys()),
                    var_dims,
                )
                return None
            # Determine time dimensions for selection/indexing
            has_time_dim = 'time' in getattr(var, 'dims', ())
            has_time_coord = 'time' in getattr(var, 'coords', {})
            has_vtime_dim = 'valid_time' in getattr(var, 'dims', ())
            has_vtime_coord = 'valid_time' in getattr(var, 'coords', {})
            time_label_dim = 'time' if (has_time_dim or has_time_coord) else ('valid_time' if (has_vtime_dim or has_vtime_coord) else None)
            time_index_dim = 'time' if has_time_dim else ('valid_time' if has_vtime_dim else None)
            # Find ensemble dimension
            ensemble_dim = next((d for d in ['number', 'realization', 'ensemble'] if d in ds.dims), None)

            if not ensemble_dim:
                self.logger.warning("No ensemble dimension found")
                return None

            # Apply spatial strategy
            if spatial_strategy in ['mean_bbox', 'hybrid']:
                buffer_size = self.get_glofas_buffer(flood_type)
                lon_min, lon_max, lat_min, lat_max = self.bbox_from_point(lat, lon, buffer_size)
                # Normalize longitudes to dataset domain
                lon_min, lon_max = self._normalize_lon_pair_for_ds(lon_min, lon_max, ds, lon_coord)
                lon_norm = self._normalize_lon_for_ds(lon, ds, lon_coord)

                try:
                    # Select data within bounding box
                    lat_slice = self._slice_for_coord(ds, lat_coord, float(lat_min), float(lat_max))
                    lon_slice = self._slice_for_coord(ds, lon_coord, float(lon_min), float(lon_max))
                    ensemble_values = var.sel(
                        {lat_coord: lat_slice,
                         lon_coord: lon_slice},
                        method='nearest'
                    )
                    if time_label_dim and (time_label_dim in ensemble_values.dims or time_label_dim in ensemble_values.coords):
                        ensemble_values = ensemble_values.sel({time_label_dim: event_date}, method='nearest')

                    # Calculate mean over spatial dimensions for each ensemble member
                    ensemble_values = ensemble_values.mean(dim=[lat_coord, lon_coord])

                except Exception as e:
                    self.logger.debug(f"Mean bbox failed for ensemble, using nearest: {e}")
                    ensemble_values = var.sel({lat_coord: lat, lon_coord: lon_norm}, method='nearest')
                    if time_label_dim and (time_label_dim in ensemble_values.dims or time_label_dim in ensemble_values.coords):
                        ensemble_values = ensemble_values.sel({time_label_dim: event_date}, method='nearest')
            else:
                # Use nearest point
                lon_norm = self._normalize_lon_for_ds(lon, ds, lon_coord)
                ensemble_values = var.sel({lat_coord: lat, lon_coord: lon_norm}, method='nearest')
                if time_label_dim and (time_label_dim in ensemble_values.dims or time_label_dim in ensemble_values.coords):
                    ensemble_values = ensemble_values.sel({time_label_dim: event_date}, method='nearest')

            # Calculate statistics (exclude control member if present)
            # If an explicit ensemble dimension exists, try to drop control (number==0)
            try:
                if 'number' in ensemble_values.dims and 0 in set(map(int, np.array(ensemble_values['number']).astype(int))):
                    try:
                        ensemble_values = ensemble_values.sel(number=[n for n in np.array(ensemble_values['number']).astype(int) if n != 0])
                    except Exception:
                        pass
            except Exception:
                pass

            values = ensemble_values.values.flatten()
            values = values[~np.isnan(values)]

            if len(values) > 0:
                return {
                    'glofas_forecast_mean': float(np.mean(values)),
                    'glofas_forecast_median': float(np.median(values)),
                    'glofas_forecast_std_dev': float(np.std(values)),
                    'glofas_forecast_10th_percentile': float(np.percentile(values, 10)),
                    'glofas_forecast_90th_percentile': float(np.percentile(values, 90))
                }

            return None

        try:
            if dataset is not None:
                return _extract_from_dataset(dataset)
            with self.open_grib_dataset(grib_file, backend_kwargs=backend_kwargs) as ds:
                return _extract_from_dataset(ds)
        except Exception as e:
            self.logger.warning(f"Ensemble extraction failed (primary): {e}")
            # Fallback: try cfgrib.open_datasets and merge
            try:
                from cfgrib import xarray_store
                combined_backend_kwargs = {"indexpath": ""}
                combined_backend_kwargs.update(backend_kwargs)
                with suppress_stderr():
                    ds_list = xarray_store.open_datasets(grib_file, backend_kwargs=combined_backend_kwargs)
                    try:
                        ds = xr.merge(ds_list)
                        return _extract_from_dataset(ds)
                    finally:
                        try:
                            for d in ds_list:
                                close_fn = getattr(d, 'close', None)
                                if callable(close_fn):
                                    close_fn()
                        except Exception:
                            pass
            except Exception as e2:
                self.logger.warning(f"Ensemble extraction failed (fallback): {e2}")
                return None
    
    def extract_forecast_features(self, events_df: pd.DataFrame, forecast_zip_file: Path) -> pd.DataFrame:
        """Extract forecast features from GRIB files with intelligent spatial strategy."""
        self.logger.info("Extracting forecast features...")
        
        # Initialize forecast features
        forecast_features = [
            'glofas_forecast_control',
            'glofas_forecast_mean',
            'glofas_forecast_median',
            'glofas_forecast_std_dev',
            'glofas_forecast_10th_percentile',
            'glofas_forecast_90th_percentile'
        ]
        
        for feature in forecast_features:
            events_df[feature] = np.nan
        
        events_df['forecast_spatial_strategy'] = np.nan
        
        # Ensure date column is datetime
        if 'flood_start_date' in events_df.columns and not np.issubdtype(events_df['flood_start_date'].dtype, np.datetime64):
            events_df['flood_start_date'] = _parse_dates_robust_series(events_df['flood_start_date'])
        
        # Check if forecast data is available
        if (events_df['flood_start_date'].dt.date < config.glofas_forecast_start_date).all():
            self.logger.info("All events before GloFAS forecast availability. Skipping.")
            return events_df
        
        if not forecast_zip_file.exists() or forecast_zip_file.stat().st_size == 0:
            self.logger.warning(f"Forecast file not found or empty: {forecast_zip_file}")
            return events_df
        
        # If file is not a valid ZIP, fall back to direct GRIB processing
        is_zip = False
        try:
            is_zip = zipfile.is_zipfile(forecast_zip_file)
        except Exception:
            is_zip = False

        if not is_zip:
            self.logger.info("Forecast file is not a ZIP. Using direct GRIB processing.")

            cf_candidates = self._cfgrib_candidate_kwargs('cf')
            pf_candidates = self._cfgrib_candidate_kwargs('pf')
            backend_cf = {'filter_by_keys_candidates': cf_candidates}
            backend_pf = {'filter_by_keys_candidates': pf_candidates}

            with ExitStack() as stack:
                # Try with resolution disambiguation first
                cf_ds = self._enter_cfgrib_with_variants(stack, forecast_zip_file, cf_candidates)
                pf_ds = self._enter_cfgrib_with_variants(stack, forecast_zip_file, pf_candidates)
                if cf_ds is None:
                    self.logger.debug("Control dataset open with fallbacks failed; will rely on per-event open.")
                if pf_ds is None:
                    self.logger.debug("Ensemble dataset open with fallbacks failed; will rely on per-event open.")

                for idx, event in events_df.iterrows():
                    try:
                        lat = event['latitude']
                        lon = event['longitude']
                        event_date = event['flood_start_date']
                        flood_type = event.get('flood_type', '')

                        spatial_strategy = self.determine_spatial_strategy(flood_type)
                        events_df.loc[idx, 'forecast_spatial_strategy'] = spatial_strategy

                        # Extract control (nearest field) from single GRIB
                        control_value = self.extract_grib_value(
                            forecast_zip_file, lat, lon, event_date,
                            spatial_strategy=spatial_strategy,
                            flood_type=flood_type,
                            backend_kwargs=backend_cf,
                            dataset=cf_ds
                        )
                        if control_value is not None:
                            events_df.loc[idx, 'glofas_forecast_control'] = control_value

                        # Try ensemble stats from same GRIB (if ensemble present)
                        ensemble_stats = self.extract_ensemble_statistics(
                            forecast_zip_file, lat, lon, event_date,
                            spatial_strategy=spatial_strategy,
                            flood_type=flood_type,
                            backend_kwargs=backend_pf,
                            dataset=pf_ds
                        )
                        if ensemble_stats:
                            for feature, value in ensemble_stats.items():
                                events_df.loc[idx, feature] = value
                    except Exception as e:
                        self.logger.warning(f"Error processing event {idx}: {e}")

            successful = events_df['glofas_forecast_control'].notna().sum()
            self.logger.info(f"Extracted forecast features for {successful}/{len(events_df)} events (direct GRIB)")
            return events_df

        # Extract forecast data with proper resource management (ZIP case)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract ZIP file
            with zipfile.ZipFile(forecast_zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # Find GRIB files
            grib_files = list(temp_path.glob("*.grib*"))
            cf_file = next((f for f in grib_files if 'cf' in f.name.lower() or 'control' in f.name.lower()), None)
            pf_file = next((f for f in grib_files if 'pf' in f.name.lower() or 'perturbed' in f.name.lower()), None)
            
            if not cf_file and not pf_file:
                self.logger.warning("Could not find forecast GRIB files in ZIP; attempting to use any GRIB present")
                # Fallback: use any GRIB for both control and ensemble attempts
                if grib_files:
                    cf_file = grib_files[0]
                    pf_file = grib_files[0]
                else:
                    return events_df
            
            with ExitStack() as stack:
                cf_candidates = self._cfgrib_candidate_kwargs('cf')
                pf_candidates = self._cfgrib_candidate_kwargs('pf')
                backend_cf = {'filter_by_keys_candidates': cf_candidates}
                backend_pf = {'filter_by_keys_candidates': pf_candidates}

                cf_ds = None
                pf_ds = None

                if cf_file:
                    try:
                        cf_ds = self._enter_cfgrib_with_variants(stack, cf_file, cf_candidates)
                    except Exception as e:
                        self.logger.debug(f"Failed to preload control dataset from ZIP: {e}")
                        cf_ds = None

                if pf_file:
                    try:
                        pf_ds = self._enter_cfgrib_with_variants(stack, pf_file, pf_candidates)
                    except Exception as e:
                        self.logger.debug(f"Failed to preload ensemble dataset from ZIP: {e}")
                        pf_ds = None

                # Process each event
                for idx, event in events_df.iterrows():
                    try:
                        lat = event['latitude']
                        lon = event['longitude']
                        event_date = event['flood_start_date']
                        flood_type = event.get('flood_type', '')

                        # Determine spatial strategy
                        spatial_strategy = self.determine_spatial_strategy(flood_type)
                        events_df.loc[idx, 'forecast_spatial_strategy'] = spatial_strategy

                        # Extract control forecast (if file available)
                        if cf_file:
                            control_value = self.extract_grib_value(
                                cf_file, lat, lon, event_date,
                                spatial_strategy=spatial_strategy,
                                flood_type=flood_type,
                                backend_kwargs=backend_cf,
                                dataset=cf_ds
                            )
                            if control_value is not None:
                                events_df.loc[idx, 'glofas_forecast_control'] = control_value

                        # Extract ensemble statistics (if file available)
                        if pf_file:
                            ensemble_stats = self.extract_ensemble_statistics(
                                pf_file, lat, lon, event_date,
                                spatial_strategy=spatial_strategy,
                                flood_type=flood_type,
                                backend_kwargs=backend_pf,
                                dataset=pf_ds
                            )
                            if ensemble_stats:
                                for feature, value in ensemble_stats.items():
                                    events_df.loc[idx, feature] = value

                    except Exception as e:
                        self.logger.warning(f"Error processing event {idx}: {e}")
        
        successful = events_df['glofas_forecast_control'].notna().sum()
        self.logger.info(f"Extracted forecast features for {successful}/{len(events_df)} events")
        
        return events_df

    def remove_leaky_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Remove known leaky features to prevent data leakage into models."""
        self.logger.info("Removing leaky features...")
        leaky_features = [
            'confidence_score', 'longitude', 'latitude', 'relative_increase',
            'glofas_rowe_relative_increase', 'glofas_swir_relative_increase',
            'glofas_dis24_relative_increase', 'glofas_rowe_ratio_on_event_to_mean',
            'glofas_swir_ratio_on_event_to_mean', 'glofas_dis24_ratio_on_event_to_mean',
            'glofas_rowe_z_on_event', 'glofas_swir_z_on_event', 'glofas_dis24_z_on_event',
            'data_quality_issues', 'data_quality_warnings', 'data_quality_valid',
            'processing_date', 'processing_version', 'aoi_buffer_meters'
        ]
        features_to_remove = [col for col in leaky_features if col in events_df.columns]
        events_df = events_df.drop(columns=features_to_remove)
        self.logger.info(f"Removed {len(features_to_remove)} leaky features")
        return events_df
    
    async def download_chunk_data(self, year: int, month: int, start_day: int, end_day: int) -> Optional[Path]:
        """Download forecast data for a chunk of days using async downloader."""
        # Generate dates string
        dates = []
        for day in range(start_day, end_day + 1):
            try:
                date_obj = datetime(year, month, day)
                if not config.is_bad_date(date_obj.date()):
                    dates.append(date_obj.strftime('%Y-%m-%d'))
            except ValueError:
                continue  # Invalid date
        
        if not dates:
            return None
        
        dates_string = ','.join(dates)
        filename = f"forecast_{year}_{month:02d}_{start_day:02d}_{end_day:02d}.zip"
        
        # Use shared async downloader
        self.logger.info(f"📤 Downloading forecast data for {filename}")
        success = await self.downloader.download_all(dates_string, filename)
        
        if success:
            return self.temp_dir / filename
        else:
            self.logger.warning(f"Failed to download {filename}")
            return None
    
    async def process_chunk_async(self, year: int, month: int, start_day: int, end_day: int,
                                  events_file: Path,
                                  prefetched_forecast_path: Optional[Path] = None) -> bool:
        """Process a single chunk of days with async download."""
        self.logger.info(f"Processing {year}-{month:02d} days {start_day}-{end_day}")
        
        # Check if already processed
        part = 1 if start_day <= 15 else 2
        if self.check_progress(year, month, part):
            return True
        
        # Load monthly events with context manager
        with open(events_file, 'r') as f:
            events_df = pd.read_csv(f)
        
        # Parse flood_start_date robustly; fallback to legacy 'start_date'
        if 'flood_start_date' in events_df.columns:
            events_df['flood_start_date'] = _parse_dates_robust_series(events_df['flood_start_date'])
        elif 'start_date' in events_df.columns:
            events_df['flood_start_date'] = _parse_dates_robust_series(events_df['start_date'])
        else:
            self.logger.error("Events file missing 'flood_start_date' column")
            return True
        
        # Drop invalid dates
        events_df = events_df.dropna(subset=['flood_start_date'])
        
        # Filter bad events
        events_df = self.filter_bad_events(events_df)
        
        if events_df.empty:
            self.logger.info("No valid events after filtering")
            return True
        
        # Filter events for this chunk
        chunk_events = events_df[
            (events_df['flood_start_date'].dt.day >= start_day) & 
            (events_df['flood_start_date'].dt.day <= end_day)
        ].copy()
        
        if len(chunk_events) == 0:
            self.logger.info("No events in this chunk")
            return True
        
        self.logger.info(f"Processing {len(chunk_events)} events")
        
        # Check if forecast data is needed
        event_dates = chunk_events['flood_start_date'].dt.date
        has_forecast_data = any(d >= config.glofas_forecast_start_date for d in event_dates)
        
        forecast_filepath = None
        if has_forecast_data:
            # Use prefetched file if provided
            if prefetched_forecast_path and prefetched_forecast_path.exists():
                forecast_filepath = prefetched_forecast_path
            else:
                # Download now
                forecast_filepath = await self.download_chunk_data(year, month, start_day, end_day)
        
        # Extract forecast features if available
        if forecast_filepath and forecast_filepath.exists():
            chunk_events = self.extract_forecast_features(chunk_events, forecast_filepath)
        else:
            # Initialize forecast features as NaN
            for feature in ['glofas_forecast_control', 'glofas_forecast_mean',
                          'glofas_forecast_median', 'glofas_forecast_std_dev',
                          'glofas_forecast_10th_percentile', 'glofas_forecast_90th_percentile']:
                chunk_events[feature] = np.nan
        
        # Remove leaky features before saving
        chunk_events = self.remove_leaky_features(chunk_events)
        
        # Save chunk results
        output_filename = f"enriched_{year}-{month:02d}_part{part}.csv"
        output_filepath = self.output_dir / output_filename
        
        with open(output_filepath, 'w') as f:
            chunk_events.to_csv(f, index=False)
        
        self.logger.info(f"✅ Saved chunk results: {output_filename}")

        # Try to combine month immediately if both parts are present
        try:
            self.try_combine_month_if_ready(year, month)
        except Exception as e:
            self.logger.warning(f"Combine check failed for {year}-{month:02d}: {e}")

        # Clean up forecast data
        if forecast_filepath and forecast_filepath.exists():
            forecast_filepath.unlink()
        
        return True
    
    async def process_month_async(self, year: int, month: int) -> bool:
        """Process a complete month with async downloads."""
        self.logger.info(f"Processing month {month} for year {year}")
        
        events_file = self.events_dir / f"events_{year}_{month:02d}.csv"
        
        if not events_file.exists():
            self.logger.warning(f"Events file not found: {events_file}")
            return False
        
        # Get event dates for this month
        event_dates = self.get_event_dates_for_month(events_file)
        month_days = monthrange(year, month)[1]
        
        # Process in two chunks
        chunks: List[Tuple[int, int]] = [(1, 15), (16, month_days)]

        # Prepare chunk metadata
        chunk_meta = []
        for (start_day, end_day) in chunks:
            part = 1 if start_day <= 15 else 2
            already_done = self.check_progress(year, month, part)
            chunk_days = set(range(start_day, end_day + 1))
            has_events = bool(chunk_days.intersection(event_dates))
            need_forecast = self.chunk_has_forecast(year, month, start_day, end_day)
            filename = f"forecast_{year}_{month:02d}_{start_day:02d}_{end_day:02d}.zip"
            dates_string = ','.join(self.get_chunk_dates(year, month, start_day, end_day))
            chunk_meta.append({
                'start': start_day,
                'end': end_day,
                'part': part,
                'has_events': has_events,
                'need_forecast': need_forecast,
                'filename': filename,
                'dates_string': dates_string,
                'already_done': already_done,
            })

        # Track all active tasks for proper cleanup
        active_tasks: set[asyncio.Task] = set()
        current_task: Optional[asyncio.Task] = None
        
        try:
            # Prefetch first chunk if needed
            if (
                chunk_meta[0]['has_events']
                and chunk_meta[0]['need_forecast']
                and chunk_meta[0]['dates_string']
                and not chunk_meta[0]['already_done']
            ):
                self.logger.info(f"Prefetching forecast for days {chunk_meta[0]['start']}-{chunk_meta[0]['end']}")
                current_task = asyncio.create_task(
                    self.downloader.download_all(chunk_meta[0]['dates_string'], chunk_meta[0]['filename'])
                )
                active_tasks.add(current_task)

            # Iterate chunks with cross-chunk prefetch
            for i, meta in enumerate(chunk_meta):
                start_day = meta['start']
                end_day = meta['end']
                part = meta['part']

                if meta['already_done']:
                    self.logger.info(
                        "Skipping days %s-%s; enriched_%d-%02d_part%d.csv already exists",
                        start_day,
                        end_day,
                        year,
                        month,
                        part,
                    )

                    if current_task is not None:
                        try:
                            await current_task
                        except Exception:
                            pass
                        finally:
                            active_tasks.discard(current_task)
                            current_task = None

                    if i + 1 < len(chunk_meta):
                        next_meta = chunk_meta[i + 1]
                        if (
                            next_meta['has_events']
                            and next_meta['need_forecast']
                            and next_meta['dates_string']
                            and not next_meta['already_done']
                        ):
                            self.logger.info(
                                "Prefetching next forecast for days %s-%s",
                                next_meta['start'],
                                next_meta['end'],
                            )
                            current_task = asyncio.create_task(
                                self.downloader.download_all(next_meta['dates_string'], next_meta['filename'])
                            )
                            active_tasks.add(current_task)
                        else:
                            current_task = None
                    else:
                        current_task = None

                    continue

                if not meta['has_events']:
                    # Do not prematurely skip; verify inside the chunk to avoid false negatives
                    self.logger.info(f"No pre-known events for days {start_day}-{end_day}; will verify within chunk")

                prefetched_path: Optional[Path] = None
                # Await current chunk download if needed
                if meta['need_forecast']:
                    if current_task is not None:
                        self.logger.info(f"Waiting for forecast for days {start_day}-{end_day}")
                        try:
                            ok = await current_task
                        except Exception as e:
                            self.logger.warning(f"Prefetch task failed for days {start_day}-{end_day}: {e}")
                            ok = False
                        finally:
                            # Remove completed task from active set
                            active_tasks.discard(current_task)
                        
                        if ok:
                            prefetched_path = self.temp_dir / meta['filename']
                        current_task = None
                    else:
                        # No prefetch started, will download inline inside process_chunk
                        prefetched_path = None

                # Start prefetch for next chunk while processing this one
                next_task: Optional[asyncio.Task] = None
                if i + 1 < len(chunk_meta):
                    next_meta = chunk_meta[i + 1]
                    if (
                        next_meta['has_events']
                        and next_meta['need_forecast']
                        and next_meta['dates_string']
                        and not next_meta['already_done']
                    ):
                        self.logger.info(f"Prefetching next forecast for days {next_meta['start']}-{next_meta['end']}")
                        next_task = asyncio.create_task(
                            self.downloader.download_all(next_meta['dates_string'], next_meta['filename'])
                        )
                        active_tasks.add(next_task)

                # Process current chunk (uses prefetched file if available)
                success = await self.process_chunk_async(
                    year, month, start_day, end_day, events_file,
                    prefetched_forecast_path=prefetched_path,
                )
                if not success:
                    self.logger.error(f"Failed to process chunk {start_day}-{end_day}")
                    # Cancel remaining tasks before returning
                    for task in active_tasks:
                        if not task.done():
                            task.cancel()
                    if active_tasks:
                        await asyncio.gather(*active_tasks, return_exceptions=True)
                    return False

                # Shift pipeline: next task becomes current task
                current_task = next_task
            
            # Combine monthly chunks if both parts exist
            self.try_combine_month_if_ready(year, month)
            
            self.logger.info(f"✅ Completed month {month}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in process_month_async: {e}", exc_info=True)
            # Cancel all active tasks
            for task in active_tasks:
                if not task.done():
                    task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            return False
        
        finally:
            # Ensure all tasks are cleaned up
            for task in list(active_tasks):
                if not task.done():
                    self.logger.warning(f"Cancelling orphaned task: {task}")
                    task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
    
    def _build_year_job_queue(self, year: int) -> deque:
        """Build a chronological queue of chunk jobs for an entire year.

        Each job includes metadata to determine if download is required and
        the forecast file name expected by the downloader.
        """
        q: deque = deque()
        for month in range(1, 13):
            events_file = self.events_dir / f"events_{year}_{month:02d}.csv"
            if not events_file.exists():
                continue
            month_days = monthrange(year, month)[1]
            chunks: List[Tuple[int, int]] = [(1, 15), (16, month_days)]
            event_dates = self.get_event_dates_for_month(events_file)
            for start_day, end_day in chunks:
                part = 1 if start_day <= 15 else 2
                already_done = self.check_progress(year, month, part)
                if already_done:
                    continue
                chunk_days = set(range(start_day, end_day + 1))
                has_events = bool(chunk_days.intersection(event_dates))
                dates_string = ','.join(self.get_chunk_dates(year, month, start_day, end_day))
                need_forecast = self.chunk_has_forecast(year, month, start_day, end_day)
                filename = f"forecast_{year}_{month:02d}_{start_day:02d}_{end_day:02d}.zip"
                q.append({
                    'year': year,
                    'month': month,
                    'start': start_day,
                    'end': end_day,
                    'part': part,
                    'events_file': events_file,
                    'has_events': has_events,
                    'need_forecast': need_forecast,
                    'dates_string': dates_string,
                    'filename': filename,
                })
        return q

    async def process_year_multi_key_async(self, year: int) -> bool:
        """Process a year using multiple CDS API keys with per-key submit-and-poll.

        - One in-flight job per key.
        - 1–15 must be requested before 16–rest for any month.
        - If 1–15 exists, 16–rest becomes eligible immediately.
        - Submit next job as soon as a key is free; enrichment runs in background tasks.
        """
        self.logger.info(f"Using multi-key submit/poll scheduler for year {year}")
        jobs = self._build_year_job_queue(year)
        if not jobs:
            self.logger.warning("No pending chunks for this year.")
            return True

        # Split jobs
        # Note: Schedule forecast downloads even if a chunk has no events; enrichment will skip if empty.
        enrich_only_jobs = [j for j in jobs if not (j['need_forecast'] and j['dates_string'])]
        download_jobs = [j for j in jobs if (j['need_forecast'] and j['dates_string'])]

        # Enrich-only first
        for j in enrich_only_jobs:
            self.logger.info(
                f"Enrich-only chunk (no forecast download): {j['year']}-{j['month']:02d} days {j['start']}-{j['end']}"
            )
            try:
                await self.process_chunk_async(j['year'], j['month'], j['start'], j['end'], j['events_file'], prefetched_forecast_path=None)
            except Exception as e:
                self.logger.warning(f"Failed to process enrich-only chunk {j['year']}-{j['month']:02d} part {j['part']}: {e}")

        if not download_jobs:
            for m in range(1, 13):
                self.try_combine_month_if_ready(year, m)
            self.logger.info(f"✅ Completed year {year}")
            return True

        # Month state (consider enriched CSV presence as already done)
        month_state: Dict[int, Dict[str, bool]] = {}
        for m in range(1, 13):
            last_day = monthrange(year, m)[1]
            p1 = self.temp_dir / f"forecast_{year}_{m:02d}_01_15.zip"
            p2 = self.temp_dir / f"forecast_{year}_{m:02d}_16_{last_day:02d}.zip"
            en1 = self.output_dir / f"enriched_{year}-{m:02d}_part1.csv"
            en2 = self.output_dir / f"enriched_{year}-{m:02d}_part2.csv"
            combined = self.output_dir / f"enriched_{year}-{m:02d}.csv"
            if combined.exists():
                p1_present = True
                p2_present = True
            else:
                p1_present = (p1.exists() and p1.stat().st_size > 0) or en1.exists()
                p2_present = (p2.exists() and p2.stat().st_size > 0) or en2.exists()
            month_state[m] = {
                'p1_present': p1_present,
                'p2_present': p2_present,
                'p1_submitted': False,
                'p2_submitted': False,
            }

        jobs_by_part: Dict[Tuple[int, int], Dict] = {(j['month'], j['part']): j for j in download_jobs}
        sched_lock = asyncio.Lock()
        enrich_tasks: set[asyncio.Task] = set()

        def can_do_p2(m: int) -> bool:
            st = month_state.get(m, {})
            return bool(st.get('p1_present'))

        async def pick_next_job() -> Optional[Dict]:
            async with sched_lock:
                # Prefer P2 when P1 present
                for m in range(1, 13):
                    st = month_state.get(m)
                    if not st:
                        continue
                    if can_do_p2(m) and (not st['p2_present']) and (not st['p2_submitted']):
                        j = jobs_by_part.get((m, 2))
                        if j:
                            st['p2_submitted'] = True
                            return j
                # Otherwise P1
                for m in range(1, 13):
                    st = month_state.get(m)
                    if not st:
                        continue
                    if (not st['p1_present']) and (not st['p1_submitted']):
                        j = jobs_by_part.get((m, 1))
                        if j:
                            st['p1_submitted'] = True
                            return j
                return None

        async def key_worker(cred_index: int):
            while True:
                job = await pick_next_job()
                if job is None:
                    return
                # Submit and poll nonblocking
                try:
                    handle = await self.downloader.submit_nonblocking(job['dates_string'], job['filename'], cred_index)
                except Exception as e:
                    self.logger.warning(
                        f"[Key {cred_index+1}] Submission failed for {job['year']}-{job['month']:02d} days {job['start']}-{job['end']}: {e}"
                    )
                    async with sched_lock:
                        st = month_state.get(job['month'])
                        if st:
                            if job['part'] == 1:
                                st['p1_submitted'] = False
                            else:
                                st['p2_submitted'] = False
                    await asyncio.sleep(2)
                    continue

                ok = await self.downloader.poll_and_download_nonblocking(handle)
                async with sched_lock:
                    st = month_state.get(job['month'])
                    if st and ok:
                        if job['part'] == 1:
                            st['p1_present'] = True
                        else:
                            st['p2_present'] = True

                # Enrich in background so key can submit next immediately
                pref_path = self.temp_dir / job['filename'] if ok and (self.temp_dir / job['filename']).exists() else None
                et = asyncio.create_task(
                    self.process_chunk_async(job['year'], job['month'], job['start'], job['end'], job['events_file'], prefetched_forecast_path=pref_path)
                )
                enrich_tasks.add(et)
                # prune finished
                done = {t for t in enrich_tasks if t.done()}
                if done:
                    for t in done:
                        enrich_tasks.discard(t)

        credentials = config.get_api_credentials()
        key_count = max(1, len(credentials) if credentials else 1)
        workers = [asyncio.create_task(key_worker(i)) for i in range(key_count)]

        try:
            await asyncio.gather(*workers)
            if enrich_tasks:
                await asyncio.gather(*enrich_tasks, return_exceptions=True)
            for m in range(1, 13):
                self.try_combine_month_if_ready(year, m)
            self.logger.info(f"✅ Completed year {year} with multi-key submit/poll scheduler")
            return True
        except Exception as e:
            self.logger.error(f"Multi-key scheduling failed: {e}", exc_info=True)
            for w in workers:
                if not w.done():
                    w.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            return False
    
    def try_combine_month_if_ready(self, year: int, month: int):
        """If both part1 and part2 CSVs exist, combine into a single monthly CSV immediately.

        Does nothing if only one or none exist to avoid warnings/partial renames.
        Idempotent: if the combined file already exists, it will not re-create.
        """
        part1_file = self.output_dir / f"enriched_{year}-{month:02d}_part1.csv"
        part2_file = self.output_dir / f"enriched_{year}-{month:02d}_part2.csv"
        combined_file = self.output_dir / f"enriched_{year}-{month:02d}.csv"

        if combined_file.exists():
            return
        if not (part1_file.exists() and part2_file.exists()):
            return

        dfs = []
        for file in (part1_file, part2_file):
            with open(file, 'r') as f:
                dfs.append(pd.read_csv(f))
        combined_df = pd.concat(dfs, ignore_index=True)
        with open(combined_file, 'w') as f:
            combined_df.to_csv(f, index=False)
        # Remove chunk files
        part1_file.unlink(missing_ok=True)
        part2_file.unlink(missing_ok=True)
        self.logger.info(f"✅ Combined chunks into {combined_file.name}")
    async def process_year_async(self, year: int) -> bool:
        """Process a year using multi-key scheduler when available."""
        self.logger.info(f"Processing year {year}")
        try:
            # Cancel any queued/running EWDS tasks from previous runs before scheduling new ones
            try:
                await self.downloader.cleanup_pending_ewds_tasks()
            except Exception as e:
                self.logger.warning(f"EWDS cleanup failed (continuing): {e}")
            creds = config.get_api_credentials()
            use_multi = bool(creds) and len(creds) >= 2
            if use_multi:
                result = await self.process_year_multi_key_async(year)
            else:
                result = await self.process_year_simple_async(year)
            return result
        except Exception as e:
            self.logger.error(f"Error processing year {year}: {e}", exc_info=True)
            return False
        finally:
            # Clean up temp directory
            if not self._temp_dir_is_external and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.logger.info("Cleaned up temporary directory")

async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(
        description="Refactored Pipeline Orchestrator with asyncio",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--year",
        type=int,
        nargs='+',
        required=True,
        help="Year(s) to process"
    )
    
    parser.add_argument(
        "--events-dir",
        type=str,
        required=True,
        help="Directory containing monthly event files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for enriched files"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    events_dir = Path(args.events_dir)
    if not events_dir.exists():
        logging.error(f"Events directory not found: {events_dir}")
        return 1
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(args.events_dir, args.output_dir)
    
    orchestrator.logger.info("=" * 60)
    orchestrator.logger.info("Refactored Pipeline Orchestrator")
    orchestrator.logger.info(f"Years to process: {args.year}")
    orchestrator.logger.info(f"Events directory: {args.events_dir}")
    orchestrator.logger.info(f"Output directory: {args.output_dir}")
    orchestrator.logger.info("=" * 60)
    orchestrator.logger.info("Configuration:")
    orchestrator.logger.info(f"• Max concurrent downloads: {config.max_concurrent_downloads}")
    orchestrator.logger.info(f"• Max retries: {config.max_retries}")
    orchestrator.logger.info(f"• Bad data periods: {len(config.bad_dates)}")
    orchestrator.logger.info(f"• GloFAS transition date: {config.glofas_transition_date}")
    orchestrator.logger.info("=" * 60)
    orchestrator.logger.info("SAFETY FEATURES ENABLED:")
    orchestrator.logger.info(f"• Bad data periods filtered: {len(config.bad_dates)} known periods")
    for start_date, end_date in config.bad_dates:
        orchestrator.logger.info(f"  - {start_date} to {end_date}")
    orchestrator.logger.info("• Pre-Nov 2019 events: No GloFAS forecast features available")
    orchestrator.logger.info("• All events: Only core GloFAS forecast features retained")
    orchestrator.logger.info("=" * 60)
    
    try:
        # Process each year
        for year in args.year:
            success = await orchestrator.process_year_async(year)
            if not success:
                orchestrator.logger.error(f"Failed to process year {year}")
        
        orchestrator.logger.info("✅ All processing completed")
        return 0
        
    except Exception as e:
        orchestrator.logger.error(f"Processing failed: {e}", exc_info=True)
        return 1


def main():
    """Entry point."""
    # Set up asyncio debug mode if configured
    if config.asyncio_debug:
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.set_debug(True)
    
    # Run async main
    exit_code = asyncio.run(main_async())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()



