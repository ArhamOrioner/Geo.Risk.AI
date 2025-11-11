#!/usr/bin/env python3
"""
Refactored Asynchronous GloFAS Forecast Downloader
Uses asyncio with Semaphore for rate limiting and proper resource management.
"""

import os
import sys
import argparse
import logging
import json
import asyncio
import aiofiles
import zipfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from itertools import cycle
from contextlib import asynccontextmanager

import cdsapi
from config import config  # Import centralized configuration

class AsyncGloFASDownloader:
    def __init__(self, output_dir: str, max_concurrent: int = None):
        """Initialize the downloader with configuration."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if config.download_dir:
            self.download_dir = Path(config.download_dir)
        else:
            self.download_dir = self.output_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Use config for settings
        self.max_concurrent = max_concurrent or config.max_concurrent_downloads
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.max_retries = config.max_retries
        self.initial_retry_delay = config.initial_retry_delay
        self.max_retry_delay = config.max_retry_delay
        
        # Setup logging
        self.setup_logging()
        
        # Track active downloads
        self.active_downloads = set()
        self.completed_downloads = []
        self.failed_downloads = []
        self._credential_pool: List[Dict] = []
        self._credential_cycle = None
        self._credential_lock: Optional[asyncio.Lock] = None
        self._credential_index = 0
        self._refresh_credentials()
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / config.log_file_name
        
        # Create formatter
        formatter = logging.Formatter(config.log_format)
        
        # File handler
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
        # Silence cdsapi info/progress logs globally
        try:
            logging.getLogger('cdsapi').setLevel(logging.WARNING)
        except Exception:
            pass
    
    async def check_environment(self) -> bool:
        """Check if required dependencies and configuration are available."""
        self.logger.info("Checking environment...")
        critical_errors = []
        warnings = []
        
        # Check Python version
        import sys
        py_version = sys.version_info
        if py_version < (3, 8):
            critical_errors.append(f"Python {py_version.major}.{py_version.minor} detected. Python 3.8+ required.")
        else:
            self.logger.info(f"✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        
        # Check CDS API configuration
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            client = await loop.run_in_executor(None, lambda: cdsapi.Client())
            self.logger.info("✅ CDS API client initialized successfully")
            
            # Check API credentials
            if hasattr(client, 'url') and hasattr(client, 'key'):
                if client.url and client.key:
                    self.logger.info(f"   API URL: {client.url}")
                    self.logger.info(f"   API Key: {'*' * 8}...{client.key[-4:] if len(client.key) > 4 else '****'}")
                else:
                    warnings.append("CDS API credentials may be incomplete")
        except Exception as e:
            critical_errors.append(f"CDS API client failed: {e}")
            self.logger.error("Please ensure:")
            self.logger.error("1. .cdsapirc file is configured or environment variables are set")
            self.logger.error("2. You have accepted the GloFAS dataset license")
            self.logger.error("3. cdsapi package is installed: pip install cdsapi")
        
        # Check cfgrib availability and version
        try:
            import cfgrib
            self.logger.info("✅ cfgrib package available")
            if hasattr(cfgrib, '__version__'):
                self.logger.info(f"   cfgrib version: {cfgrib.__version__}")
        except ImportError:
            warnings.append("cfgrib package not found - GRIB reading may fail")
            self.logger.warning("Install with: pip install cfgrib")
        
        # Check ecCodes availability and version
        try:
            import eccodes
            self.logger.info("✅ ecCodes package available")
            # Try to get version
            try:
                version = eccodes.codes_get_api_version()
                self.logger.info(f"   ecCodes version: {version}")
            except:
                pass
            
            # Check ecCodes data directory
            try:
                import os
                eccodes_dir = os.environ.get('ECCODES_DIR')
                if eccodes_dir:
                    self.logger.info(f"   ECCODES_DIR: {eccodes_dir}")
            except:
                pass
        except ImportError:
            warnings.append("ecCodes not found - GRIB processing may be limited")
            self.logger.warning("Install with: conda install -c conda-forge eccodes python-eccodes")
            self.logger.warning("Or: pip install eccodes-python")
        
        # Check xarray availability and version
        try:
            import xarray
            self.logger.info("✅ xarray package available")
            if hasattr(xarray, '__version__'):
                self.logger.info(f"   xarray version: {xarray.__version__}")
        except ImportError:
            critical_errors.append("xarray not found - required for GRIB processing")
            self.logger.error("Install with: pip install xarray")
        
        # Check pandas availability
        try:
            import pandas
            self.logger.info("✅ pandas package available")
            if hasattr(pandas, '__version__'):
                self.logger.info(f"   pandas version: {pandas.__version__}")
        except ImportError:
            critical_errors.append("pandas not found - required for data processing")
            self.logger.error("Install with: pip install pandas")
        
        # Check numpy availability
        try:
            import numpy
            self.logger.info("✅ numpy package available")
            if hasattr(numpy, '__version__'):
                self.logger.info(f"   numpy version: {numpy.__version__}")
        except ImportError:
            critical_errors.append("numpy not found - required for numerical operations")
            self.logger.error("Install with: pip install numpy")
        
        # Check disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage(self.output_dir)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 10:
                warnings.append(f"Low disk space: {free_gb:.1f} GB free")
            else:
                self.logger.info(f"✅ Disk space: {free_gb:.1f} GB free")
        except:
            pass
        
        # Summary
        self.logger.info("="*60)
        if critical_errors:
            self.logger.error(f"❌ {len(critical_errors)} critical errors found:")
            for error in critical_errors:
                self.logger.error(f"   - {error}")
            return False

    def _refresh_credentials(self):
        """Refresh credential cycle from configuration if not already available."""
        creds = config.get_api_credentials()
        if not creds:
            self._credential_pool = []
            self._credential_cycle = None
            self._credential_lock = None
            self._credential_index = 0
            return

        if self._credential_pool != creds or self._credential_cycle is None:
            self._credential_pool = creds
            self._credential_cycle = cycle(self._credential_pool)
            self._credential_lock = asyncio.Lock()
            self._credential_index = 0

    async def _get_client(self) -> cdsapi.Client:
        """Create a cdsapi.Client using rotating credentials if configured."""
        self._refresh_credentials()

        if not self._credential_cycle:
            return cdsapi.Client(quiet=True, progress=False)

        async with self._credential_lock:
            creds = next(self._credential_cycle)
            idx = self._credential_index
            self._credential_index = (self._credential_index + 1) % len(self._credential_pool)

        key_preview = creds.get("key", "")
        if isinstance(key_preview, str) and len(key_preview) > 8:
            key_preview = f"{key_preview[:4]}…{key_preview[-4:]}"
        elif not key_preview:
            key_preview = "<empty>"

        self.logger.info(
            "Using CDS credential slot #%d (key %s)",
            idx + 1,
            key_preview
        )
        return cdsapi.Client(url=creds.get("url"), key=creds.get("key"), quiet=True, progress=False)

    def _get_client_by_index(self, index: int, nonblocking: bool = False) -> cdsapi.Client:
        """Return a cdsapi.Client pinned to a specific credential index.
        
        If nonblocking=True, returns a client with wait_until_complete=False (submit-only, poll later).
        Falls back to default cdsapi.Client() if no credentials are configured.
        """
        self._refresh_credentials()
        if not self._credential_pool:
            # Default client with requested blocking/nonblocking behavior
            return cdsapi.Client(wait_until_complete=not nonblocking, quiet=True, progress=False)

        # Clamp index to valid range
        idx = max(0, min(int(index), len(self._credential_pool) - 1))
        creds = self._credential_pool[idx]

        key_preview = creds.get("key", "")
        if isinstance(key_preview, str) and len(key_preview) > 8:
            key_preview = f"{key_preview[:4]}…{key_preview[-4:]}"
        elif not key_preview:
            key_preview = "<empty>"

        self.logger.info(
            "Using PINNED CDS credential slot #%d (key %s)",
            idx + 1,
            key_preview,
        )
        return cdsapi.Client(
            url=creds.get("url"), key=creds.get("key"), wait_until_complete=not nonblocking, quiet=True, progress=False
        )

    async def submit_nonblocking(self, dates_string: str, filename: str, cred_index: int) -> Dict:
        """Submit a forecast request without waiting; return a job handle.

        The returned dict includes: 'result', 'request', 'target', 'filename', 'cred_index'.
        """
        # Parse dates and build single-month request
        dates = self.parse_dates(dates_string)
        if not dates:
            raise ValueError("No valid dates provided for submission")
        # If dates cross hydrological model switch, submit composite subjobs
        model_groups = self.split_dates_by_model(dates)
        loop = asyncio.get_event_loop()
        client = self._get_client_by_index(cred_index, nonblocking=True)
        target = self.download_dir / filename

        if len(model_groups) > 1:
            self.logger.info(
                f"[Key {cred_index+1}] 📤 Submitting composite request (model switch within chunk) for {filename}"
            )
            subjobs: List[Dict] = []
            for i, grp in enumerate(model_groups, start=1):
                req_i = self.build_forecast_request(grp['dates'], grp['model'])
                part_name = f"{filename}.part{i}.zip"
                part_target = self.download_dir / part_name
                res_i = await loop.run_in_executor(
                    None,
                    lambda req=req_i: client.retrieve("cems-glofas-forecast", req),
                )
                # Log initial state
                try:
                    st = getattr(res_i, "reply", {}).get("state")
                    self.logger.info(f"[Key {cred_index+1}]   ↳ Subjob {i} initial state: {st}")
                except Exception:
                    pass
                # Register each subjob
                try:
                    reply = getattr(res_i, "reply", {}) or {}
                    rid = reply.get("request_id") or reply.get("task_id")
                    if rid:
                        await self._register_pending_task({
                            "request_id": rid,
                            "filename": part_name,
                            "cred_index": int(cred_index),
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to register pending subtask for {part_name}: {e}")
                subjobs.append({
                    "result": res_i,
                    "request": req_i,
                    "target": str(part_target),
                    "filename": part_name,
                    "cred_index": cred_index,
                })
            return {
                "composite": True,
                "filename": filename,
                "merge_target": str(target),
                "subjobs": subjobs,
            }

        # Single-model case
        request = self.build_forecast_request(dates, self.get_model_for_date(dates[0]))
        self.logger.info(
            f"[Key {cred_index+1}] 📤 Submitting non-blocking request for {filename}"
        )
        result_obj = await loop.run_in_executor(
            None,
            lambda: client.retrieve("cems-glofas-forecast", request),
        )
        # Log initial state if available
        try:
            state = getattr(result_obj, "reply", {}).get("state")
            self.logger.info(f"[Key {cred_index+1}]   ↳ Initial state: {state}")
        except Exception:
            pass

        # Extract request_id and persist task in registry for cleanup on next run
        try:
            reply = getattr(result_obj, "reply", {}) or {}
            request_id = reply.get("request_id") or reply.get("task_id")
            if request_id:
                await self._register_pending_task({
                    "request_id": request_id,
                    "filename": filename,
                    "cred_index": int(cred_index),
                })
        except Exception as e:
            self.logger.warning(f"Failed to register pending task for {filename}: {e}")

        return {
            "result": result_obj,
            "request": request,
            "target": str(target),
            "filename": filename,
            "cred_index": cred_index,
        }

    async def poll_and_download_nonblocking(self, job_handle: Dict) -> bool:
        """Poll a submitted job until completed, then download to target path.

        Returns True on success, False on failure. Uses config.download_check_interval between polls.
        """
        loop = asyncio.get_event_loop()
        # Handle composite job: poll each subjob and merge parts
        if job_handle.get("composite"):
            subjobs: List[Dict] = job_handle.get("subjobs", [])
            if not subjobs:
                return False
            cred_index = int(subjobs[0].get("cred_index", 0))
            parts: List[Path] = []
            for sj in subjobs:
                ok = await self.poll_and_download_nonblocking(sj)
                if not ok:
                    return False
                parts.append(Path(sj["target"]))
            # Merge parts into final
            final_target = Path(job_handle["merge_target"])
            try:
                await loop.run_in_executor(None, self._merge_zips, parts, final_target)
                if final_target.exists() and final_target.stat().st_size > 0:
                    self.logger.info(f"[Key {cred_index+1}] ✅ Downloaded {final_target.name}")
                    # Cleanup part files
                    for p in parts:
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    return True
                else:
                    self.logger.error(f"[Key {cred_index+1}] Merge failed or empty: {final_target}")
                    return False
            except Exception as e:
                self.logger.error(f"[Key {cred_index+1}] Error merging parts into {final_target.name}: {e}")
                return False

        # Single subjob
        result = job_handle["result"]
        target = Path(job_handle["target"])
        cred_index = int(job_handle["cred_index"])

        poll_seconds = max(5, int(config.download_check_interval))
        attempt = 0
        last_state = None
        last_log_ts = 0.0
        while True:
            attempt += 1
            # Update state
            try:
                await loop.run_in_executor(None, result.update)
            except Exception as e:
                self.logger.warning(
                    f"[Key {cred_index+1}] update() failed on attempt {attempt}: {e}"
                )

            reply = getattr(result, "reply", {}) or {}
            state = reply.get("state", "unknown")
            # Only log on state change (or every 5 minutes as a heartbeat)
            now = time.time()
            if state != last_state or (now - last_log_ts) >= 300:
                self.logger.info(
                    f"[Key {cred_index+1}] [{job_handle['filename']}] State: {state}"
                )
                last_state = state
                last_log_ts = now

            if state == "completed":
                try:
                    # Prefer our own progress-aware downloader using URL if available
                    url = None
                    try:
                        reply = getattr(result, "reply", {}) or {}
                        url = reply.get("location") or getattr(result, "location", None)
                    except Exception:
                        url = None
                    used_custom = False
                    if url:
                        prefix = f"[Key {cred_index+1}] {job_handle['filename']}"
                        try:
                            await self._download_with_progress(url, target, prefix)
                            used_custom = True
                        except Exception as ex:
                            self.logger.warning(f"Custom downloader failed, falling back: {ex}")
                            used_custom = False
                    if not used_custom:
                        # Fallback to cdsapi download (progress disabled)
                        if hasattr(result, "download"):
                            await loop.run_in_executor(None, lambda: result.download(str(target)))
                        else:
                            client = self._get_client_by_index(cred_index)
                            await loop.run_in_executor(
                                None,
                                client.retrieve,
                                "cems-glofas-forecast",
                                job_handle["request"],
                                str(target),
                            )
                    if target.exists() and target.stat().st_size > 0:
                        self.logger.info(
                            f"[Key {cred_index+1}] ✅ Downloaded {target.name}"
                        )
                        # Remove from registry
                        await self._unregister_pending_task_by_filename(job_handle["filename"])
                        return True
                    self.logger.error(
                        f"[Key {cred_index+1}] Download reported completed but file missing/empty: {target}"
                    )
                    await self._unregister_pending_task_by_filename(job_handle["filename"])
                    return False
                except Exception as e:
                    self.logger.error(
                        f"[Key {cred_index+1}] Download error for {target.name}: {e}"
                    )
                    await self._unregister_pending_task_by_filename(job_handle["filename"])
                    return False

            if state == "failed":
                self.logger.error(
                    f"[Key {cred_index+1}] ❌ Server reported failure for {job_handle['filename']}"
                )
                await self._unregister_pending_task_by_filename(job_handle["filename"])
                return False

            # Not done yet
            await asyncio.sleep(poll_seconds)

    def _merge_zips(self, part_paths: List[Path], final_path: Path):
        """Merge multiple zip files into one. Skips duplicate filenames."""
        names_added = set()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(final_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for p in part_paths:
                if not p.exists() or p.stat().st_size == 0:
                    continue
                with zipfile.ZipFile(p, 'r') as zin:
                    for info in zin.infolist():
                        # Avoid duplicates
                        if info.filename in names_added:
                            continue
                        with zin.open(info.filename) as src:
                            data = src.read()
                        zout.writestr(info.filename, data)
                        names_added.add(info.filename)

    def _print_progress_bar(self, prefix: str, downloaded: int, total: int, done: bool = False):
        try:
            if total and total > 0:
                pct = max(0, min(100, int(downloaded * 100 / total)))
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = '#' * filled + '-' * (bar_len - filled)
                msg = f"\r{prefix} [{bar}] {pct}% ({downloaded/1e6:.1f}MB/{total/1e6:.1f}MB)"
            else:
                msg = f"\r{prefix} [downloading] {downloaded/1e6:.1f}MB"
            sys.stdout.write(msg)
            sys.stdout.flush()
            if done:
                sys.stdout.write("\n")
                sys.stdout.flush()
        except Exception:
            pass

    async def _download_with_progress(self, url: str, target: Path, prefix: str):
        """Stream download with a minimal console progress bar (console-only).

        Updates at most every ~2 seconds to avoid spam. Falls back silently if headers missing.
        """
        loop = asyncio.get_event_loop()

        def _do_download():
            try:
                try:
                    import requests  # Local import to avoid hard dependency at module load
                except Exception as err:
                    raise RuntimeError(f"requests not available: {err}")
                with requests.get(url, stream=True, timeout=None) as r:
                    r.raise_for_status()
                    total = int(r.headers.get('Content-Length') or 0)
                    downloaded = 0
                    last = time.time()
                    last_logged_pct = -10  # Log every 10%
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with open(target, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            # Console progress bar (TTY only) every ~2 seconds
                            if sys.stdout and sys.stdout.isatty() and (now - last >= 2 or (total and downloaded >= total)):
                                self._print_progress_bar(prefix, downloaded, total, done=False)
                                last = now
                            # Coarse log progress every 10% (or at completion)
                            if total and total > 0:
                                pct = int(downloaded * 100 / total)
                                if pct >= last_logged_pct + 10 or pct == 100:
                                    self.logger.info(f"{prefix} progress: {pct}% ({downloaded/1e6:.1f}MB/{total/1e6:.1f}MB)")
                                    last_logged_pct = pct - (pct % 10)
                    if sys.stdout and sys.stdout.isatty():
                        self._print_progress_bar(prefix, downloaded, total, done=True)
            except Exception as e:
                raise e

        await loop.run_in_executor(None, _do_download)

    # ===== Pending task registry & cleanup (EWDS) =====
    def _task_registry_path(self) -> Path:
        return self.output_dir / ".pending_ewds_tasks.json"

    async def _register_pending_task(self, entry: Dict):
        path = self._task_registry_path()
        try:
            # Load existing
            data = []
            if path.exists():
                async with aiofiles.open(path, 'r') as f:
                    try:
                        text = await f.read()
                        data = json.loads(text) if text.strip() else []
                    except Exception:
                        data = []
            # Append new (avoid duplicates by request_id)
            rid = entry.get("request_id")
            if rid and not any(item.get("request_id") == rid for item in data):
                data.append(entry)
                async with aiofiles.open(path, 'w') as f:
                    await f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.warning(f"Failed to update pending task registry: {e}")

    async def _unregister_pending_task_by_filename(self, filename: str):
        path = self._task_registry_path()
        try:
            if not path.exists():
                return
            async with aiofiles.open(path, 'r') as f:
                try:
                    text = await f.read()
                    data = json.loads(text) if text.strip() else []
                except Exception:
                    data = []
            new_data = [item for item in data if item.get("filename") != filename]
            if new_data != data:
                async with aiofiles.open(path, 'w') as f:
                    await f.write(json.dumps(new_data, indent=2))
        except Exception as e:
            self.logger.warning(f"Failed to unregister pending task for {filename}: {e}")

    async def cleanup_pending_ewds_tasks(self):
        """Cancel any pending EWDS tasks stored in the registry (accepted/queued/running).

        Only attempts deletion; completed tasks are not stored or will fail fast. This should be
        called at pipeline startup before new submissions to avoid duplicate queues.
        """
        path = self._task_registry_path()
        if not path.exists():
            return
        try:
            async with aiofiles.open(path, 'r') as f:
                try:
                    text = await f.read()
                    tasks = json.loads(text) if text.strip() else []
                except Exception:
                    tasks = []
            if not tasks:
                return
            self.logger.info(f"Attempting to cancel {len(tasks)} pending EWDS task(s) from previous run(s)")
            loop = asyncio.get_event_loop()
            for item in tasks:
                rid = item.get("request_id")
                cred_index = int(item.get("cred_index", 0))
                if not rid:
                    continue
                try:
                    # Reconstruct a Result handle and delete it
                    client = self._get_client_by_index(cred_index, nonblocking=True)
                    # Import internal Result class lazily
                    from cdsapi.api import Result as _CdsResult  # type: ignore
                    res = _CdsResult(client, {"request_id": rid})
                    await loop.run_in_executor(None, res.delete)
                    self.logger.info(f"🗑️ Cancelled EWDS task {rid} (key #{cred_index+1})")
                except Exception as e:
                    self.logger.warning(f"Failed to cancel EWDS task {rid}: {e}")
            # Clear registry after attempts (avoid infinite retries)
            async with aiofiles.open(path, 'w') as f:
                await f.write(json.dumps([], indent=2))
        except Exception as e:
            self.logger.warning(f"Cleanup of pending EWDS tasks failed: {e}")
    
    def parse_dates(self, dates_string: str) -> List[datetime]:
        """Parse comma-separated dates string into list of datetime objects."""
        dates = []
        for date_str in dates_string.split(','):
            try:
                date_obj = datetime.strptime(date_str.strip(), '%Y-%m-%d')
                # Check if it's a bad date
                if not config.is_bad_date(date_obj.date()):
                    dates.append(date_obj)
                else:
                    self.logger.warning(f"Skipping bad date: {date_str}")
            except ValueError as e:
                self.logger.error(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")
                raise
        
        if not dates:
            raise ValueError("No valid dates provided")
        
        dates.sort()
        return dates
    
    def get_model_for_date(self, date: datetime) -> str:
        """Determine which model to use based on date."""
        version = config.get_glofas_version(date.date())
        if version == "3.1":
            return "htessel_lisflood"
        else:
            return "lisflood"
    
    def split_dates_by_model(self, dates: List[datetime]) -> List[Dict]:
        """Split dates into chunks based on model transition."""
        if not dates:
            return []
        
        transition_date = datetime.combine(config.glofas_transition_date, datetime.min.time())
        min_date = min(dates)
        max_date = max(dates)
        
        if min_date < transition_date <= max_date:
            # Split by transition date
            pre_transition = [d for d in dates if d < transition_date]
            post_transition = [d for d in dates if d >= transition_date]
            
            chunks = []
            if pre_transition:
                chunks.append({
                    'dates': pre_transition,
                    'model': 'htessel_lisflood'
                })
            if post_transition:
                chunks.append({
                    'dates': post_transition,
                    'model': 'lisflood'
                })
            return chunks
        else:
            # No transition needed
            model = self.get_model_for_date(dates[0])
            return [{'dates': dates, 'model': model}]

    def chunk_dates(self, dates: List[datetime]) -> List[Dict]:
        """Chunk dates by model and by month to satisfy API constraints.

        Returns a list of dicts with keys: {'dates': List[datetime], 'model': str}
        """
        # First split across model transition boundaries
        model_groups = self.split_dates_by_model(dates)
        final_chunks: List[Dict] = []
        for group in model_groups:
            grouped: Dict[Tuple[int, int], List[datetime]] = {}
            for d in group['dates']:
                key = (d.year, d.month)
                grouped.setdefault(key, []).append(d)
            # Within each (year, month), keep all requested days together
            for (y, m), dlist in grouped.items():
                dlist.sort()
                final_chunks.append({'dates': dlist, 'model': group['model']})
        return final_chunks
    
    def build_forecast_request(self, dates: List[datetime], model: str) -> Dict:
        """Build the forecast request dictionary."""
        # Validate that all dates are from the same year and month
        years = {d.year for d in dates}
        months = {d.month for d in dates}
        if len(years) != 1 or len(months) != 1:
            raise ValueError(f"Chunk must contain dates from single year/month. Found: {years}, {months}")
        
        # Convert dates to day strings
        days = [f"{d.day:02d}" for d in dates]
        year = dates[0].year
        month = dates[0].month
        
        # Determine product type based on date
        product_type = config.get_product_type(dates[0].date())
        
        request = {
            "system_version": ["operational"],
            "hydrological_model": [model],
            "product_type": [
                "control_forecast",
                "ensemble_perturbed_forecasts"
            ],
            "variable": ["river_discharge_in_the_last_24_hours"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": days,
            "leadtime_hour": ["24"],
            "format": "grib",
        }
        
        return request
    
    @asynccontextmanager
    async def rate_limited(self):
        """Context manager for rate limiting."""
        async with self.semaphore:
            yield
    
    async def download_with_retry(self, request: Dict, output_path: Path, chunk_info: str, cred_index: Optional[int] = None) -> bool:
        """Download with exponential backoff retry."""
        retry_delay = self.initial_retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.rate_limited():
                    self.logger.info(f"Downloading {chunk_info} (attempt {attempt + 1}/{self.max_retries + 1})")
                    
                    # Run CDS API call in executor (it's synchronous)
                    loop = asyncio.get_event_loop()
                    if cred_index is not None:
                        client = self._get_client_by_index(cred_index)
                    else:
                        client = await self._get_client()
                    
                    # Add timeout to prevent hanging
                    retrieve_coro = loop.run_in_executor(
                        None,
                        client.retrieve,
                        "cems-glofas-forecast",
                        request,
                        str(output_path)
                    )
                    if config.request_timeout:
                        result = await asyncio.wait_for(
                            retrieve_coro,
                            timeout=config.request_timeout
                        )
                    else:
                        result = await retrieve_coro
                    
                    self.logger.info(f"✅ Successfully downloaded: {output_path.name}")
                    return True
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"⏱️ Download timeout for {chunk_info}")
                if attempt < self.max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)
                    
            except Exception as e:
                error_msg = str(e).lower()
                
                # Handle specific errors
                if "429" in error_msg or "rate limit" in error_msg:
                    self.logger.warning(f"⚠️ Rate limit hit for {chunk_info}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
                elif "401" in error_msg or "unauthorized" in error_msg:
                    self.logger.error(f"❌ Authentication failed: {chunk_info}")
                    return False
                elif "403" in error_msg or "forbidden" in error_msg:
                    self.logger.error(f"❌ Access forbidden: {chunk_info}")
                    return False
                else:
                    self.logger.warning(f"⚠️ Download failed for {chunk_info}: {e}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, self.max_retry_delay)
        
        self.logger.error(f"❌ Failed to download {chunk_info} after {self.max_retries} retries")
        return False
    
    async def download_chunk(self, chunk: Dict, filename: str, chunk_num: int, total_chunks: int, cred_index: Optional[int] = None) -> Tuple[str, bool]:
        """Download a single chunk."""
        dates = chunk['dates']
        model = chunk['model']
        
        # Build request
        request = self.build_forecast_request(dates, model)
        
        # Determine output filename
        if total_chunks > 1:
            base_name = Path(filename).stem
            chunk_filename = f"{base_name}_chunk_{chunk_num}_{model}.zip"
        else:
            chunk_filename = filename
        
        output_path = self.download_dir / chunk_filename
        
        # Create chunk info string for logging
        date_range = f"{min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"
        chunk_info = f"{chunk_filename} ({date_range}, {model})"
        
        # Download with retry (optionally pinned to a specific credential slot)
        success = await self.download_with_retry(request, output_path, chunk_info, cred_index=cred_index)
        
        if success:
            self.completed_downloads.append(chunk_filename)
        else:
            self.failed_downloads.append(chunk_filename)
        
        return chunk_filename, success
    
    async def write_manifest(self, filename: str, chunk_status: Dict[str, Dict[str, str]], status: str):
        """Write manifest file asynchronously with per-chunk status mapping."""
        successful = sum(1 for v in chunk_status.values() if v.get('status') == 'completed')
        total = len(chunk_status)
        manifest = {
            "requested": filename,
            "status": status,
            "chunks": chunk_status,
            "submitted": True,
            "timestamp": datetime.utcnow().isoformat(),
            "total_chunks": total,
            "successful_chunks": successful,
            "failed_chunks": [k for k, v in chunk_status.items() if v.get('status') != 'completed'],
            "config": {
                "max_concurrent": self.max_concurrent,
                "max_retries": self.max_retries,
                "glofas_version_transition": config.glofas_transition_date.isoformat()
            }
        }
        
        manifest_path = self.output_dir / (filename + config.manifest_extension)
        temp_manifest_path = manifest_path.with_suffix('.tmp')
        
        try:
            # Write asynchronously
            async with aiofiles.open(temp_manifest_path, 'w') as f:
                await f.write(json.dumps(manifest, indent=2))
            
            # Atomic rename
            temp_manifest_path.rename(manifest_path)
            self.logger.info(f"✅ Created manifest: {manifest_path.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create manifest: {e}")
            if temp_manifest_path.exists():
                temp_manifest_path.unlink()
    
    def check_manifest_status(self, filename: str) -> Dict[str, any]:
        """Check download status from manifest file."""
        manifest_path = self.output_dir / f"{filename}{config.manifest_extension}"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read manifest: {e}")
        return {}
    
    async def download_all(self, dates_string: str, filename: str, cred_index: Optional[int] = None) -> bool:
        """Download all dates and combine into a single ZIP file named `filename`."""
        self.logger.info(f"Starting download for {filename}")
        
        # Check manifest for previous status
        manifest = self.check_manifest_status(filename)
        if manifest.get('status') == 'completed':
            self.logger.info(f"Download already completed (from manifest): {filename}")
            return True
        
        # Parse and chunk dates
        dates = self.parse_dates(dates_string)
        if not dates:
            self.logger.error("No valid dates to download")
            return False
        self.logger.info(f"Processing {len(dates)} dates")
        all_chunks = self.chunk_dates(dates)
        total_chunks = len(all_chunks)
        self.logger.info(f"Split into {total_chunks} chunk(s) by model/month")
        
        # Determine expected chunk filenames
        base_name = Path(filename).stem
        chunk_filenames_expected: List[str] = []
        for i, chunk in enumerate(all_chunks, 1):
            if total_chunks > 1:
                chunk_fn = f"{base_name}_chunk_{i}_{chunk['model']}.zip"
            else:
                chunk_fn = filename
            chunk_filenames_expected.append(chunk_fn)
        
        # Identify already completed chunks from manifest
        completed_from_manifest = set()
        if isinstance(manifest.get('chunks'), dict):
            for fn, data in manifest['chunks'].items():
                if data.get('status') == 'completed':
                    # Ensure file still exists
                    if (self.output_dir / fn).exists():
                        completed_from_manifest.add(fn)
                        self.logger.info(f"Chunk already downloaded: {fn}")
        
        # Create tasks for missing chunks. If a cred_index is pinned, enforce sequential
        # execution to avoid multiple concurrent requests on the same key.
        results: List[Tuple[str, bool]] = []
        if cred_index is not None:
            for i, chunk in enumerate(all_chunks, 1):
                chunk_fn = chunk_filenames_expected[i - 1]
                if chunk_fn in completed_from_manifest:
                    continue
                try:
                    res = await self.download_chunk(chunk, filename, i, total_chunks, cred_index=cred_index)
                    results.append(res)
                except Exception as e:
                    self.logger.error(f"Sequential chunk download failed for {chunk_fn}: {e}")
        else:
            tasks: List[asyncio.Task] = []
            task_descriptions: List[str] = []
            for i, chunk in enumerate(all_chunks, 1):
                chunk_fn = chunk_filenames_expected[i - 1]
                if chunk_fn in completed_from_manifest:
                    continue
                task = asyncio.create_task(self.download_chunk(chunk, filename, i, total_chunks, cred_index=cred_index))
                tasks.append(task)
                task_descriptions.append(chunk_fn)
            if tasks:
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, res in enumerate(gathered):
                    if isinstance(res, Exception):
                        self.logger.error(f"Task exception for {task_descriptions[idx]}: {res}")
                    else:
                        results.append(res)
        
        # Build status map for all expected chunks (merge manifest info and new results)
        chunk_status: Dict[str, Dict[str, str]] = {}
        # Start with manifest-completed
        for fn in completed_from_manifest:
            chunk_status[fn] = {"status": "completed"}
        # Apply new results
        for fn, ok in results:
            chunk_status[fn] = {"status": "completed" if ok else "failed"}
        # Any remaining expected chunks not in status map are failed/missing
        for fn in chunk_filenames_expected:
            if fn not in chunk_status:
                # Check if file exists (may have been present without manifest)
                if (self.output_dir / fn).exists():
                    chunk_status[fn] = {"status": "completed"}
                else:
                    chunk_status[fn] = {"status": "failed"}
        
        successful = sum(1 for v in chunk_status.values() if v['status'] == 'completed')
        overall_status = 'completed' if successful == total_chunks else ('partial' if successful > 0 else 'failed')
        
        # Combine chunk ZIPs into single output ZIP if needed
        combined_path = self.output_dir / filename
        if total_chunks == 1:
            # Already downloaded directly to `filename` by download_chunk
            pass
        else:
            if successful > 0:
                # Create combined ZIP by adding GRIB chunk content as entries
                try:
                    with zipfile.ZipFile(combined_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
                        for fn, meta in chunk_status.items():
                            if meta['status'] != 'completed':
                                continue
                            src_path = self.output_dir / fn
                            if not src_path.exists():
                                continue
                            # If the chunk is a ZIP, add its members; otherwise add the raw file
                            try:
                                if zipfile.is_zipfile(src_path):
                                    with zipfile.ZipFile(src_path, 'r') as zin:
                                        for member in zin.namelist():
                                            data = zin.read(member)
                                            arcname = f"{Path(fn).stem}_{Path(member).name}"
                                            zout.writestr(arcname, data)
                                else:
                                    with open(src_path, 'rb') as fh:
                                        data = fh.read()
                                    arcname = f"{Path(fn).stem}.grib2"
                                    zout.writestr(arcname, data)
                            except Exception as e:
                                self.logger.warning(f"Failed to add chunk {fn} to combined ZIP: {e}")
                    self.logger.info(f"✅ Created combined ZIP: {combined_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to create combined ZIP: {e}")
                    overall_status = 'partial' if successful > 0 else 'failed'
            else:
                self.logger.error("No successful chunks to combine")
        
        # Write manifest with detailed statuses
        await self.write_manifest(filename, chunk_status, overall_status)
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("Download Summary")
        self.logger.info(f"Total chunks: {total_chunks}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {total_chunks - successful}")
        if any(v['status'] != 'completed' for v in chunk_status.values()):
            failed_names = [k for k, v in chunk_status.items() if v['status'] != 'completed']
            self.logger.info(f"Failed chunks: {', '.join(failed_names)}")
        self.logger.info("=" * 60)
        
        return successful > 0


async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(
        description="Refactored Asynchronous GloFAS Forecast Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--dates",
        type=str,
        required=True,
        help="Comma-separated list of dates (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./downloads",
        help="Output directory for downloads"
    )
    
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Output filename for the combined ZIP file"
    )
    
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run in background mode (detached)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help=f"Maximum concurrent downloads (default: {config.max_concurrent_downloads})"
    )
    
    args = parser.parse_args()
    
    # Handle background mode
    if args.background:
        # Properly detach process for background mode
        import signal
        import platform
        
        # Write PID file for tracking
        pid_file = Path(args.output_dir) / f".{args.filename}.pid"
        
        if platform.system() != 'Windows':
            # Unix-like systems: proper daemonization
            if hasattr(os, 'fork'):
                # First fork
                pid = os.fork()
                if pid > 0:
                    # Parent process exits
                    print(f"Started background process with PID: {pid}")
                    sys.exit(0)
                
                # Decouple from parent environment
                os.setsid()
                
                # Second fork
                pid = os.fork()
                if pid > 0:
                    # Parent exits
                    sys.exit(0)
                
                # Now we're in the daemon process
                pid_file.write_text(str(os.getpid()))
                
                # Ignore hangup signal
                if hasattr(signal, 'SIGHUP'):
                    signal.signal(signal.SIGHUP, signal.SIG_IGN)
        else:
            # Windows: use subprocess to start detached process
            import subprocess
            
            # Build command without --background flag
            cmd = [sys.executable, __file__]
            cmd.extend(['--dates', args.dates])
            cmd.extend(['--output-dir', args.output_dir])
            cmd.extend(['--filename', args.filename])
            if args.max_concurrent:
                cmd.extend(['--max-concurrent', str(args.max_concurrent)])
            
            # Start detached process
            log_file = Path(args.output_dir) / f"{args.filename}.background.log"
            with open(log_file, 'w') as log_handle:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                    if platform.system() == 'Windows' else 0,
                    start_new_session=True
                )
            
            pid_file.write_text(str(proc.pid))
            print(f"Started background process with PID: {proc.pid}")
            print(f"Log file: {log_file}")
            sys.exit(0)
        
        # Redirect output to log file (for the daemon process)
        log_file = Path(args.output_dir) / f"{args.filename}.background.log"
        log_handle = open(log_file, 'a', buffering=1)
        sys.stdout = log_handle
        sys.stderr = log_handle
    
    # Create downloader
    downloader = AsyncGloFASDownloader(args.output_dir)
    
    # Check environment
    if not await downloader.check_environment():
        return 1
    
    # Download data
    success = await downloader.download_all(args.dates, args.filename)
    
    # Clean up PID file if in background mode
    if args.background:
        pid_file = Path(args.output_dir) / f".{args.filename}.pid"
        if pid_file.exists():
            pid_file.unlink()
    
    return 0 if success else 1


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
