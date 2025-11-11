# download_glofas_v3.py
import cdsapi
import os
import argparse
import logging
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import threading

logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

MAX_RETRIES = 5
WAIT_BASE = 30  # seconds


# API Credentials - Hardcoded rotation
CDS_API_TOKENS = [
    "22ab58cb-994c-4b37-b50b-8fd987915ac6",
    "e76fc245-3205-4965-97d8-9f4280c4695a", 
    "e512d7ab-849c-4c78-ae9c-d5d77d603c4c"
]
CDS_API_URL = "https://ewds.climate.copernicus.eu/api"
# Build credential pool
_CRED_POOL: List[Dict] = [{"url": CDS_API_URL, "key": token} for token in CDS_API_TOKENS]
_CRED_LOCK = threading.Lock()
_CRED_INDEX = 0

def get_rotating_client(nonblocking: bool = False) -> cdsapi.Client:
    """Get client with next credential (round-robin, no error waiting)."""
    global _CRED_INDEX
    if not _CRED_POOL:
        raise ValueError("No CDS API credentials configured")
    with _CRED_LOCK:
        cred = _CRED_POOL[_CRED_INDEX % len(_CRED_POOL)]
        key_num = (_CRED_INDEX % len(_CRED_POOL)) + 1
        _CRED_INDEX += 1
    logging.info(f"Using CDS API Key #{key_num} of {len(_CRED_POOL)}")
    return cdsapi.Client(
        url=cred["url"],
        key=cred["key"],
        wait_until_complete=not nonblocking,
        quiet=True,
        progress=False,
    )


def robust_retrieve(dataset, request, target):
    """Retrieve with retries using rotating credentials."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_rotating_client(nonblocking=False)  # Auto-rotates each attempt
            client.retrieve(dataset, request, target)
            if Path(target).exists() and Path(target).stat().st_size > 0:
                logging.info(f"✅ Successfully downloaded {target}")
                return True
            else:
                logging.warning(f"File {target} is empty after attempt {attempt}")
        except Exception as e:
            logging.error(f"Error on attempt {attempt} for {target}: {e}")

        wait_time = WAIT_BASE * attempt
        logging.info(f"Retrying in {wait_time}s...")
        time.sleep(wait_time)

    logging.error(f"All {MAX_RETRIES} retries failed for {target}")
    return False


def _build_base_request(year: int, month: int):
    from calendar import monthrange
    month_str = f"{month:02d}"
    num_days = monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, num_days + 1)]
    return month_str, {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": [
            "river_discharge_in_the_last_24_hours",
            "runoff_water_equivalent",
            "snow_depth_water_equivalent",
            "soil_wetness_index",
        ],
        "hyear": [str(year)],
        "hmonth": [month_str],
        "hday": days,
    }


def submit_month_request(client_submit: cdsapi.Client, year: int, month: int, temp_dir: Path):
    """Submit a monthly request without waiting. Prefer NetCDF, fallback to GRIB.

    Returns a dict with keys: year, month, format, result, request, dataset, target
    or None if both formats fail at submission.
    """
    month_str, base_request = _build_base_request(year, month)

    target_nc = temp_dir / f"glofas_{year}_{month_str}.nc"
    target_grib = temp_dir / f"glofas_{year}_{month_str}.grib2"

    # Skip if already downloaded
    if target_nc.exists() and target_nc.stat().st_size > 0:
        logging.info(f"✅ NetCDF data for {year}-{month_str} already exists. Skipping submission.")
        return None
    if target_grib.exists() and target_grib.stat().st_size > 0:
        logging.info(f"✅ GRIB2 data for {year}-{month_str} already exists. Skipping submission.")
        return None

    dataset = "cems-glofas-historical"

    # Try NetCDF first (use provided client; rotation occurs at call site)
    try:
        req_nc = dict(base_request)
        req_nc["format"] = "netcdf"
        logging.info(f"📤 Submitting NetCDF request for {year}-{month_str} (non-blocking)...")
        res_nc = client_submit.retrieve(dataset, req_nc)
        state = getattr(res_nc, "reply", {}).get("state")
        logging.info(f"   ↳ Server state for {year}-{month_str} (NetCDF): {state}")
        return {
            "year": year,
            "month": month,
            "format": "netcdf",
            "result": res_nc,
            "request": req_nc,
            "dataset": dataset,
            "target": str(target_nc),
        }
    except Exception as e:
        logging.warning(
            f"NetCDF submission failed for {year}-{month_str} with error: {e}. Will try GRIB2."
        )

    # Fallback to GRIB (use provided client)
    try:
        req_grib = dict(base_request)
        req_grib["format"] = "grib"
        logging.info(f"📤 Submitting GRIB2 request for {year}-{month_str} (non-blocking)...")
        res_grib = client_submit.retrieve(dataset, req_grib)
        state = getattr(res_grib, "reply", {}).get("state")
        logging.info(f"   ↳ Server state for {year}-{month_str} (GRIB): {state}")
        return {
            "year": year,
            "month": month,
            "format": "grib",
            "result": res_grib,
            "request": req_grib,
            "dataset": dataset,
            "target": str(target_grib),
        }
    except Exception as e:
        logging.error(f"Both NetCDF and GRIB submissions failed for {year}-{month_str}: {e}")
        return None


def poll_and_download_job(job: dict) -> bool:
    """Poll a submitted job until it's completed, then download it.

    If a NetCDF job fails, fallback to submitting GRIB for the same month automatically.
    Returns True on success, False otherwise.
    """
    year, month = job["year"], job["month"]
    fmt = job["format"]
    result = job["result"]
    request = job["request"]
    dataset = job["dataset"]
    target = Path(job["target"]) 

    # Skip if file already present
    if target.exists() and target.stat().st_size > 0:
        logging.info(f"✅ Already present: {target}")
        return True

    # Poll loop
    attempt = 0
    while True:
        attempt += 1
        try:
            if hasattr(result, "update"):
                result.update()
        except Exception as e:
            logging.warning(f"[{year}-{month:02d} {fmt}] update() failed on attempt {attempt}: {e}")

        reply = getattr(result, "reply", {}) or {}
        state = reply.get("state", "unknown")
        logging.info(f"[{year}-{month:02d} {fmt}] State: {state}")

        if state == "completed":
            # Prefer using the Result object to download
            try:
                if hasattr(result, "download"):
                    result.download(str(target))
                else:
                    # Fallback: retrieve with rotating credential
                    if not robust_retrieve(dataset, request, str(target)):
                        return False
                if target.exists() and target.stat().st_size > 0:
                    logging.info(f"✅ Downloaded: {target}")
                    return True
            except Exception as e:
                logging.error(f"[{year}-{month:02d} {fmt}] download error: {e}")
                # Final fallback using retrieve
                if robust_retrieve(dataset, request, str(target)):
                    return True
                return False

        if state == "failed":
            if fmt == "netcdf":
                # Fallback to GRIB for this month
                month_str, base_request = _build_base_request(year, month)
                req_grib = dict(base_request)
                req_grib["format"] = "grib"
                try:
                    logging.info(f"[{year}-{month:02d} netcdf] Failed. Falling back to GRIB2 submission...")
                    client_submit = get_rotating_client(nonblocking=True)
                    result = client_submit.retrieve(dataset, req_grib)
                    request = req_grib
                    fmt = "grib"
                    target = Path(target.parent / f"glofas_{year}_{month:02d}.grib2")
                    continue
                except Exception as e:
                    logging.error(f"[{year}-{month:02d}] GRIB2 submission after netcdf failure also failed: {e}")
                    return False
            else:
                logging.error(f"[{year}-{month:02d} {fmt}] Request failed on server.")
                return False

        # Not completed yet, wait with backoff
        wait_time = min(WAIT_BASE * attempt, 300)  # cap at 5 min between polls
        time.sleep(wait_time)


def main(args):
    start_year, end_year = args.start_year, args.end_year
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)

    # Build month list: include previous year's December for cross-year negatives
    months = []
    # Add December from year before start_year (for Jan-Feb negatives that go back 30 days)
    if start_year > 1981:
        months.append((start_year - 1, 12))
    # Add all months for requested years
    months.extend([(year, month) for year in range(start_year, end_year + 1) for month in range(1, 12 + 1)])

    # Apply continuation filter for the start year
    c = getattr(args, "continue_from", 1) or 1
    if not 1 <= int(c) <= 12:
        logging.error(f"Invalid --continue-from value: {c}. Must be 1-12.")
        sys.exit(1)
    if c > 1:
        # Keep prior December only if continuation month is Jan (1); otherwise drop prior December
        filtered = []
        for (y, m) in months:
            if y < start_year:
                # prior December: include only if continuing from January
                if c == 1 and y == start_year - 1 and m == 12:
                    filtered.append((y, m))
                continue
            if y == start_year and m < c:
                # Skip early months of the start year
                continue
            filtered.append((y, m))
        months = filtered
        logging.info(f"Continuation enabled (-c {c}). Starting downloads from {start_year}-{c:02d} (kept prior Dec only if needed)")
    jobs = []

    logging.info(f"Total months to download: {len(months)} (including prior December if needed)")
    logging.info("📤 Phase 1: Submitting monthly requests without waiting...")
    for (y, m) in months:
        job = submit_month_request(get_rotating_client(nonblocking=True), y, m, temp_dir)
        if job is not None:
            jobs.append(job)

    total_jobs = len(jobs)
    logging.info(f"📥 Phase 2: Polling and downloading {total_jobs} pending chunks...")

    if args.max_parallel > 4:
        logging.warning(
            f"You requested max_parallel={args.max_parallel}. The CDS service often limits concurrency per user; "
            f"values >2 may lead to throttling (HTTP 429) or longer waits."
        )

    success_count = 0
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [executor.submit(poll_and_download_job, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                if fut.result():
                    success_count += 1
            except Exception as e:
                logging.error(f"Unhandled error during download: {e}")

    logging.info("✅ All jobs processed.")
    logging.info(f"Downloaded {success_count}/{total_jobs} chunks successfully (or already present).")
    logging.info(f"Files are available in {temp_dir}. You can open them with xarray/cfgrib individually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustly download GloFAS v4.0 monthly GRIB2 data.")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store downloaded GRIB2 files")
    parser.add_argument(
        "--continue-from", "-c",
        type=int,
        default=1,
        metavar="MONTH",
        help="Skip months 1..(MONTH-1) in the start year and begin from MONTH (1-12). Default: 1"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Maximum number of parallel CDS requests to run. 1-2 is recommended due to CDS limits.",
    )
    args = parser.parse_args()
    main(args)
