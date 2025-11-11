#!/usr/bin/env python3
"""
Split Yearly Enriched Event Files into Monthly Files
Takes large yearly enriched CSV files and splits them into smaller monthly files
"""

import os
import sys
import argparse
import logging
import glob
import re
from pathlib import Path
from datetime import datetime

import pandas as pd

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "split_yearly_to_monthly.log"))
        ]
    )

def find_yearly_files(input_dir):
    """Find all enriched_events_*.csv files in the input directory"""
    pattern = os.path.join(input_dir, "enriched_events_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        logging.warning(f"No enriched_events_*.csv files found in {input_dir}")
        return []
    
    # Sort files by name (which should be chronological)
    files.sort()
    
    logging.info(f"Found {len(files)} yearly files:")
    for file in files:
        logging.info(f"  - {os.path.basename(file)}")
    
    return files

def _normalize_dashes(s: pd.Series) -> pd.Series:
    """Replace various unicode dashes with ASCII hyphen and strip whitespace."""
    if s is None:
        return s
    # Common unicode dashes to hyphen
    return (
        s.astype(str)
        .str.strip()
        .str.replace('\u2013', '-', regex=False)  # EN DASH
        .str.replace('\u2014', '-', regex=False)  # EM DASH
        .str.replace('\u2212', '-', regex=False)  # MINUS SIGN
    )

def _pick_date_column(df: pd.DataFrame) -> str:
    """Heuristically find the date column with flood start date.
    Prefers 'flood_start_date', falls back to 'start_date' or similar variants.
    """
    candidates = []
    # Build normalized map
    norm = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
    for original, lowered in norm.items():
        if lowered in (
            'flood_start_date', 'start_date', 'event_start_date',
            'flood_start', 'start', 'date', 'event_date'
        ):
            candidates.append((lowered, original))
    # Prefer flood_start_date
    for lowered, original in candidates:
        if lowered == 'flood_start_date':
            return original
    # Else pick start_date if present
    for lowered, original in candidates:
        if lowered == 'start_date':
            return original
    # Else return first candidate if any
    if candidates:
        return candidates[0][1]
    return ''

def _parse_dates_robust(s: pd.Series, logger: logging.Logger) -> pd.Series:
    """Parse dates robustly handling ISO (YYYY-MM-DD), DD-MM-YYYY, and common variants.
    Also strips time components if present.
    """
    if s is None or len(s) == 0:
        return pd.to_datetime(pd.Series([], dtype=object), errors='coerce')

    # Normalize unicode dashes and strip
    s = _normalize_dashes(s)
    # Remove time part if present (e.g., '2020-01-04 00:00:00' or '2020-01-04T00:00:00')
    s_no_time = s.str.replace(r"[T\s].*$", '', regex=True)

    # 1) Try exact ISO first (fast path)
    iso_mask = s_no_time.str.match(r"^\d{4}-\d{2}-\d{2}$")
    parsed = pd.to_datetime(s_no_time.where(iso_mask), format='%Y-%m-%d', errors='coerce')

    # 2) For remaining, try dayfirst=True generic parser (handles DD-MM-YYYY and DD/MM/YYYY)
    remaining_mask = parsed.isna()
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(
            s_no_time[remaining_mask], errors='coerce', dayfirst=True
        )

    # 3) If still missing, try without dayfirst (handles MM/DD/YYYY, etc.)
    remaining_mask = parsed.isna()
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(
            s_no_time[remaining_mask], errors='coerce', dayfirst=False
        )

    # 4) As a last resort, handle pure numeric Excel serials if any (e.g., 43831)
    remaining_mask = parsed.isna()
    if remaining_mask.any():
        numeric_mask = s_no_time[remaining_mask].str.match(r'^\d+(\.\d+)?$')
        if numeric_mask.any():
            try:
                serials = s_no_time[remaining_mask][numeric_mask].astype(float)
                # Excel serial date: days since 1899-12-30
                base = pd.Timestamp('1899-12-30')
                parsed.loc[remaining_mask][numeric_mask] = base + pd.to_timedelta(serials, unit='D')
            except Exception as e:
                logger.debug(f"Excel serial parse failed: {e}")

    return parsed

def split_yearly_file(yearly_file, output_dir):
    """Split a single yearly file into monthly files"""
    filename = os.path.basename(yearly_file)
    logging.info(f"Processing {filename}...")
    
    try:
        # Load CSV
        df = pd.read_csv(yearly_file)
        logging.info(f"Loaded {len(df)} events from {filename}")
        
        # Find the most likely date column and parse robustly
        date_col = _pick_date_column(df)
        if not date_col:
            logging.error("No date column found (expected 'flood_start_date' or 'start_date')")
            return []

        parsed_dates = _parse_dates_robust(df[date_col], logging.getLogger())
        df['flood_start_date'] = parsed_dates
        
        # Remove any rows with invalid dates
        invalid_dates = df['flood_start_date'].isna()
        if invalid_dates.sum() > 0:
            # Show a few examples to aid debugging
            invalid_examples = (
                df.loc[invalid_dates, date_col]
                .astype(str)
                .dropna()
                .unique()[:10]
            )
            logging.warning(f"Removing {invalid_dates.sum()} rows with invalid dates")
            if len(invalid_examples) > 0:
                logging.warning(f"Invalid examples from column '{date_col}': {list(invalid_examples)}")
            df = df.dropna(subset=['flood_start_date'])
        
        # Add year and month columns
        df['year'] = df['flood_start_date'].dt.year
        df['month'] = df['flood_start_date'].dt.month
        
        # Group by year-month and save monthly files
        monthly_files = []
        for (year, month), month_df in df.groupby(['year', 'month']):
            # Create filename: events_YYYY_MM.csv (consistent with run_pipeline.py)
            monthly_filename = f"events_{year}_{month:02d}.csv"
            monthly_filepath = os.path.join(output_dir, monthly_filename)
            
            # Save monthly data
            month_df.to_csv(monthly_filepath, index=False)
            monthly_files.append(monthly_filepath)
            
            # Verify file was created successfully
            if not os.path.exists(monthly_filepath):
                logging.error(f"  ❌ Failed to create {monthly_filename}")
            else:
                logging.info(f"  ✅ Created {monthly_filename} with {len(month_df)} events")
        
        logging.info(f"✅ Successfully split {filename} into {len(monthly_files)} monthly files")
        return monthly_files
        
    except Exception as e:
        logging.error(f"❌ Error processing {filename}: {e}")
        return []

def split_all_yearly_files(input_dir, output_dir):
    """Split all yearly files into monthly files"""
    logging.info("=" * 60)
    logging.info("Splitting Yearly Files to Monthly Files")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 60)
    
    # Find all yearly files
    yearly_files = find_yearly_files(input_dir)
    
    if not yearly_files:
        logging.error("No yearly files found to process")
        return
    
    # Process each yearly file
    total_monthly_files = 0
    successful_files = 0
    
    for yearly_file in yearly_files:
        monthly_files = split_yearly_file(yearly_file, output_dir)
        
        if monthly_files:
            successful_files += 1
            total_monthly_files += len(monthly_files)
        else:
            logging.error(f"Failed to process {os.path.basename(yearly_file)}")
    
    # Summary
    logging.info("=" * 60)
    logging.info("Splitting Summary")
    logging.info(f"Yearly files processed: {successful_files}/{len(yearly_files)}")
    logging.info(f"Monthly files created: {total_monthly_files}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Split yearly enriched event files into monthly files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split all yearly files in a directory
  python split_yearly_to_monthly.py --input-dir ./enriched_data --output-dir ./monthly_data
  
  # Split files from specific directory
  python split_yearly_to_monthly.py --input-dir ./drive/enriched_data --output-dir ./drive/monthly_data
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing enriched_events_*.csv files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for monthly CSV files"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.output_dir)
    
    # Run splitting
    try:
        split_all_yearly_files(args.input_dir, args.output_dir)
        logging.info("✅ Splitting completed successfully")
    except Exception as e:
        logging.error(f"Splitting failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
