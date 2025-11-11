"""
Centralized configuration for the GloFAS data processing pipeline.
Uses Pydantic for type validation and environment variable support.
"""

from typing import List, Tuple, Optional
from datetime import date, timedelta
from pathlib import Path
from pydantic import Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import re
import json
import configparser


class PipelineConfig(BaseSettings):
    """Main configuration for the data processing pipeline."""
    
    # ============== API Configuration ==============
    cds_api_url: str = Field(
        default="https://ewds.climate.copernicus.eu/api",
        description="CDS API endpoint URL"
    )
    cds_api_key: Optional[str] = Field(
        default=None,
        env="CDS_API_KEY",
        description="CDS API key (from environment variable)"
    )
    cds_api_uid: Optional[str] = Field(
        default=None,
        env="CDS_API_UID",
        description="CDS API user ID (from environment variable)"
    )
    cds_api_tokens_raw: Optional[str] = Field(
        default="e512d7ab-849c-4c78-ae9c-d5d77d603c4c dae0c4c5-6c11-4e29-88f8-9804389cc7fa e76fc245-3205-4965-97d8-9f4280c4695a",
        env="CDS_API_TOKENS",
        description="Optional list of CDS API tokens for credential rotation (comma, whitespace, or JSON list)"
    )
    
    # ============== Rate Limiting ==============
    max_concurrent_downloads: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum concurrent download requests (to avoid rate limiting)"
    )
    max_retries: int = Field(
        default=5,
        ge=1,
        description="Maximum number of retry attempts for failed requests"
    )
    initial_retry_delay: float = Field(
        default=60.0,
        ge=1.0,
        description="Initial retry delay in seconds"
    )
    max_retry_delay: float = Field(
        default=3600.0,
        ge=60.0,
        description="Maximum retry delay in seconds"
    )
    request_timeout: int = Field(
        default=0,
        ge=0,
        description="Request timeout in seconds (0 disables the limit)"
    )
    
    # ============== Download Configuration ==============
    chunk_size_mb: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum chunk size in MB for downloads"
    )
    download_check_interval: int = Field(
        default=30,
        ge=5,
        description="Interval in seconds to check download status"
    )
    max_wait_minutes: int = Field(
        default=60,
        ge=10,
        description="Maximum time to wait for a download in minutes"
    )
    
    # ============== Data Quality ==============
    bad_dates: List[Tuple[date, date]] = Field(
        default=[
            (date(2025, 1, 1), date(2025, 1, 4)),
            (date(2024, 11, 27), date(2024, 11, 28)),
            (date(2024, 3, 19), date(2024, 3, 21)),
        ],
        description="Known periods with corrupt or erroneous GloFAS data"
    )
    
    # ============== GloFAS Configuration ==============
    glofas_transition_date: date = Field(
        default=date(2021, 5, 26),
        description="Date when GloFAS operational hydrological model switched to LISFLOOD (pre: HTESSEL-LISFLOOD)"
    )
    glofas_forecast_start_date: date = Field(
        default=date(2019, 11, 5),
        description="Date when GloFAS forecast data became available"
    )
    glofas_product_type_v31: str = Field(
        default="control_reforecast",
        description="Product type for GloFAS v3.1"
    )
    glofas_product_type_v40: str = Field(
        default="ensemble_perturbed_reforecasts",
        description="Product type for GloFAS v4.0"
    )
    
    # ============== Spatial Configuration ==============
    aoi_buffer_meters: int = Field(
        default=25000,
        ge=1000,
        le=100000,
        description="Default buffer size in meters for AOI (25km)"
    )
    glofas_riverine_bbox_meters: int = Field(
        default=20000,
        ge=1000,
        le=100000,
        description="Buffer size for riverine floods (20km)"
    )
    
    # ============== File System Configuration ==============
    temp_dir_name: str = Field(
        default="temp_downloads",
        description="Name of temporary download directory"
    )
    download_dir: Optional[str] = Field(
        default="/content/temp_downloads",
        description="Absolute path for storing large forecast downloads (overrides temp_dir_name if set)"
    )
    log_file_name: str = Field(
        default="pipeline.log",
        description="Name of the main log file"
    )
    manifest_extension: str = Field(
        default=".manifest.json",
        description="Extension for manifest files"
    )
    
    # ============== Processing Configuration ==============
    chunk_days_part1: Tuple[int, int] = Field(
        default=(1, 15),
        description="First chunk of days in a month"
    )
    chunk_days_part2_start: int = Field(
        default=16,
        description="Start day for second chunk of month"
    )
    
    # ============== Logging Configuration ==============
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    log_format: str = Field(
        default="%(asctime)s [%(levelname)s] %(message)s",
        description="Log message format"
    )
    
    # ============== Async Configuration ==============
    asyncio_debug: bool = Field(
        default=False,
        env="ASYNCIO_DEBUG",
        description="Enable asyncio debug mode"
    )
    process_pool_size: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Size of the process pool for parallel processing"
    )
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    @staticmethod
    def _read_cdsapirc_file(path: Path) -> dict:
        """Read ~/.cdsapirc in either INI ([cds]) or simple key:value format.
        Returns a dict with keys: url, api_key, uid, raw_key.
        """
        result = {"url": None, "api_key": None, "uid": None, "raw_key": None}
        try:
            # Try INI format first
            cp = configparser.ConfigParser()
            read_ok = cp.read(path)
            if read_ok and cp.has_section("cds"):
                # INI-style entries
                url = cp.get("cds", "url", fallback=None)
                uid = cp.get("cds", "uid", fallback=None)
                raw_key = cp.get("cds", "key", fallback=None)
            else:
                # Fallback: simple key:value parsing
                url = uid = raw_key = None
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    text = path.read_text(errors="ignore")
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ':' in line:
                        k, v = line.split(':', 1)
                        k = k.strip().lower()
                        v = v.strip()
                        if k == 'url':
                            url = v
                        elif k == 'uid':
                            uid = v
                        elif k == 'key':
                            raw_key = v
            # Process key/uid
            api_key = None
            if raw_key:
                if ':' in raw_key:
                    maybe_uid, maybe_key = raw_key.split(':', 1)
                    if not uid:
                        uid = maybe_uid.strip()
                    api_key = maybe_key.strip()
                else:
                    api_key = raw_key.strip()
            result.update({"url": url, "api_key": api_key, "uid": uid, "raw_key": raw_key})
        except Exception:
            # Silent fallback; caller will decide
            pass
        return result
        
    @field_validator("cds_api_key", "cds_api_uid", mode="after")
    def check_cds_credentials(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        """Validate CDS API credentials and fall back to ~/.cdsapirc if missing.
        Supports both INI ([cds]) and simple key:value formats. If the key is
        provided as "uid:apikey", it is split accordingly.
        """
        if v is None or (isinstance(v, str) and not v.strip()):
            cdsapirc_path = Path.home() / ".cdsapirc"
            if cdsapirc_path.exists():
                parsed = cls._read_cdsapirc_file(cdsapirc_path)
                if info.field_name == "cds_api_key":
                    return parsed.get("api_key") or parsed.get("raw_key") or v
                elif info.field_name == "cds_api_uid":
                    return parsed.get("uid") or v
        return v

    @field_validator("bad_dates")
    def validate_bad_dates(cls, v: List[Tuple[date, date]]) -> List[Tuple[date, date]]:
        """Ensure bad dates are valid date ranges."""
        for start_date, end_date in v:
            if start_date > end_date:
                raise ValueError(f"Invalid date range: {start_date} > {end_date}")
        return v
    
    @property
    def max_wait_seconds(self) -> int:
        """Convert max_wait_minutes to seconds."""
        return self.max_wait_minutes * 60
    
    @property
    def chunk_size_bytes(self) -> int:
        """Convert chunk_size_mb to bytes."""
        return self.chunk_size_mb * 1024 * 1024
    
    def is_bad_date(self, event_date: date) -> bool:
        """Check if a date falls within any known bad data period."""
        for start_date, end_date in self.bad_dates:
            if start_date <= event_date <= end_date:
                return True
        return False
    
    def get_glofas_version(self, request_date: date) -> str:
        """Determine GloFAS version based on date."""
        if request_date >= self.glofas_transition_date:
            return "4.0"
        return "3.1"
    
    def get_product_type(self, request_date: date) -> str:
        """Get the appropriate product type for a given date."""
        if request_date >= self.glofas_transition_date:
            return self.glofas_product_type_v40
        return self.glofas_product_type_v31

    @staticmethod
    def _parse_token_value(value) -> List[str]:
        """Normalize various token representations (JSON, CSV, whitespace)."""
        if value in (None, ""):
            return []

        # Direct list or tuple input
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            # Try JSON list first
            try:
                loaded = json.loads(text)
                if isinstance(loaded, (list, tuple)):
                    return [str(item).strip() for item in loaded if str(item).strip()]
            except json.JSONDecodeError:
                pass

            # Fallback: split by whitespace or commas
            parts = re.split(r"[\s,]+", text)
            return [p for p in parts if p]

        # Fallback: single token coerced to string
        return [str(value).strip()]

    @property
    def cds_api_tokens(self) -> List[str]:
        """Expose parsed CDS API tokens as a list of strings."""
        return self._parse_token_value(self.cds_api_tokens_raw)

    def get_api_tokens(self) -> List[str]:
        """Return list of API tokens, combining explicit tokens and legacy key/uid."""
        tokens = [token for token in self.cds_api_tokens if token]
        if tokens:
            return tokens
        if self.cds_api_key:
            key = self.cds_api_key.strip()
            if not key:
                return []
            if self.cds_api_uid and ":" not in key:
                return [f"{self.cds_api_uid}:{key}"]
            return [key]
        return []

    def get_api_credentials(self) -> List[dict]:
        """Return list of credential dictionaries for cdsapi.Client rotation."""
        tokens = self.get_api_tokens()
        if not tokens:
            return []
        url = (self.cds_api_url or "").strip() or "https://ewds.climate.copernicus.eu/api"
        return [{"url": url, "key": token} for token in tokens]


# Create a singleton instance
config = PipelineConfig()


# Export commonly used values for backward compatibility
MAX_RETRIES = config.max_retries
TRANSITION_DATE = config.glofas_transition_date
BAD_DATES = config.bad_dates
AOI_BUFFER_METERS = config.aoi_buffer_meters
GLOFAS_RIVERINE_BBOX_METERS = config.glofas_riverine_bbox_meters
