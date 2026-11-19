"""Dataset retrieval adapters.

The platform ships with the Kaggle snapshot, but retrieval is deliberately
pluggable:

* :func:`fetch_kaggle_dataset` - downloads a Kaggle dataset archive (works
  anonymously for public datasets; ``KAGGLE_USERNAME``/``KAGGLE_KEY`` are used
  when present for private data).
* :func:`fetch_fred_series` - downloads an official FRED series CSV, which is
  the supported path for refreshing the dataset with new observations.

Neither adapter is required for training: :func:`load_raw_frame` reads whatever
CSV is configured in ``data.raw_dir``.
"""

from __future__ import annotations

import os
import time
import zipfile
from io import BytesIO
from pathlib import Path

import requests

from interest_rate_mlops.exceptions import IngestionError
from interest_rate_mlops.logging_utils import get_logger

logger = get_logger(__name__)

KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{slug}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
REQUEST_TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3


def _get(url: str, params: dict[str, str] | None = None) -> requests.Response:
    """GET with bounded retries and explicit failure messages."""
    headers = {"User-Agent": "interest-rate-mlops/0.1"}
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")
    if username and key:
        headers["Authorization"] = f"Basic {username}:{key}"

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS
            )
            if response.status_code == 200:
                return response
            last_error = IngestionError(
                f"HTTP {response.status_code} from {url}: {response.text[:200]}"
            )
            if response.status_code in {401, 403, 404}:  # not retryable
                break
        except requests.RequestException as exc:
            last_error = exc
        if attempt < MAX_ATTEMPTS:
            time.sleep(2 * attempt)
    raise IngestionError(f"failed to download {url}: {last_error}")


def fetch_kaggle_dataset(
    slug: str,
    destination: str | Path,
    force: bool = False,
) -> Path:
    """Download and extract a Kaggle dataset, returning the CSV path.

    Args:
        slug: Kaggle dataset identifier, e.g. ``federalreserve/interest-rates``.
        destination: Directory that will receive the CSV.
        force: Re-download even when a CSV is already present.

    Returns:
        Path to the extracted CSV.
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest.glob("*.csv"))
    if existing and not force:
        logger.info(
            "raw dataset already present, skipping download",
            extra={"event": "kaggle_skip_download", "path": str(existing[0])},
        )
        return existing[0]

    logger.info("downloading Kaggle dataset", extra={"event": "kaggle_download", "dataset": slug})
    response = _get(KAGGLE_DOWNLOAD_URL.format(slug=slug))
    try:
        archive = zipfile.ZipFile(BytesIO(response.content))
    except zipfile.BadZipFile as exc:
        raise IngestionError(
            "Kaggle response was not a zip archive. For private datasets set "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        ) from exc

    csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if not csv_names:
        raise IngestionError(f"no CSV found in Kaggle archive for {slug}")

    for name in csv_names:
        target = dest / Path(name).name
        with archive.open(name) as source, target.open("wb") as sink:
            sink.write(source.read())
        logger.info(
            "extracted raw CSV",
            extra={"event": "kaggle_extract", "path": str(target)},
        )

    return dest / Path(csv_names[0]).name


def fetch_fred_series(
    series_id: str,
    destination: str | Path,
    force: bool = False,
) -> Path:
    """Download an official FRED series as CSV (no API key required).

    FRED returns ``observation_date,<SERIES_ID>`` rows, missing observations are
    the literal ``.`` which the loader normalises to ``NaN``.

    Args:
        series_id: FRED series identifier, e.g. ``FEDFUNDS`` or ``DGS10``.
        destination: Directory that will receive ``<series_id>.csv``.
        force: Re-download even when the file already exists.
    """
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{series_id}.csv"
    if target.exists() and not force:
        return target

    response = _get(FRED_CSV_URL, params={"id": series_id})
    if "observation_date" not in response.text.splitlines()[0]:
        raise IngestionError(f"unexpected FRED payload for {series_id}")
    target.write_bytes(response.content)
    logger.info(
        "downloaded FRED series",
        extra={"event": "fred_download", "path": str(target), "series": series_id},
    )
    return target
