"""Stable identities for FIRMS observations across reordered pipeline outputs."""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


IDENTITY_FIELDS = (
    "satellite", "instrument", "acq_date", "acq_time", "latitude", "longitude",
    "scan", "track", "bright_ti4", "bright_ti5", "frp",
)


def _canonical(value: Any, numeric: bool) -> str:
    if value is None or pd.isna(value):
        return ""
    if numeric:
        try:
            return format(float(value), ".5f")
        except (TypeError, ValueError):
            pass
    return str(value).strip().lower()


def _stable_id_from_values(values: Any, source: str) -> str:
    numeric_fields = {"latitude", "longitude", "scan", "track", "bright_ti4", "bright_ti5", "frp"}
    canonical = [str(source).strip().lower()]
    for field, value in zip(IDENTITY_FIELDS, values):
        canonical.append(_canonical(value, field in numeric_fields))
    token = hashlib.sha256("|".join(canonical).encode("utf-8")).hexdigest()[:24]
    return f"FIRMS-{token}"


def stable_observation_id(row: Any, source: str = "VIIRS_NOAA20_SP") -> str:
    """Hash stable sensor/acquisition/location attributes, never the row number."""
    if hasattr(row, "get"):
        values = (row.get(field, "") for field in IDENTITY_FIELDS)
    else:
        values = (getattr(row, field, "") for field in IDENTITY_FIELDS)
    return _stable_id_from_values(values, source)


def fingerprint_observation_ids(ids: Any) -> str:
    """Order-independent compact fingerprint; callers separately verify uniqueness."""
    xor_value = 0
    sum_value = 0
    count = 0
    for observation_id in ids:
        digest = int.from_bytes(hashlib.sha256(str(observation_id).encode("utf-8")).digest(), "big")
        xor_value ^= digest
        sum_value = (sum_value + digest) % (1 << 256)
        count += 1
    return f"{count}:{xor_value:064x}:{sum_value:064x}"


def ensure_observation_ids(frame: pd.DataFrame, source: str = "VIIRS_NOAA20_SP") -> pd.DataFrame:
    """Preserve valid IDs or create stable IDs for source records."""
    result = frame.copy()
    if "obs_id" in result.columns:
        existing_ids = result["obs_id"].astype("string").str.strip()
        if (
            result["obs_id"].notna().all()
            and existing_ids.ne("").all()
            and existing_ids.str.startswith("FIRMS-").all()
        ):
            result["obs_id"] = existing_ids.astype(str)
            if result["obs_id"].duplicated().any():
                raise ValueError("Existing durable FIRMS obs_id values are not unique")
            return result
    identity_frame = result.reindex(columns=IDENTITY_FIELDS)
    result["obs_id"] = [
        _stable_id_from_values(row, source)
        for row in identity_frame.itertuples(index=False, name=None)
    ]
    if result["obs_id"].duplicated().any():
        duplicates = int(result["obs_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"{duplicates} FIRMS rows have identical natural-key fields; cannot create unique obs_id values. "
            "Inspect source duplicates or add a stable FIRMS-provided discriminator before processing."
        )
    return result
