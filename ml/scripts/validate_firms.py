"""
Validate FIRMS NOAA-20 SP raw CSVs.

Pipeline:
    146 immutable raw CSVs
        -> file/schema checks
        -> cleaning and standardization
        -> exact-duplicate detection
        -> one processed dataset + validation report

Do NOT modify anything in data/raw/firms/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw/firms")
PROCESSED_DIR = Path("data/processed")

RAW_GLOB = "VIIRS_NOAA20_SP_*.csv"

PROCESSED_CSV = (
    PROCESSED_DIR
    / "firms_noaa20_sp_bbox_68_8_97_37_2024-06-01_to_2026-05-31.csv"
)
REPORT_JSON = (
    PROCESSED_DIR
    / "firms_noaa20_sp_bbox_68_8_97_37_validation_report.json"
)
REJECTED_CSV = PROCESSED_DIR / "firms_noaa20_sp_rejected_records.csv"

REQUIRED_COLUMNS = {
    "latitude",
    "longitude",
    "bright_ti4",
    "bright_ti5",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "frp",
    "daynight",
}

DUPLICATE_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "bright_ti4",
    "bright_ti5",
    "frp",
]

# Looser key for the cross-check (drops radiometric fields).
LOOSE_DUPLICATE_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
]


# ---------------------------------------------------------------------------
# Step 1: File discovery + schema + provenance + merge
# ---------------------------------------------------------------------------

def discover_raw_files() -> list[Path]:
    raw_files = sorted(RAW_DIR.glob(RAW_GLOB))
    if not raw_files:
        raise FileNotFoundError(
            f"No NOAA-20 SP raw CSV files found in {RAW_DIR} "
            f"(pattern: {RAW_GLOB})."
        )
    return raw_files


def load_and_validate_files(
    raw_files: list[Path],
) -> tuple[pd.DataFrame, int, list[dict]]:
    """Read each CSV, validate schema, tag provenance, and concatenate.

    Returns (combined_df, total_rows, input_file_summary).
    """
    all_frames: list[pd.DataFrame] = []
    input_file_summary: list[dict] = []
    total_rows = 0

    for file_path in raw_files:
        df = pd.read_csv(file_path)

        # Step 2: schema check
        missing_columns = REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing: {sorted(missing_columns)}"
            )

        # Step 3: provenance
        df["source_file"] = file_path.name
        df["data_source"] = "NASA FIRMS"
        df["product"] = "VIIRS_NOAA20_SP"

        row_count = len(df)
        total_rows += row_count
        print(f"{file_path.name}: {row_count} rows")

        input_file_summary.append(
            {
                "file_name": file_path.name,
                "row_count": row_count,
                "file_size_bytes": file_path.stat().st_size,
            }
        )

        all_frames.append(df)

    # Step 4: merge
    combined_df = pd.concat(all_frames, ignore_index=True)
    return combined_df, total_rows, input_file_summary


# ---------------------------------------------------------------------------
# Step 5: Validation rules
# ---------------------------------------------------------------------------

def build_rejection_reasons(df: pd.DataFrame) -> pd.Series:
    """Return a Series of rejection reasons (empty string = valid)."""
    reasons = pd.Series([""] * len(df), index=df.index, dtype="object")

    def _mark(mask: pd.Series, reason: str) -> None:
        current = reasons.loc[mask]
        reasons.loc[mask] = current.where(current == "", current + "; ") + reason

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    frp = pd.to_numeric(df["frp"], errors="coerce")
    acq_date = pd.to_datetime(df["acq_date"], errors="coerce")

    _mark(lat.isna() | (lat < -90) | (lat > 90),
          "latitude out of range or not numeric")
    _mark(lon.isna() | (lon < -180) | (lon > 180),
          "longitude out of range or not numeric")
    _mark(frp.isna() | (frp < 0),
          "frp negative or not numeric")
    _mark(acq_date.isna(),
          "acq_date invalid")
    _mark(~df["daynight"].isin(["D", "N"]),
          "daynight not D/N")
    _mark(df["instrument"].astype(str).str.upper() != "VIIRS",
          "instrument not VIIRS")

    return reasons


def split_valid_and_rejected(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reasons = build_rejection_reasons(df)
    rejected_mask = reasons != ""

    rejected = df.loc[rejected_mask].copy()
    rejected["rejection_reason"] = reasons.loc[rejected_mask]

    valid = df.loc[~rejected_mask].copy()
    return valid, rejected


# ---------------------------------------------------------------------------
# Step 6: Exact-duplicate detection
# ---------------------------------------------------------------------------

def detect_exact_duplicates(df: pd.DataFrame) -> pd.Series:
    return df.duplicated(subset=DUPLICATE_COLUMNS, keep="first")


def detect_loose_duplicates(df: pd.DataFrame) -> pd.Series:
    return df.duplicated(subset=LOOSE_DUPLICATE_COLUMNS, keep="first")


# ---------------------------------------------------------------------------
# Step 7: Outputs
# ---------------------------------------------------------------------------

def write_outputs(
    processed: pd.DataFrame,
    rejected: pd.DataFrame,
    report: dict,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    processed.to_csv(PROCESSED_CSV, index=False)
    rejected.to_csv(REJECTED_CSV, index=False)

    with REPORT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1: discovery
    raw_files = discover_raw_files()
    print("Raw CSV files found:", len(raw_files))

    # Steps 2-4: schema, provenance, merge
    combined_df, total_rows, input_file_summary = load_and_validate_files(
        raw_files
    )
    print("Total raw detection records:", total_rows)
    print("Merged rows:", len(combined_df))

    if len(combined_df) != total_rows:
        raise RuntimeError(
            "Merged row count does not match total raw rows; "
            "aborting before validation."
        )

    # Step 5: validation rules
    valid_df, rejected_df = split_valid_and_rejected(combined_df)
    print("Rows passing validation:", len(valid_df))
    print("Rows rejected:", len(rejected_df))

    # Step 6: exact duplicates (inspect only, do not delete)
    dup_mask = detect_exact_duplicates(valid_df)
    duplicate_count = int(dup_mask.sum())
    print("Exact duplicate rows (by FIRMS key fields):", duplicate_count)

    # Loose duplicate cross-check (same lat/lon/time/satellite/instrument,
    # ignoring radiometric fields).
    loose_dup_mask = detect_loose_duplicates(valid_df)
    loose_duplicate_count = int(loose_dup_mask.sum())
    print(
        "Loose duplicate rows (lat/lon/time/satellite/instrument):",
        loose_duplicate_count,
    )

    # Acquisition datetime range (for the report).
    acq_dt = pd.to_datetime(
        valid_df["acq_date"].astype(str)
        + " "
        + valid_df["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        errors="coerce",
        utc=True,
    )
    min_dt = acq_dt.min()
    max_dt = acq_dt.max()

    # Categorical distributions.
    confidence_counts = (
        valid_df["confidence"].value_counts(dropna=False).to_dict()
    )
    daynight_counts = (
        valid_df["daynight"].value_counts(dropna=False).to_dict()
    )
    satellite_counts = (
        valid_df["satellite"].value_counts(dropna=False).to_dict()
    )
    instrument_counts = (
        valid_df["instrument"].value_counts(dropna=False).to_dict()
    )

    # Rejection counts by reason.
    rejection_counts: dict[str, int] = {}
    if not rejected_df.empty:
        rejection_counts = (
            rejected_df["rejection_reason"]
            .value_counts(dropna=False)
            .to_dict()
        )

    # Step 7: write processed output + validation report.
    report = {
        "pipeline": "validate_firms.py",
        "source": "NASA FIRMS",
        "product": "VIIRS_NOAA20_SP",
        "raw_directory": str(RAW_DIR),
        "raw_file_pattern": RAW_GLOB,
        "raw_file_count": len(raw_files),
        "total_raw_records": total_rows,
        "merged_records": len(combined_df),
        "valid_records": len(valid_df),
        "rejected_records": len(rejected_df),
        "exact_duplicate_records": duplicate_count,
        "loose_duplicate_records": loose_duplicate_count,
        "duplicates_removed": False,
        "duplicate_columns": DUPLICATE_COLUMNS,
        "loose_duplicate_columns": LOOSE_DUPLICATE_COLUMNS,
        "required_columns": sorted(REQUIRED_COLUMNS),
        "minimum_acquisition_datetime_utc": (
            min_dt.isoformat() if pd.notna(min_dt) else None
        ),
        "maximum_acquisition_datetime_utc": (
            max_dt.isoformat() if pd.notna(max_dt) else None
        ),
        "rejection_counts": rejection_counts,
        "confidence_value_counts": confidence_counts,
        "daynight_value_counts": daynight_counts,
        "satellite_value_counts": satellite_counts,
        "instrument_value_counts": instrument_counts,
        "has_type_column": "type" in valid_df.columns,
        "input_file_summary": input_file_summary,
        "processed_csv": str(PROCESSED_CSV),
        "rejected_csv": str(REJECTED_CSV),
        "notes": [
            "Raw FIRMS CSV files were not modified.",
            "VIIRS confidence remains categorical: l, n, h.",
            "FIRMS type is preserved if present but is not used as a "
            "training label.",
            "Exact duplicates are flagged but retained for later review.",
            "Loose duplicates ignore bright_ti4, bright_ti5, and frp.",
        ],
    }

    write_outputs(valid_df, rejected_df, report)

    print("\nWrote:")
    print(f"  {PROCESSED_CSV}")
    print(f"  {REJECTED_CSV}")
    print(f"  {REPORT_JSON}")


if __name__ == "__main__":
    main()