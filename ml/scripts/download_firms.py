#live data ingestion of NASA Firms

from dotenv import load_dotenv
import os
import argparse
from datetime import datetime
import requests
from pathlib import Path
import json
import hashlib
from datetime import datetime,timezone,timedelta


load_dotenv()

map_key=os.getenv("MAP_KEY")

if not map_key:
    raise ValueError("No key in .env")

availability_url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/"
    f"{map_key}/VIIRS_NOAA20_SP"
)
availability_response = requests.get(availability_url, timeout=60)

if not availability_response.ok:
    print("Could not check SP availability")
    print("Status:", availability_response.status_code)
    raise SystemExit(1)

print(availability_response.text[:2000])

def get_arguments():
    parser = argparse.ArgumentParser(
        description="Download NASA FIRMS thermal anomaly data"
    )

    parser.add_argument(
        "--source",
        default="VIIRS_NOAA20_SP",
        help="FIRMS source/product. Default: VIIRS_NOAA20_SP"
    )

    location_group=parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument(
        "--country",
        help="Three-letter country code,for example IND"
    )
    location_group.add_argument(
        "--bbox",
        help="Bounding box:west,south,east,north . Example 68,8,97,37"
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format"
    )

    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format"
    )

    return parser.parse_args()
def validate_arguments(args):
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Dates must use YYYY-MM-DD format, for example 2026-09-08")

    if start_date > end_date:
        raise ValueError("start-date cannot be later than end-date")

    bbox_values = None

    if args.bbox:
        try:
            bbox_values = [float(value.strip()) for value in args.bbox.split(",")]
        except ValueError:
            raise ValueError("bbox must contain four numbers: west,south,east,north")

        if len(bbox_values) != 4:
            raise ValueError("bbox must contain exactly four values: west,south,east,north")

        west, south, east, north = bbox_values

        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("bbox longitude values must be between -180 and 180")

        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValueError("bbox latitude values must be between -90 and 90")

        if west >= east:
            raise ValueError("bbox west longitude must be less than east longitude")

        if south >= north:
            raise ValueError("bbox south latitude must be less than north latitude")
    if args.country:
        print(
            "NASA FIRMS country endpoint is currently unavailable. "
            "Please use --bbox instead."
        )
        raise SystemExit(1)
    return start_date, end_date, bbox_values

def build_firms_url(map_key, source, country, bbox_values, start_date, day_range=1):
    base_url = "https://firms.modaps.eosdis.nasa.gov/api"

    if country:
        return (
            f"{base_url}/country/csv/"
            f"{map_key}/{source}/{country.upper()}/{day_range}/{start_date}"
        )

    bbox_text = ",".join(str(value) for value in bbox_values)

    return (
        f"{base_url}/area/csv/"
        f"{map_key}/{source}/{bbox_text}/{day_range}/{start_date}"
    )


def save_raw_download(response,args,start_date,end_date,bbox_values,record_count):
    output_dir=Path("data/raw/firms")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.country:
        location_name=f"country_{args.country.upper()}"
        query_type="country"
    else:
        bbox_name="_".join(str(value) for value in bbox_values)
        location_name=f"bbox_{bbox_name}"
        query_type="bounding_box"
    file_stem=(
        f"{args.source}_{location_name}_"
        f"{start_date}_to_{end_date}"
    )
    csv_path=output_dir/f"{file_stem}.csv"
    metadata_path=output_dir/f"{file_stem}.metadata.json"

    csv_path.write_bytes(response.content)
    metadata={
        "source":args.source,
        "query_type": query_type,
        "country": args.country.upper() if args.country else None,
        "bounding_box": bbox_values,
        "requested_start_date": str(start_date),
        "requested_end_date": str(end_date),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type"),
        "raw_file": csv_path.name,
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "note": "Raw NASA FIRMS response preserved without transformation.",
        "record_count": record_count,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    return csv_path,metadata_path
def download_date_range(args, start_date, end_date, bbox_values):

    current_start = start_date

    while current_start <= end_date:

        # FIRMS allows a maximum of five days in one area request.
        current_end = min(
            current_start + timedelta(days=4),
            end_date
        )

        day_range = (
            current_end - current_start
        ).days + 1

        url = build_firms_url(
            map_key=map_key,
            source=args.source,
            country=args.country,
            bbox_values=bbox_values,
            start_date=current_start,
            day_range=day_range
        )

        print(
            f"Downloading {current_start} to {current_end} "
            f"({day_range} day(s))..."
        )

        response = requests.get(
            url,
            timeout=60
        )

        if not response.ok:
            print("NASA FIRMS request failed")
            print("Status code:", response.status_code)
            print("NASA response:", response.text[:500])
            raise SystemExit(1)

        record_count = max(
            0,
            len(response.text.splitlines()) - 1
        )

        if record_count == 0:
            print(
                f"No detections returned for "
                f"{current_start} to {current_end}."
            )
        else:
            print(
                f"Received {record_count} detection(s)."
            )

        csv_path, metadata_path = save_raw_download(
            response=response,
            args=args,
            start_date=current_start,
            end_date=current_end,
            bbox_values=bbox_values,
            record_count=record_count
        )

        print("Saved:", csv_path)
        print("Metadata:", metadata_path)

        current_start = current_end + timedelta(days=1)


if __name__ == "__main__":
    args = get_arguments()

    try:
        start_date, end_date, bbox_values = validate_arguments(args)
    except ValueError as error:
        print(f"Input error: {error}")
        raise SystemExit(1)

    download_date_range(
        args=args,
        start_date=start_date,
        end_date=end_date,
        bbox_values=bbox_values
    )