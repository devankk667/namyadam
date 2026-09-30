import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "firms_ids.py"
SPEC = importlib.util.spec_from_file_location("firms_ids", SCRIPT)
assert SPEC and SPEC.loader
firms_ids = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(firms_ids)


def sample_rows():
    return pd.DataFrame([
        {"satellite": "N20", "instrument": "VIIRS", "acq_date": "2024-06-01", "acq_time": 1234,
         "latitude": 20.123456, "longitude": 73.234567, "scan": 0.4, "track": 0.5,
         "bright_ti4": 330.0, "bright_ti5": 290.0, "frp": 4.2},
        {"satellite": "N20", "instrument": "VIIRS", "acq_date": "2024-06-02", "acq_time": 1235,
         "latitude": 21.123456, "longitude": 74.234567, "scan": 0.4, "track": 0.5,
         "bright_ti4": 331.0, "bright_ti5": 291.0, "frp": 4.3},
    ])


def test_observation_ids_do_not_change_when_rows_reordered():
    frame = sample_rows()
    first = firms_ids.ensure_observation_ids(frame)
    reversed_rows = firms_ids.ensure_observation_ids(frame.iloc[::-1].reset_index(drop=True))
    assert dict(zip(first.acq_date, first.obs_id)) == dict(zip(reversed_rows.acq_date, reversed_rows.obs_id))


def test_duplicate_natural_keys_fail_instead_of_using_row_number():
    frame = pd.concat([sample_rows().iloc[[0]], sample_rows().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="identical natural-key fields"):
        firms_ids.ensure_observation_ids(frame)


def test_existing_nonempty_observation_ids_are_preserved():
    frame = sample_rows()
    frame["obs_id"] = ["FIRMS-a", "FIRMS-b"]
    assert firms_ids.ensure_observation_ids(frame).obs_id.tolist() == ["FIRMS-a", "FIRMS-b"]


def test_legacy_row_number_ids_are_replaced_with_stable_ids():
    frame = sample_rows()
    frame["obs_id"] = [0, 1]
    generated = firms_ids.ensure_observation_ids(frame)
    assert generated.obs_id.str.startswith("FIRMS-").all()
    assert generated.obs_id.tolist() != ["0", "1"]
