import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("evaluate_spatial_holdout", SCRIPTS / "evaluate_spatial_holdout.py")
assert SPEC and SPEC.loader
holdout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(holdout)


def test_block_bootstrap_returns_deterministic_macro_and_per_class_intervals():
    truth = np.array(["agricultural_burning", "industrial_fire", "mining_activity", "gas_flare"] * 2)
    predicted = np.array(["agricultural_burning", "industrial_fire", "agricultural_burning", "gas_flare"] * 2)
    blocks = np.array(["b1"] * 4 + ["b2"] * 4)

    first = holdout.bootstrap_block_f1(truth, predicted, blocks, iterations=50, seed=7)
    second = holdout.bootstrap_block_f1(truth, predicted, blocks, iterations=50, seed=7)

    assert first == second
    assert first["blocks"] == 2
    assert first["iterations"] == 50
    assert set(first["per_class_f1_95_ci"]) == set(holdout.CLASSES)
    assert len(first["macro_f1_95_ci"]) == 2


def test_block_bootstrap_reports_unavailable_ci_for_one_block():
    result = holdout.bootstrap_block_f1(
        np.array(["agricultural_burning"]),
        np.array(["agricultural_burning"]),
        np.array(["one-block"]),
        iterations=10,
    )
    assert result["iterations"] == 0
    assert result["macro_f1_95_ci"] is None
