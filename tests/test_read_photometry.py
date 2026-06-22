"""Smoke test for the photometry reader, exercised against the sample dataset.

Uses the sample file shipped under ``docs/`` — the r'-band photometry that
generates the example figures in the README — to confirm that the default
input format is parsed correctly.

Runs standalone (``python tests/test_read_photometry.py``) or under pytest.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spindoc import read_photometry

SAMPLE = os.path.join(REPO_ROOT, "docs", "Target_Calibrated_FinalErr_rp_cleaned.txt")


def test_sample_file_parses():
    data = read_photometry(SAMPLE)

    expected_keys = {"time", "helio", "geo", "alpha", "mags", "merr", "filters"}
    assert set(data) == expected_keys

    # All seven columns must have the same length (one row per observation).
    lengths = {len(v) for v in data.values()}
    assert lengths == {434}, f"expected 434 rows in every column, got {lengths}"

    # Single-filter dataset.
    assert set(data["filters"].tolist()) == {"rp"}

    # No missing values made it through the column mapping.
    for key in ("time", "helio", "geo", "alpha", "mags", "merr"):
        assert not np.isnan(data[key]).any(), f"NaNs found in {key}"

    # Physical sanity checks confirm columns map to the right quantities.
    assert (data["merr"] > 0).all()
    assert 4.9 < data["helio"].min() and data["helio"].max() < 5.1   # Rhelio (AU)
    assert 4.0 < data["geo"].min() and data["geo"].max() < 4.2       # Delta (AU)
    assert 0 < data["alpha"].min() and data["alpha"].max() < 10      # phase angle (deg)
    assert 18 < data["mags"].min() and data["mags"].max() < 20       # calibrated mag
    assert 58701 < data["time"].min() and data["time"].max() < 58752  # MJD


if __name__ == "__main__":
    test_sample_file_parses()
    print("OK: sample photometry file parsed correctly")
