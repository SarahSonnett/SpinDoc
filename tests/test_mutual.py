"""Tests for the binary mutual-event detectors.

Injects a known recurring eclipse into flat residuals and checks that the
box-least-squares search recovers its orbital period, and that a single injected
dip is flagged by the individual-event detector.

Runs standalone (``python tests/test_mutual.py``) or under pytest.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spindoc import bls_search, find_events


def _flat_residuals(n=400, span=120.0, sigma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, span, n))
    err = np.full_like(t, sigma)
    resid = rng.normal(0.0, sigma, size=t.size)
    return t, resid, err


def test_bls_recovers_injected_orbital_period():
    t, resid, err = _flat_residuals()
    porb = 18.3
    ph = (t % porb) / porb
    # eclipse (fainter -> positive residual) in a narrow phase window
    resid = resid + np.where(np.abs(((ph + 0.5) % 1) - 0.5) < 0.04, 0.08, 0.0)

    periods = np.linspace(6.0, 40.0, 1500)
    res = bls_search(t, resid, err, periods)
    assert res["best"] is not None
    assert abs(res["best"]["period"] - porb) < 0.2
    assert res["best"]["depth"] > 0            # fainter during event
    assert res["best"]["zscore"] > 10


def test_bls_null_is_weak():
    # Flat noise: the best "detection" must be far weaker than a real signal.
    t, resid, err = _flat_residuals(seed=3)
    periods = np.linspace(6.0, 40.0, 1500)
    res = bls_search(t, resid, err, periods)
    assert res["best"]["zscore"] < 8


def test_two_box_recovers_both_eclipses():
    # Primary + shallower secondary half a period apart: the two-box model must
    # recover the FULL orbital period and both dips, where a single-box search
    # aliases to half the period.
    t, resid, err = _flat_residuals(seed=0)
    porb = 18.3
    ph = (t % porb) / porb
    prim = np.abs(((ph - 0.2 + 0.5) % 1) - 0.5) < 0.04
    sec = np.abs(((ph - 0.7 + 0.5) % 1) - 0.5) < 0.04
    resid = resid + np.where(prim, 0.08, 0.0) + np.where(sec, 0.04, 0.0)

    periods = np.linspace(6.0, 40.0, 1500)
    one = bls_search(t, resid, err, periods, two_box=False)
    two = bls_search(t, resid, err, periods, two_box=True)

    assert two["best"] is not None
    assert abs(two["best"]["period"] - porb) < 0.2          # full orbital period
    assert two["best"]["depth1"] > 0 and two["best"]["depth2"] > 0
    # the two recovered dips sit ~half a period apart
    dphase = abs(two["best"]["phase1"] - two["best"]["phase2"]) % 1.0
    assert abs(min(dphase, 1 - dphase) - 0.5) < 0.1
    # single-box aliases to about half the true period
    assert abs(one["best"]["period"] - porb / 2) < 0.3


def test_find_events_flags_single_dip():
    t, resid, err = _flat_residuals(seed=1)
    resid[(t > 59.0) & (t < 60.0)] += 0.12     # a burst of faint points
    events = find_events(t, resid, err, nsigma=4.0)
    assert len(events) >= 1
    hit = max(events, key=lambda e: e["max_zscore"])
    assert 58.5 < hit["t_start"] and hit["t_end"] < 60.5
    assert hit["mean_depth"] > 0


if __name__ == "__main__":
    test_bls_recovers_injected_orbital_period()
    test_bls_null_is_weak()
    test_find_events_flags_single_dip()
    print("OK: mutual-event detectors recover injected signals and stay quiet on noise")
