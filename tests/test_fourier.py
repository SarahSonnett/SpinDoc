"""Regression tests for the phased Fourier model.

The light curve is folded to a rotation phase in [0, 1) before fitting, so the
model must be exactly periodic over that interval: the value at phase 0 must
equal the value at phase 1. Guards against reintroducing a free fundamental
period, which lets the fit land on a fractional number of cycles and leaves the
two ends of the phased model not meeting.

Runs standalone (``python tests/test_fourier.py``) or under pytest.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spindoc import fourier


def test_model_closes_at_phase_wrap():
    rng = np.random.default_rng(0)
    for order in (2, 3, 5, 8):
        coeff = rng.normal(size=order * 2 + 1)
        start = fourier(np.array([0.0]), *coeff)[0]
        end = fourier(np.array([1.0]), *coeff)[0]
        assert np.isclose(start, end, atol=1e-9), (
            f"order {order}: model(0)={start} != model(1)={end}"
        )


def test_model_is_periodic_across_integer_shifts():
    # Shifting phase by any integer number of rotations must not change the model.
    rng = np.random.default_rng(1)
    coeff = rng.normal(size=3 * 2 + 1)
    phase = np.linspace(0.0, 1.0, 50)
    baseline = fourier(phase, *coeff)
    for shift in (1, 2, 7):
        assert np.allclose(fourier(phase + shift, *coeff), baseline, atol=1e-9)


if __name__ == "__main__":
    test_model_closes_at_phase_wrap()
    test_model_is_periodic_across_integer_shifts()
    print("OK: phased Fourier model is periodic over [0, 1]")
