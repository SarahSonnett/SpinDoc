"""Tests for the joint H-G + Fourier model and fit.

Checks the model reduces to the phase function when the rotational amplitudes
are zero, that its rotational part is periodic in phase, and that a joint fit on
the sample dataset recovers physically sensible H, G, and amplitude at the known
rotation period of asteroid 16152 (P = 22.931 h).

Runs standalone (``python tests/test_joint.py``) or under pytest.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spindoc import hg_fourier, HGfunction, read_photometry
import period_search as ps

SAMPLE = os.path.join(REPO_ROOT, "docs", "Target_Calibrated_FinalErr_rp_cleaned.txt")


def test_zero_amplitude_reduces_to_phase_function():
    alpha = np.linspace(0.5, 8.0, 40)
    phase = np.linspace(0.0, 1.0, 40)
    # order-2 model with both amplitudes zero -> pure H-G phase function
    params = [12.0, 0.35, 0.0, 0.3, 0.0, 1.1]
    model = hg_fourier(np.vstack([alpha, phase]), *params)
    assert np.allclose(model, HGfunction(alpha, 12.0, 0.35), atol=1e-9)


def test_rotational_part_is_phase_periodic():
    # At fixed alpha, shifting phase by an integer must not change the model.
    alpha = np.full(30, 3.0)
    phase = np.linspace(0.0, 1.0, 30)
    params = [12.0, 0.35, 0.05, 0.8, 0.18, 1.8]
    X0 = np.vstack([alpha, phase])
    X1 = np.vstack([alpha, phase + 3])
    assert np.allclose(hg_fourier(X0, *params), hg_fourier(X1, *params), atol=1e-9)


def test_joint_fit_recovers_sensible_parameters():
    raw = read_photometry(SAMPLE)
    data = ps.build_filter_data(raw, "rp", "16152", 0.0, None)
    res = ps.fit_joint(data, 22.931, 2)
    assert res is not None
    assert 11.5 < res["H"] < 12.1
    assert 0.2 < res["G"] < 0.5
    assert 0.35 < res["amp"] < 0.45
    assert res["chi2nu"] < 2.0                # dof = N - k
    assert len(res["coeff"]) == 2 + 2 * 2     # H, G + (A_n, phi_n) for order 2
    assert res["cov"].shape == (6, 6)


if __name__ == "__main__":
    test_zero_amplitude_reduces_to_phase_function()
    test_rotational_part_is_phase_periodic()
    test_joint_fit_recovers_sensible_parameters()
    print("OK: joint H-G + Fourier model and fit behave correctly")
