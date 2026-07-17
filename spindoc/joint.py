import numpy as np

from .hg import HGfunction


def hg_fourier(X, *params):
    """Joint IAU H-G phase function + zero-mean Fourier rotation model.

    Evaluates the solar phase function and the rotational light curve together at
    a fixed rotation phase (the data are already folded to phase), so that H, G,
    and the Fourier harmonics can be fit simultaneously and share a single
    covariance matrix. The rotational series carries no constant term -- the mean
    brightness level is set by H -- which removes the degeneracy between H and a
    free Fourier mean.

    Parameters
    ----------
    X : array_like, shape (2, N)
        Row 0: solar phase angle alpha (degrees).
        Row 1: rotation phase in [0, 1).
    params : floats
        Layout ``[H, G, A_1, phi_1, A_2, phi_2, ...]``. The Fourier order is
        inferred as ``(len(params) - 2) // 2``.

    Returns
    -------
    ndarray
        Model reduced magnitude at each observation.
    """
    alpha = np.asarray(X[0])
    phase = np.asarray(X[1])
    H, G = params[0], params[1]
    fcoeff = params[2:]
    order = len(fcoeff) // 2
    omega = 2. * np.pi
    ret = HGfunction(alpha, H, G)
    for n in range(1, order + 1):
        amp = fcoeff[2 * n - 2]
        phi = fcoeff[2 * n - 1]
        ret = ret + amp * np.sin(omega * n * phase + phi)
    return ret
