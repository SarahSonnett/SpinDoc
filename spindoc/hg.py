import numpy as np


def HGfunction(x, H, G):
    """IAU H-G phase function for asteroid photometry."""
    aradians = np.radians(x)
    W = np.exp(-90.56 * np.tan(aradians / 2.) ** 2.)

    sin_a = np.sin(aradians)
    tan_a2 = np.tan(aradians / 2.)
    denom = 0.119 + 1.341 * sin_a - 0.754 * sin_a ** 2.

    phi1 = W * (1. - 0.986 * sin_a / denom) + (1. - W) * np.exp(-3.332 * tan_a2 ** 0.631)
    phi2 = W * (1. - 0.238 * sin_a / denom) + (1. - W) * np.exp(-1.862 * tan_a2 ** 1.218)

    return H - 2.5 * np.log10((1. - G) * phi1 + G * phi2)
