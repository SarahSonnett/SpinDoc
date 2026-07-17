import numpy as np


def fourier(phase, *coeff):
    """Nth-order Fourier series model for a phased light curve.

    The data are already folded to a rotation phase in [0, 1), so the
    fundamental period in phase space is fixed to exactly 1 (one rotation per
    unit phase, omega = 2*pi). This guarantees the model is exactly periodic
    over [0, 1] -- the two ends meet at phase 0 and phase 1 for every fit. The
    rotation period itself is set by the folding and is *not* a free parameter
    of this model.

    coeff layout: [mean, phi_1, A_1, phi_2, A_2, ...]
    """
    omega = 2. * np.pi
    ret = coeff[0] + coeff[2] * np.sin(omega * phase + coeff[1])
    nord = int((len(coeff) - 1) / 2)
    i = 3
    for iord in range(nord - 1):
        ret += coeff[i] * np.sin((iord + 2) * omega * phase + coeff[i + 1])
        i += 2
    return ret


def fourier_binary(dt, *coeff):
    """Two-period Fourier series for binary or tumbling rotators.

    coeff layout: [P1, P2, mean1, A1, phi1, mean2, A2, phi2,
                   mean3, A3, phi3, mean4, A4, phi4]
    """
    omega1 = 2. * np.pi / coeff[0]
    omega2 = 2. * np.pi / coeff[1]
    ret = (coeff[2] + coeff[3] * np.sin(omega1 * dt + coeff[4])
           + coeff[5] + coeff[6] * np.sin(omega2 * dt + coeff[7])
           + coeff[8] + coeff[9] * np.sin(omega1 * dt + coeff[10])
           + coeff[11] + coeff[12] * np.sin(omega2 * dt + coeff[13]))
    return ret
