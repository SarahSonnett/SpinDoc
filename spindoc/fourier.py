import numpy as np


def fourier(phase, *coeff):
    """Nth-order Fourier series model for a phased light curve.

    coeff layout: [period, mean, phi_1, A_1, phi_2, A_2, ...]
    """
    period = coeff[0]
    omega = 2. * np.pi / period
    ret = coeff[1] + coeff[3] * np.sin(omega * phase + coeff[2])
    nord = int((len(coeff) - 2) / 2)
    i = 3
    for iord in range(nord - 1):
        ret += coeff[i + 1] * np.sin((iord + 2) * omega * phase + coeff[i + 2])
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
