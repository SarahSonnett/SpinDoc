import os
from math import trunc, modf

import numpy as np


def lighttimecorrection(geo):
    return np.asarray(-0.005778 * geo)


def makenewdir(dirname):
    try:
        os.mkdir(dirname)
    except OSError:
        pass


def chisqrpdf(data, model, error):
    chisq = sum(((d - m) ** 2) / (e ** 2) for d, m, e in zip(data, model, error))
    return chisq / (len(data) - 1)


def converttoUT(mjd):
    # adapted from https://gist.github.com/jiffyclub/1294443
    jd = mjd + 2400000.5
    F, I = modf(jd)
    I = int(I)
    A = trunc((I - 1867216.25) / 36524.25)
    B = I + 1 + A - trunc(A / 4.) if I > 2299160 else I
    C = B + 1524
    D = trunc((C - 122.1) / 365.25)
    E = trunc(365.25 * D)
    G = trunc((C - E) / 30.6001)
    day = C - E + F - trunc(30.6001 * G)
    month = G - 1 if G < 13.5 else G - 13
    year = D - 4716 if month > 2.5 else D - 4715
    return year, month, day


def converttoMJD(utc):
    """Convert a UTC date [year, month, day] to Modified Julian Date.

    Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
    4th ed., Duffet-Smith and Zwart, 2011.
    """
    year, month, day = float(utc[0]), float(utc[1]), float(utc[2])
    if month in (1, 2):
        yearp, monthp = year - 1, month + 12
    else:
        yearp, monthp = year, month
    if (year < 1582
            or (year == 1582 and month < 10)
            or (year == 1582 and month == 10 and day < 15)):
        B = 0
    else:
        A = trunc(yearp / 100.)
        B = 2 - A + trunc(A / 4.)
    C = trunc((365.25 * yearp) - 0.75) if yearp < 0 else trunc(365.25 * yearp)
    D = trunc(30.6001 * (monthp + 1))
    jd = B + C + D + day + 1720994.5
    return jd - 2400000.5
