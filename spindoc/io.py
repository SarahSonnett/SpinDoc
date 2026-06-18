import numpy as np


def read_photometry(infile, fmt=None):
    """Read calibrated photometry file.

    Parameters
    ----------
    infile : str
        Path to whitespace-delimited photometry file with a one-line header.
    fmt : str or None
        If None, uses the default column layout (MJD in col 7, mags in col 8).
        If any other value, uses the compact layout (MJD in col 1, mags in col 5).

    Returns
    -------
    dict with keys: time, helio, geo, alpha, mags, merr, filters
    """
    if fmt is None:
        cols = dict(time=7, helio=1, geo=2, alpha=3, mags=8, merr=10, filters=5)
    else:
        cols = dict(time=1, helio=2, geo=3, alpha=4, mags=5, merr=6, filters=7)

    kw = dict(unpack=True, skip_header=1)
    return {
        'time':    np.genfromtxt(infile, dtype=float, usecols=cols['time'],    **kw),
        'helio':   np.genfromtxt(infile, dtype=float, usecols=cols['helio'],   **kw),
        'geo':     np.genfromtxt(infile, dtype=float, usecols=cols['geo'],     **kw),
        'alpha':   np.genfromtxt(infile, dtype=float, usecols=cols['alpha'],   **kw),
        'mags':    np.genfromtxt(infile, dtype=float, usecols=cols['mags'],    **kw),
        'merr':    np.genfromtxt(infile, dtype=float, usecols=cols['merr'],    **kw),
        'filters': np.genfromtxt(infile, dtype=str,   usecols=cols['filters'], **kw),
    }
