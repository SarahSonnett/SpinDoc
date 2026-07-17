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

    # Read the whole table once as strings, then cast per column. (Reading each
    # column with a separate genfromtxt call would re-parse the file 7 times.)
    table = np.atleast_2d(np.genfromtxt(infile, dtype=str, skip_header=1))

    return {
        'time':    table[:, cols['time']].astype(float),
        'helio':   table[:, cols['helio']].astype(float),
        'geo':     table[:, cols['geo']].astype(float),
        'alpha':   table[:, cols['alpha']].astype(float),
        'mags':    table[:, cols['mags']].astype(float),
        'merr':    table[:, cols['merr']].astype(float),
        'filters': table[:, cols['filters']],
    }
