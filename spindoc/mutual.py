"""Search model-subtracted residuals for mutual events of a binary asteroid.

After the rotational light curve and solar phase function have been removed, a
binary companion reveals itself through brief, recurring attenuations (mutual
eclipses/occultations) at the *orbital* period -- distinct from the primary's
rotation period. Because the object grows fainter during an event, in magnitude
space these are positive residual excursions.

Two complementary detectors are provided:

* ``bls_search`` -- a box-least-squares-style scan that folds the residuals at a
  grid of trial orbital periods and looks for a recurring dip (box), returning a
  detection statistic per period. Recurrence is the defining binary signature.
* ``find_events`` -- flags individual significant faint excursions localised in
  time, clustering nearby points into candidate events. Useful when coverage is
  sparse or only one event was caught.

Sign convention: ``resid = observed_mag - model_mag``; an eclipse makes the
object fainter, so events appear as POSITIVE residuals.
"""
import numpy as np


def bls_search(time, resid, err, periods, nbins=120, min_dur=0.02, max_dur=0.25):
    """Box-least-squares-style search for a recurring dip in the residuals.

    Parameters
    ----------
    time : array_like
        Observation times (any consistent unit; hours here). Trial ``periods``
        must share the same unit.
    resid : array_like
        Model-subtracted residual magnitudes (positive = fainter than model).
    err : array_like
        1-sigma uncertainty per point.
    periods : array_like
        Trial orbital periods to scan.
    nbins : int
        Number of phase bins per fold.
    min_dur, max_dur : float
        Minimum/maximum event duration as a fraction of the orbital period.

    Returns
    -------
    dict
        ``periods`` (array), ``power`` (best detection z-score per period), and
        ``best`` (dict for the peak period: period, depth, zscore, phase, width),
        or ``best=None`` if nothing could be evaluated.
    """
    time = np.asarray(time, dtype=float)
    resid = np.asarray(resid, dtype=float)
    err = np.asarray(err, dtype=float)
    periods = np.asarray(periods, dtype=float)

    w = 1.0 / err ** 2
    W = w.sum()
    WY = np.sum(w * resid)

    dmin = max(int(min_dur * nbins), 1)
    dmax = max(int(max_dur * nbins), dmin)

    power = np.zeros(len(periods))
    best = None

    for pi, P in enumerate(periods):
        phase = (time % P) / P
        b = np.minimum((phase * nbins).astype(int), nbins - 1)
        Sw = np.bincount(b, weights=w, minlength=nbins)
        Swy = np.bincount(b, weights=w * resid, minlength=nbins)

        # Double the binned arrays so a box may wrap past phase 1 -> 0.
        cSw = np.concatenate([[0.0], np.cumsum(np.concatenate([Sw, Sw]))])
        cSwy = np.concatenate([[0.0], np.cumsum(np.concatenate([Swy, Swy]))])

        bestz = 0.0
        bestrec = None
        for d in range(dmin, dmax + 1):
            inSw = cSw[d:d + nbins] - cSw[0:nbins]
            inSwy = cSwy[d:d + nbins] - cSwy[0:nbins]
            outSw = W - inSw
            valid = (inSw > 0) & (outSw > 0)
            if not valid.any():
                continue
            mu_in = np.where(inSw > 0, inSwy / np.where(inSw > 0, inSw, 1.0), 0.0)
            mu_out = np.where(outSw > 0, (WY - inSwy) / np.where(outSw > 0, outSw, 1.0), 0.0)
            depth = mu_in - mu_out                       # >0 => fainter in box
            sig = np.sqrt(1.0 / np.where(inSw > 0, inSw, np.inf)
                          + 1.0 / np.where(outSw > 0, outSw, np.inf))
            z = np.where(valid & (depth > 0), depth / sig, 0.0)
            k = int(np.argmax(z))
            if z[k] > bestz:
                bestz = z[k]
                bestrec = dict(period=float(P), depth=float(depth[k]),
                               zscore=float(z[k]),
                               phase=float(((k + d / 2.0) % nbins) / nbins),
                               width=float(d) / nbins)
        power[pi] = bestz
        if bestrec is not None and (best is None or bestz > best['zscore']):
            best = bestrec

    return dict(periods=periods, power=power, best=best)


def find_events(time, resid, err, nsigma=4.0, min_points=2, max_gap=None):
    """Flag individual significant faint excursions, clustered in time.

    A point is significant if ``resid / err > nsigma`` (fainter than the model).
    Significant points closer together than ``max_gap`` in time are grouped into
    one candidate event; clusters with at least ``min_points`` points are kept.

    Returns a list of event dicts (t_start, t_end, n_points, max_zscore,
    mean_depth), sorted by time.
    """
    time = np.asarray(time, dtype=float)
    resid = np.asarray(resid, dtype=float)
    err = np.asarray(err, dtype=float)

    order = np.argsort(time)
    t, r, e = time[order], resid[order], err[order]
    z = r / e

    if max_gap is None:
        dt = np.diff(t)
        max_gap = 5.0 * np.median(dt) if len(dt) else np.inf

    flagged = np.where(z > nsigma)[0]
    if len(flagged) == 0:
        return []

    clusters = [[flagged[0]]]
    for k in flagged[1:]:
        if t[k] - t[clusters[-1][-1]] <= max_gap:
            clusters[-1].append(k)
        else:
            clusters.append([k])

    events = []
    for c in clusters:
        if len(c) >= min_points:
            c = np.asarray(c)
            events.append(dict(t_start=float(t[c[0]]), t_end=float(t[c[-1]]),
                               n_points=int(len(c)), max_zscore=float(z[c].max()),
                               mean_depth=float(r[c].mean())))
    return events
