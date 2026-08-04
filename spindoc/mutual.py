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


def bls_search(time, resid, err, periods, nbins=120, min_dur=0.02, max_dur=0.25,
               two_box=False):
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
    two_box : bool
        If False (default), search for a single recurring dip. If True, fit an
        eclipsing-binary model with TWO dips per orbit: a primary and a secondary
        half a period apart (circular / tidally-locked orbit), of equal duration
        but independent depths. This is more sensitive when both eclipses are
        present, at the cost of the equal-spacing/equal-duration assumption.

    Returns
    -------
    dict
        ``periods`` (array), ``power`` (best statistic per period), and ``best``
        (dict for the peak period), or ``best=None`` if nothing could be
        evaluated. For the single-box search ``power``/``best['zscore']`` is a
        difference-of-means z-score and ``best`` carries (period, depth, zscore,
        phase, width). For the two-box search ``power`` is a chi-squared(2) signal
        residue and ``best`` carries (period, power, depth1, z1, phase1, depth2,
        z2, phase2, width).
    """
    time = np.asarray(time, dtype=float)
    resid = np.asarray(resid, dtype=float)
    err = np.asarray(err, dtype=float)
    periods = np.asarray(periods, dtype=float)

    w = 1.0 / err ** 2
    W = w.sum()
    WY = np.sum(w * resid)
    Ybar = WY / W

    shift = nbins // 2                          # half-period offset (two-box)
    dmin = max(int(min_dur * nbins), 1)
    dmax = max(int(max_dur * nbins), dmin)
    if two_box:
        dmax = min(dmax, shift)                 # keep the two boxes disjoint

    power = np.zeros(len(periods))
    best = None
    best_val = -np.inf

    for pi, P in enumerate(periods):
        phase = (time % P) / P
        b = np.minimum((phase * nbins).astype(int), nbins - 1)
        Sw = np.bincount(b, weights=w, minlength=nbins)
        Swy = np.bincount(b, weights=w * resid, minlength=nbins)

        # Double the binned arrays so a box may wrap past phase 1 -> 0.
        cSw = np.concatenate([[0.0], np.cumsum(np.concatenate([Sw, Sw]))])
        cSwy = np.concatenate([[0.0], np.cumsum(np.concatenate([Swy, Swy]))])

        bestval = 0.0
        bestrec = None
        for d in range(dmin, dmax + 1):
            inSw1 = cSw[d:d + nbins] - cSw[0:nbins]
            inSwy1 = cSwy[d:d + nbins] - cSwy[0:nbins]

            if not two_box:
                outSw = W - inSw1
                valid = (inSw1 > 0) & (outSw > 0)
                if not valid.any():
                    continue
                mu_in = np.where(inSw1 > 0, inSwy1 / np.where(inSw1 > 0, inSw1, 1.0), 0.0)
                mu_out = np.where(outSw > 0, (WY - inSwy1) / np.where(outSw > 0, outSw, 1.0), 0.0)
                depth = mu_in - mu_out                   # >0 => fainter in box
                sig = np.sqrt(1.0 / np.where(inSw1 > 0, inSw1, np.inf)
                              + 1.0 / np.where(outSw > 0, outSw, np.inf))
                val = np.where(valid & (depth > 0), depth / sig, 0.0)
                k = int(np.argmax(val))
                if val[k] > bestval:
                    bestval = val[k]
                    bestrec = dict(period=float(P), depth=float(depth[k]),
                                   zscore=float(val[k]),
                                   phase=float(((k + d / 2.0) % nbins) / nbins),
                                   width=float(d) / nbins)
            else:
                # Second box a half-period after the first, same width.
                inSw2 = cSw[shift + d:shift + d + nbins] - cSw[shift:shift + nbins]
                inSwy2 = cSwy[shift + d:shift + d + nbins] - cSwy[shift:shift + nbins]
                Wout = W - inSw1 - inSw2
                valid = (inSw1 > 0) & (inSw2 > 0) & (Wout > 0)
                if not valid.any():
                    continue
                mu1 = np.where(inSw1 > 0, inSwy1 / np.where(inSw1 > 0, inSw1, 1.0), 0.0)
                mu2 = np.where(inSw2 > 0, inSwy2 / np.where(inSw2 > 0, inSw2, 1.0), 0.0)
                muo = np.where(Wout > 0, (WY - inSwy1 - inSwy2) / np.where(Wout > 0, Wout, 1.0), 0.0)
                depth1 = mu1 - muo
                depth2 = mu2 - muo
                # chi^2 reduction of the two-box model over a flat mean (~chi^2 with 2 dof)
                reduction = (Wout * (muo - Ybar) ** 2
                             + inSw1 * (mu1 - Ybar) ** 2
                             + inSw2 * (mu2 - Ybar) ** 2)
                val = np.where(valid & (depth1 > 0) & (depth2 > 0), reduction, 0.0)
                k = int(np.argmax(val))
                if val[k] > bestval:
                    bestval = val[k]
                    z1 = depth1[k] / np.sqrt(1.0 / inSw1[k] + 1.0 / Wout[k])
                    z2 = depth2[k] / np.sqrt(1.0 / inSw2[k] + 1.0 / Wout[k])
                    bestrec = dict(period=float(P), power=float(val[k]),
                                   depth1=float(depth1[k]), z1=float(z1),
                                   phase1=float(((k + d / 2.0) % nbins) / nbins),
                                   depth2=float(depth2[k]), z2=float(z2),
                                   phase2=float(((k + shift + d / 2.0) % nbins) / nbins),
                                   width=float(d) / nbins)

        power[pi] = bestval
        if bestrec is not None and bestval > best_val:
            best_val = bestval
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
