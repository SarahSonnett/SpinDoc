"""
Iterative period and H-G phase function fitter using Fourier series.

Fits asteroid rotation period, amplitude, H (absolute magnitude), and G (phase slope
parameter) simultaneously by minimizing reduced chi-squared over a grid of trial periods,
then alternating period and H-G fits until convergence.

Usage
-----
python period_search.py --infile TARGET.txt --object 3923 [options]

See --help for full argument list.
"""

import os
import warnings
import argparse
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from spindoc import (HGfunction, fourier, hg_fourier, chisqrpdf, makenewdir,
                     converttoUT, converttoMJD, lighttimecorrection, read_photometry,
                     bls_search, find_events)

warnings.simplefilter(action='ignore', category=FutureWarning)

# Marker cycle long enough for any realistic number of nights
_MARKERS = (['o', 's', '^', '*', 'x', '+'] * 12)


# ---------------------------------------------------------------------------
# Per-filter data container
# ---------------------------------------------------------------------------

class FilterData:
    """Holds all arrays and metadata for a single filter's observations."""

    def __init__(self, time, mjd, helio, geo, alpha, mags, merr,
                 reducedmagsdistance, mjdlowerlim, mjdupperlim,
                 fltr, objname, phaseshift, excludedates):
        self.time = time
        self.mjd = mjd
        self.helio = helio
        self.geo = geo
        self.alpha = alpha
        self.mags = mags
        self.merr = merr
        self.reducedmagsdistance = reducedmagsdistance
        self.mjdlowerlim = mjdlowerlim
        self.mjdupperlim = mjdupperlim
        self.fltr = fltr
        self.objname = objname
        self.phaseshift = phaseshift
        self.excludedates = excludedates

    def fit_mask(self):
        """Boolean mask: True for points outside the excluded date range."""
        if self.mjdlowerlim > 0:
            return np.logical_or(self.mjd < self.mjdlowerlim, self.mjd > self.mjdupperlim)
        return np.ones(len(self.mjd), dtype=bool)


# ---------------------------------------------------------------------------
# Core fitting routines
# ---------------------------------------------------------------------------

def fit_hg(model, xmodel, period, order, iteration, data):
    """Subtract rotational modulation and fit the H-G phase function.

    Returns (H, Hsigma, G, Gsigma); all np.nan on failure.
    """
    modelmean = np.average(xmodel)
    modeldiffs = model - modelmean

    mask = data.fit_mask()
    rmags_fit = data.reducedmagsdistance[mask]
    merr_fit = data.merr[mask]
    alpha_fit = data.alpha[mask]
    residuals = rmags_fit - modeldiffs

    try:
        popt, pcov = curve_fit(
            HGfunction, alpha_fit, residuals,
            sigma=merr_fit, bounds=((-np.inf, -0.25), (np.inf, 0.8))
        )
        H, G = popt
        try:
            Hsigma, Gsigma = np.sqrt(np.diag(pcov))
        except ValueError:
            Hsigma, Gsigma = 10., 10.

        alphaarray = np.linspace(0, max(alpha_fit) + 1, 1000)
        makenewdir(f'PeriodHGSearch_{data.fltr}/PhaseFunctions')
        plt.errorbar(alpha_fit, residuals, yerr=merr_fit, color='b', fmt='o', ms=6,
                     markeredgecolor='k', markeredgewidth=0.5, label='Model-subtracted data')
        plt.xlabel('Solar phase angle (degrees)')
        plt.ylabel('Reduced mean magnitude')
        plt.title(
            f"{data.objname}, H = {H:.2f}±{Hsigma:.2f} mags, "
            f"G = {G:.2f}±{Gsigma:.2f}, T = {round(period, iteration+1)} hrs",
            fontsize=12
        )
        plt.gca().invert_yaxis()
        plt.plot(alphaarray, HGfunction(alphaarray, *popt), 'k-', label='H-G fitted function')
        plt.legend(loc='lower left')
        plt.savefig(f'PeriodHGSearch_{data.fltr}/PhaseFunctions/HGFit_order{order}_iter{iteration}.png')
        plt.close()

        return H, Hsigma, G, Gsigma

    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def fit_period_chisq(time, minper, maxper, dP, H, G, order, iteration, data,
                     writesubtracteddata=False, infile=None):
    """Search a period grid and fit Fourier models; return best-fit diagnostics.

    Returns (finalper, finalamp, finalmodel, finalxmodel, finalchi2).
    All np.nan on failure.
    """
    reducedmags = data.reducedmagsdistance - HGfunction(data.alpha, H, G) + H

    datebins = list(np.unique(np.floor(data.mjd + 0.5)))
    datebins.append(max(datebins) + 1)
    starterarray = [1.] * (order * 2 + 1)
    modelx = np.linspace(0., 1., 10000)
    mask = data.fit_mask()

    # --- Period grid chi-squared ---
    chi2array = []
    for period in np.arange(minper, maxper + dP, dP):
        rphase = (time % period) / period
        rphase = (rphase + data.phaseshift) % 1.
        idx = np.argsort(rphase)
        rp_s = rphase[idx]
        mg_s = reducedmags[idx]
        me_s = data.merr[idx]
        mj_s = data.mjd[idx]
        if data.mjdlowerlim > 0:
            fm = np.logical_or(mj_s < data.mjdlowerlim, mj_s > data.mjdupperlim)
            rp_s, mg_s, me_s = rp_s[fm], mg_s[fm], me_s[fm]
        try:
            popt, _ = curve_fit(fourier, rp_s, mg_s, starterarray, sigma=me_s)
            chi2array.append(chisqrpdf(mg_s, fourier(rp_s, *popt), me_s, nparams=len(popt)))
        except Exception:
            chi2array.append(np.nan)

    courseperiods = np.arange(minper, maxper + dP, dP)
    topper = courseperiods[np.asarray(chi2array).argsort()]

    plt.plot(courseperiods, chi2array)
    plt.xlabel('Period (hrs)', fontsize=12)
    plt.ylabel(r'$\chi^{2}_{\nu}$', fontsize=12)
    plt.title(f'{data.objname} Periodogram')
    plt.savefig(f'PeriodHGSearch_{data.fltr}/Periodograms/Chisq_order{order}_iter{iteration}.png')
    plt.close()

    finalper = finalamp = finalmodel = finalxmodel = finalchi2 = np.nan

    for j in range(5):
        bestper = topper[j]
        _plot_phased(time, bestper, reducedmags, datebins, modelx, starterarray,
                     order, iteration, j, data, mask,
                     writesubtracteddata=writesubtracteddata, infile=infile,
                     store=(j == 0))
        if j == 0:
            rotphase = (time % bestper) / bestper
            rotphase = (rotphase + data.phaseshift) % 1.
            idx = np.argsort(rotphase)
            rp_s = rotphase[idx]
            mg_s = reducedmags[idx]
            me_s = data.merr[idx]
            mj_s = data.mjd[idx]
            if data.mjdlowerlim > 0:
                fm = np.logical_or(mj_s < data.mjdlowerlim, mj_s > data.mjdupperlim)
                rp_s_fit = rp_s[fm]; mg_s_fit = mg_s[fm]; me_s_fit = me_s[fm]
                rotphase_fit = rotphase[mask]
            else:
                rp_s_fit = rp_s; mg_s_fit = mg_s; me_s_fit = me_s
                rotphase_fit = rotphase
            try:
                popt, _ = curve_fit(fourier, rp_s_fit, mg_s_fit, starterarray, sigma=me_s_fit)
                finalamp = max(fourier(modelx, *popt)) - min(fourier(modelx, *popt))
                finalper = bestper
                finalmodel = fourier(rotphase_fit, *popt)
                finalxmodel = fourier(modelx, *popt)
                finalchi2 = chisqrpdf(mg_s_fit, fourier(rp_s_fit, *popt), me_s_fit, nparams=len(popt))
                if writesubtracteddata and infile:
                    _write_subtracted(infile, order, reducedmags - fourier(rotphase, *popt))
            except Exception:
                pass

    return finalper, finalamp, finalmodel, finalxmodel, finalchi2


def _plot_phased(time, bestper, reducedmags, datebins, modelx, starterarray,
                 order, iteration, j, data, mask, writesubtracteddata=False,
                 infile=None, store=False):
    """Plot phased light curve at bestper and 2*bestper."""
    for multiplier, suffix in [(1, ''), (2, '2xPer_')]:
        per = bestper * multiplier
        rotphase = (time % per) / per
        rotphase = (rotphase + data.phaseshift) % 1.
        idx = np.argsort(rotphase)
        rp_s = rotphase[idx]
        mg_s = reducedmags[idx]
        me_s = data.merr[idx]
        mj_s = data.mjd[idx]
        if data.mjdlowerlim > 0:
            fm = np.logical_or(mj_s < data.mjdlowerlim, mj_s > data.mjdupperlim)
            rp_fit, mg_fit, me_fit = rp_s[fm], mg_s[fm], me_s[fm]
        else:
            rp_fit, mg_fit, me_fit = rp_s, mg_s, me_s

        try:
            popt, _ = curve_fit(fourier, rp_fit, mg_fit, starterarray, sigma=me_fit)
            modelmags = fourier(rp_fit, *popt)
            amp = max(fourier(modelx, *popt)) - min(fourier(modelx, *popt))
            chi2final = chisqrpdf(mg_fit, modelmags, me_fit, nparams=len(popt))
        except Exception:
            continue

        zord = 1
        for i in range(len(datebins) - 1):
            night = np.logical_and(data.mjd + 0.5 >= datebins[i], data.mjd + 0.5 < datebins[i + 1])
            lbl = converttoUT(datebins[i])
            lblstr = f"{int(lbl[0]-2000)}/{int(lbl[1])}/{int(lbl[2])}"
            plt.errorbar(rotphase[night], reducedmags[night], yerr=data.merr[night],
                         zorder=zord, fmt=_MARKERS[(zord-1) % len(_MARKERS)], ms=6,
                         label=lblstr, markeredgewidth=0.5, markeredgecolor='k')
            zord += 1

        chi2_label = r'$\chi^{2}_{\nu}$' + f' = {chi2final:.3f}'
        if data.mjdlowerlim > 0:
            excl = data.excludedates
            chi2_label += f', no {excl.split("_")[0]} - {excl.split("_")[1]}'
        plt.text(0., max(reducedmags + data.merr), chi2_label)
        plt.xlim((-0.1, 1.1))
        plt.gca().invert_yaxis()
        plt.plot(modelx, fourier(modelx, *popt), 'k-', label=f'Model order: {order}', lw=2, zorder=zord)
        lcol = int(np.ceil(len(datebins) / 21.))
        plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1., 1.),
                   title='UT Date: 20YY/M/D', ncol=lcol)
        plt.xlabel('Rotation phase')
        plt.ylabel('Reduced magnitude')
        try:
            plt.title(f'{data.objname}, Per. = {per:.4f} h, Amp. = {amp:.3f} mags, N = {len(reducedmags)}')
        except Exception:
            plt.title(f'{data.objname}, Per. = {per:.4f} h, Unable to solve model, N = {len(reducedmags)}')
        makenewdir(f'PeriodHGSearch_{data.fltr}/PhasedLightcurves/Order{order}')
        plt.savefig(
            f'PeriodHGSearch_{data.fltr}/PhasedLightcurves/Order{order}/'
            f'{suffix}Chi2_iter{iteration}_fit{j+1}.png',
            bbox_inches='tight'
        )
        plt.close()


def _write_subtracted(infile, order, datasubmodel):
    outname = infile.split('.')[0] + f'_subtracted_ord{order}.txt'
    lines = open(infile).readlines()
    with open(outname, 'w') as f:
        for i, line in enumerate(lines[1:]):
            parts = line.split(' ')
            parts[8] = str(round(datasubmodel[i], 3))
            f.write(' '.join(parts[:11]) + '\n')


# ---------------------------------------------------------------------------
# Joint H-G + Fourier refinement
# ---------------------------------------------------------------------------

def fit_joint(data, period, order):
    """Fit the H-G phase function and Fourier rotation series simultaneously.

    Performs a single least-squares fit of {H, G, A_1, phi_1, ...} to the
    reduced magnitudes at the (fixed) converged rotation period, so all
    parameters share one covariance matrix. This yields self-consistent,
    correlated uncertainties and a reduced chi-squared computed with the correct
    degrees of freedom (dof = N - k). The rotation period itself stays fixed
    from the grid search because the chi-squared surface over period is strongly
    multimodal and a local optimiser cannot search it globally.

    Returns a dict of results, or None on failure.
    """
    mask = data.fit_mask()
    alpha = data.alpha[mask]
    merr = data.merr[mask]
    rmag = data.reducedmagsdistance[mask]
    phase = ((data.time[mask] % period) / period + data.phaseshift) % 1.

    X = np.vstack([alpha, phase])
    p0 = [15., 0.4] + [1.] * (2 * order)          # H, G, then A_n, phi_n
    lower = [-np.inf, -0.25] + [-np.inf] * (2 * order)
    upper = [np.inf, 0.80] + [np.inf] * (2 * order)
    try:
        popt, pcov = curve_fit(hg_fourier, X, rmag, p0=p0, sigma=merr,
                               bounds=(lower, upper), max_nfev=10000)
    except Exception:
        return None

    perr = np.sqrt(np.diag(pcov))

    # Peak-to-peak amplitude of the rotational part only (phase-function removed).
    grid = np.linspace(0., 1., 10000)
    rot = np.zeros_like(grid)
    fcoeff = popt[2:]
    for n in range(1, order + 1):
        rot += fcoeff[2 * n - 2] * np.sin(2. * np.pi * n * grid + fcoeff[2 * n - 1])
    amp = rot.max() - rot.min()

    model = hg_fourier(X, *popt)
    chi2nu = chisqrpdf(rmag, model, merr, nparams=len(popt))

    return dict(period=period, order=order, H=popt[0], Hsig=perr[0],
                G=popt[1], Gsig=perr[1], amp=amp, coeff=popt, perr=perr,
                cov=pcov, chi2nu=chi2nu, npts=len(rmag))


def write_model(fltr, res):
    """Write a joint best-fit model to a self-describing text file.

    Records everything needed to reconstruct the model downstream (e.g. for
    shape-model fitting): the rotation period, H, G, the Fourier coefficients
    with 1-sigma errors, the full covariance matrix, and a dense sampled model
    curve.
    """
    order = res['order']
    popt, perr, cov = res['coeff'], res['perr'], res['cov']
    makenewdir(f'PeriodHGSearch_{fltr}/Models')
    path = f'PeriodHGSearch_{fltr}/Models/Model_order{order}.txt'

    names = ['H', 'G']
    for n in range(1, order + 1):
        names += [f'A_{n}', f'phi_{n}']

    with open(path, 'w') as f:
        f.write('# SpinDoc best-fit joint H-G + Fourier model\n')
        f.write(f'# filter                    = {fltr}\n')
        f.write(f'# rotation_period_hours      = {res["period"]:.6f}\n')
        f.write(f'# fourier_order              = {order}\n')
        f.write(f'# n_points                   = {res["npts"]}\n')
        f.write(f'# n_free_parameters (k)      = {len(popt)}\n')
        f.write(f'# reduced_chi2 (dof = N - k) = {res["chi2nu"]:.4f}\n')
        f.write(f'# H_mag                      = {res["H"]:.4f} +/- {res["Hsig"]:.4f}\n')
        f.write(f'# G                          = {res["G"]:.4f} +/- {res["Gsig"]:.4f}\n')
        f.write(f'# peak_to_peak_amplitude_mag = {res["amp"]:.4f}\n')
        f.write('#\n')
        f.write('# Model: reduced_mag(alpha, phase) = HGfunction(alpha; H, G)\n')
        f.write('#          + sum_n A_n * sin(2*pi*n*phase + phi_n),   phase in [0, 1)\n')
        f.write('#\n')
        f.write('# Fitted parameters:\n')
        f.write('# name     value            sigma\n')
        for nm, v, s in zip(names, popt, perr):
            f.write(f'{nm:9s} {v: .8e} {s: .8e}\n')
        f.write('#\n')
        f.write('# Covariance matrix (parameter order as above):\n')
        for row in cov:
            f.write('# ' + ' '.join(f'{x: .6e}' for x in row) + '\n')
        f.write('#\n')
        f.write('# Dense model light curve at alpha = 0 (H-level):\n')
        f.write('# phase      model_mag\n')
        grid = np.linspace(0., 1., 1000)
        curve = hg_fourier(np.vstack([np.zeros_like(grid), grid]), *popt)
        for ph, mm in zip(grid, curve):
            f.write(f'{ph:.5f}  {mm:.6f}\n')

    return path


def recommend_order(results):
    """Suggest a Fourier order by minimising the Bayesian Information Criterion.

    BIC = chi2 + k*ln(N), where chi2 is the total (not reduced) chi-squared, k is
    the number of free parameters, and N is the number of points. BIC penalises
    extra harmonics more strongly than AIC (ln N > 2 for realistic N), which
    guards against over-fitting. This is only a recommendation -- the final order
    remains a human decision.

    Returns (recommended_order, {order: bic}); (None, {}) if there are no results.
    """
    bic = {}
    for r in results:
        k = len(r['coeff'])
        dof = max(r['npts'] - k, 1)
        chi2_total = r['chi2nu'] * dof
        bic[r['order']] = chi2_total + k * np.log(r['npts'])
    if not bic:
        return None, {}
    return min(bic, key=bic.get), bic


# ---------------------------------------------------------------------------
# Mutual-event (binary eclipse) search
# ---------------------------------------------------------------------------

def run_mutual_events(data, res, args):
    """Search the model-subtracted residuals for binary mutual events.

    Recomputes residuals from the joint best-fit model of the chosen order, then
    runs a box-least-squares scan over trial orbital periods (recurring dip) and
    an individual-event flagger (localised faint excursions). Writes a summary
    and diagnostic plots under ``MutualEvents/``. A candidate orbital period near
    the rotation period (or a simple multiple) is flagged as a likely
    rotation-model residual rather than a genuine binary signal.
    """
    mask = data.fit_mask()
    alpha = data.alpha[mask]
    merr = data.merr[mask]
    rmag = data.reducedmagsdistance[mask]
    t = data.time[mask]
    phase = ((t % res['period']) / res['period'] + data.phaseshift) % 1.
    resid = rmag - hg_fourier(np.vstack([alpha, phase]), *res['coeff'])

    rot = res['period']
    baseline = t.max() - t.min()
    minorb = args.minorbital if args.minorbital is not None else 2. * rot
    maxorb = args.maxorbital if args.maxorbital is not None else baseline / 2.
    if baseline <= 0 or maxorb <= minorb:
        print(f'[filter {data.fltr}] mutual-event search skipped: insufficient time baseline.')
        return

    periods = np.linspace(minorb, maxorb, max(args.norbital, 2))
    bls = bls_search(t, resid, merr, periods)
    events = find_events(t, resid, merr, nsigma=args.eventsigma)
    best = bls['best']

    outdir = f'PeriodHGSearch_{data.fltr}/MutualEvents'
    makenewdir(outdir)

    with open(f'{outdir}/MutualEvents_summary.txt', 'w') as f:
        f.write('# SpinDoc mutual-event (binary eclipse) search\n')
        f.write(f'# residuals from joint model, Fourier order {res["order"]}\n')
        f.write(f'# rotation_period_hours = {rot:.5f}\n')
        f.write(f'# orbital search range  = {minorb:.3f} .. {maxorb:.3f} h ({len(periods)} steps)\n')
        f.write('#\n# --- Periodic-dip (BLS) search ---\n')
        if best is not None:
            f.write(f'best_orbital_period_hours = {best["period"]:.5f}\n')
            f.write(f'event_depth_mag           = {best["depth"]:.4f}\n')
            f.write(f'detection_zscore          = {best["zscore"]:.2f}\n')
            f.write(f'event_phase               = {best["phase"]:.3f}\n')
            f.write(f'event_width_fraction      = {best["width"]:.3f}\n')
            ratio = best['period'] / rot
            for r_ in (0.5, 1.0, 2.0):
                if abs(ratio - r_) < 0.03:
                    f.write(f'# WARNING: candidate is ~{r_:g}x the rotation period -- '
                            f'likely a rotation-model residual, not a binary signal.\n')
            span = maxorb - minorb
            if best['period'] - minorb < 0.02 * span or maxorb - best['period'] < 0.02 * span:
                f.write('# WARNING: candidate sits at the edge of the search range -- often a\n')
                f.write('#          long-timescale systematic; adjust --minorbital/--maxorbital.\n')
            if best['width'] > 0.24:
                f.write('# WARNING: fitted event width is near the maximum -- a broad trend rather\n')
                f.write('#          than a localised eclipse; inspect the folded residuals.\n')
            f.write('# NOTE: the z-score is a single-trial statistic and assumes independent points;\n')
            f.write('#       assess a false-alarm probability (and correlated nightly noise) before\n')
            f.write('#       claiming a detection.\n')
        else:
            f.write('no periodic candidate found\n')
        f.write('#\n# --- Individual events (> %.1f sigma faint excursions) ---\n' % args.eventsigma)
        f.write('# t_start_h  t_end_h  n_points  max_zscore  mean_depth_mag\n')
        for e in events:
            f.write(f'{e["t_start"]:.4f}  {e["t_end"]:.4f}  {e["n_points"]:3d}  '
                    f'{e["max_zscore"]:.2f}  {e["mean_depth"]:.4f}\n')
        if not events:
            f.write('# (none)\n')

    # BLS periodogram
    plt.figure()
    plt.plot(bls['periods'], bls['power'], lw=1)
    if best is not None:
        plt.axvline(best['period'], color='r', ls='--', lw=1,
                    label=f'best = {best["period"]:.3f} h (z={best["zscore"]:.1f})')
        plt.legend()
    plt.xlabel('Trial orbital period (hrs)')
    plt.ylabel('BLS detection z-score')
    plt.title(f'{data.objname} mutual-event search (order {res["order"]})')
    plt.savefig(f'{outdir}/BLS_periodogram.png', bbox_inches='tight')
    plt.close()

    # residuals vs time with flagged events shaded
    plt.figure(figsize=(9, 4))
    plt.errorbar(t, resid, yerr=merr, fmt='o', ms=3, color='0.5', lw=0.5)
    for e in events:
        plt.axvspan(e['t_start'], e['t_end'], color='orange', alpha=0.3)
    plt.axhline(0, color='k', lw=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('Elapsed time (hrs)')
    plt.ylabel('Residual (mag; fainter downward)')
    plt.title(f'{data.objname} model-subtracted residuals')
    plt.savefig(f'{outdir}/Residuals_vs_time.png', bbox_inches='tight')
    plt.close()

    # residuals folded at the candidate orbital period
    if best is not None:
        ph = (t % best['period']) / best['period']
        plt.figure()
        plt.errorbar(ph, resid, yerr=merr, fmt='o', ms=3, color='0.5', lw=0.5)
        plt.axhline(0, color='k', lw=0.5)
        plt.gca().invert_yaxis()
        plt.xlabel(f'Orbital phase (P = {best["period"]:.4f} h)')
        plt.ylabel('Residual (mag; fainter downward)')
        plt.title(f'{data.objname} residuals folded at candidate orbital period')
        plt.savefig(f'{outdir}/Residuals_folded.png', bbox_inches='tight')
        plt.close()

    if best is not None:
        print(f'[filter {data.fltr}] mutual-event search (order {res["order"]}): '
              f'best candidate P_orb = {best["period"]:.3f} h, depth {best["depth"]:.3f} mag, '
              f'z = {best["zscore"]:.1f}; {len(events)} individual event(s) flagged.')
    else:
        print(f'[filter {data.fltr}] mutual-event search: no periodic candidate; '
              f'{len(events)} individual event(s) flagged.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Fit asteroid rotation period, H, and G iteratively.')
    p.add_argument('--infile',            dest='infile',            help='Input photometry file')
    p.add_argument('--object',            dest='objname',           help='Object name')
    p.add_argument('--format',            dest='format',            default=None,
                   help='File format: None=default columns, anything else=compact columns')
    p.add_argument('--minper',            dest='minper',            type=float, default=2.,
                   help='Minimum search period (hours)')
    p.add_argument('--maxper',            dest='maxper',            type=float, default=300.,
                   help='Maximum search period (hours)')
    p.add_argument('--dPstart',           dest='dPstart',           type=float, default=0.1,
                   help='Initial period grid step (hours)')
    p.add_argument('--niterations',       dest='niter',             type=int,   default=3,
                   help='Number of period-HG convergence iterations')
    p.add_argument('--sepfilter',         dest='sepfilter',         default='True',
                   help='Analyze filters separately (True/False)')
    p.add_argument('--exactrange',        dest='exactrange',        default='False',
                   help='Keep period search range fixed across iterations (True/False)')
    p.add_argument('--phaseshift',        dest='phaseshift',        type=float, default=0.,
                   help='Additive phase shift applied to rotation phase')
    p.add_argument('--writesubtracteddata', dest='writesubtracteddata', default='False',
                   help='Write model-subtracted data to file (True/False)')
    p.add_argument('--excludedates',      dest='excludedates',      default=None,
                   help='Date range to exclude from fitting, e.g. 20100829_20100831')
    p.add_argument('--mutualevents',      dest='mutualevents',      default='True',
                   help='Search residuals for binary mutual events (True/False)')
    p.add_argument('--eventorder',        dest='eventorder',        type=int, default=None,
                   help='Fourier order whose residuals to search (default: BIC-recommended)')
    p.add_argument('--minorbital',        dest='minorbital',        type=float, default=None,
                   help='Minimum trial orbital period, hours (default: 2x rotation period)')
    p.add_argument('--maxorbital',        dest='maxorbital',        type=float, default=None,
                   help='Maximum trial orbital period, hours (default: half the data baseline)')
    p.add_argument('--norbital',          dest='norbital',          type=int, default=2000,
                   help='Number of trial orbital periods in the mutual-event search')
    p.add_argument('--eventsigma',        dest='eventsigma',        type=float, default=4.0,
                   help='Per-point significance threshold for individual event flagging')
    return p.parse_args()


def build_filter_data(raw, fltr, objname, phaseshift, excludedates_str):
    """Extract and pre-process data for one filter."""
    sel = raw['filters'] == fltr
    time  = raw['time'][sel].copy()
    helio = raw['helio'][sel].copy()
    geo   = raw['geo'][sel].copy()
    alpha = raw['alpha'][sel].copy()
    mags  = raw['mags'][sel].copy()
    merr  = raw['merr'][sel].copy()

    time += lighttimecorrection(geo)
    mjd = time.copy()

    if excludedates_str and excludedates_str != 'None':
        if excludedates_str[0] == '5':
            mjdlowerlim = float(excludedates_str.split('_')[0])
            mjdupperlim = float(excludedates_str.split('_')[1])
        else:
            mjdlowerlim = converttoMJD([excludedates_str[0:4], excludedates_str[4:6], excludedates_str[6:8]])
            mjdupperlim = converttoMJD([excludedates_str[9:13], excludedates_str[13:15], excludedates_str[15:17]])
    else:
        mjdlowerlim = mjdupperlim = -1.

    time = (time - time[0]) * 24.
    reducedmags = mags - 5. * np.log10(helio * geo)

    return FilterData(
        time=time, mjd=mjd, helio=helio, geo=geo, alpha=alpha, mags=mags, merr=merr,
        reducedmagsdistance=reducedmags, mjdlowerlim=mjdlowerlim, mjdupperlim=mjdupperlim,
        fltr=fltr, objname=objname, phaseshift=phaseshift,
        excludedates=excludedates_str if excludedates_str else 'None'
    )


def run_filter(data, args):
    """Run the full period + H-G search for one filter."""
    minper_orig = args.minper
    maxper_orig = args.maxper
    exactrange = args.exactrange == 'True'

    makenewdir(f'PeriodHGSearch_{data.fltr}/')
    makenewdir(f'PeriodHGSearch_{data.fltr}/PhasedLightcurves/')

    modelx = np.linspace(0., 1., 10000)
    orderarray = range(2, 10)

    sumfile = open(f'PeriodHGSearch_{data.fltr}/Summary.txt', 'w')
    sumfile.write('Order Iteration chi2_nu Per(hrs) Amp(mags) H(mags) Hsigma G    Gsigma\n')

    # Joint-fit summary: statistically self-consistent H, G, amplitude and an
    # honest reduced chi-squared (dof = N - k) from one simultaneous fit per order.
    jointfile = open(f'PeriodHGSearch_{data.fltr}/Summary_joint.txt', 'w')
    jointfile.write('Order chi2_nu(N-k) Per(hrs) Amp(mags) H(mags) Hsigma G    Gsigma\n')

    all_rows = []
    joint_results = []

    for order in orderarray:
        minper, maxper, dP = minper_orig, maxper_orig, args.dPstart
        H, G = 15., 0.4

        for j in range(args.niter):
            makenewdir(f'PeriodHGSearch_{data.fltr}/Periodograms')

            if j == 0:
                topper, topamp, model, xmodel, chi2final = fit_period_chisq(
                    data.time, minper, maxper, dP, H, G, order, j + 1, data,
                    writesubtracteddata=(args.writesubtracteddata == 'True'),
                    infile=args.infile
                )

            H, Hsigma, G, Gsigma = fit_hg(model, xmodel, topper, order, j + 1, data)
            topper, topamp, model, xmodel, chi2final = fit_period_chisq(
                data.time, minper, maxper, dP, H, G, order, j + 1, data,
                writesubtracteddata=(args.writesubtracteddata == 'True'),
                infile=args.infile
            )

            row = (order, j+1, chi2final, topper, topamp, H,
                   Hsigma if not np.isnan(Hsigma) else 0.,
                   G, Gsigma if not np.isnan(Gsigma) else 0.)
            all_rows.append(row)
            sumfile.write('%2i  %2i  %2.3f  %3.4f  %1.3f  %2.4f  %1.4f  %2.4f  %1.4f\n' % row)
            sumfile.flush()

            if exactrange:
                minper, maxper = minper_orig, maxper_orig
            else:
                minper = topper - 20. * dP
                maxper = topper + 20. * dP
            dP /= 10.

        # Simultaneous H-G + Fourier refinement at the converged period, and
        # write the reproducible best-fit model for this order.
        res = fit_joint(data, topper, order)
        if res is not None:
            write_model(data.fltr, res)
            joint_results.append(res)
            jointfile.write('%2i  %8.4f  %3.4f  %1.3f  %2.4f  %1.4f  %2.4f  %1.4f\n'
                            % (order, res['chi2nu'], res['period'], res['amp'],
                               res['H'], res['Hsig'], res['G'], res['Gsig']))
            jointfile.flush()

    # Recommend an order (BIC) -- a suggestion only; the choice stays with you.
    rec, bic = recommend_order(joint_results)
    if rec is not None:
        msg = (f'Recommended Fourier order (lowest BIC): {rec}. '
               f'This is a suggestion only -- please confirm by inspecting the '
               f'phased light curves and parameter convergence, and choose the '
               f'lowest order that fits well.')
        print(f'[filter {data.fltr}] {msg}')
        jointfile.write('#\n# ' + msg + '\n')
        jointfile.write('# BIC by order: '
                        + ', '.join(f'{o}:{bic[o]:.1f}' for o in sorted(bic)) + '\n')

    sumfile.close()
    jointfile.close()

    # Search the chosen order's residuals for binary mutual events.
    if args.mutualevents == 'True' and joint_results:
        want = args.eventorder if args.eventorder is not None else rec
        res_ev = next((r for r in joint_results if r['order'] == want), joint_results[-1])
        run_mutual_events(data, res_ev, args)

    # Summary convergence plot
    try:
        order_a, iter_a, chi2_a, per_a, amp_a, H_a, Hsig_a, G_a, Gsig_a = np.loadtxt(
            f'PeriodHGSearch_{data.fltr}/Summary.txt', unpack=True, dtype=float, skiprows=1
        )
        last = iter_a == max(iter_a)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        ax1.plot(order_a[last], per_a[last], 'k')
        ax2.plot(order_a[last], amp_a[last], 'b')
        ax3.errorbar(order_a[last], H_a[last], yerr=Hsig_a[last], fmt='o', color='r')
        ax4.errorbar(order_a[last], G_a[last], yerr=Gsig_a[last], fmt='o', color='g')
        for ax, lbl in zip([ax1, ax2, ax3, ax4],
                           ['Period (hrs)', 'Amplitude (mags)', 'H (mags)', 'G']):
            ax.set_ylabel(lbl)
            ax.set_xlabel('Order of fit')
            ax.set_xlim((min(order_a[last]) - 1, max(order_a[last]) + 1))
        plt.tight_layout()
        plt.savefig(f'PeriodHGSearch_{data.fltr}/Summary.png')
        plt.close()
    except Exception:
        pass


def main():
    args = parse_args()
    raw = read_photometry(args.infile, fmt=args.format)

    for fltr in np.unique(raw['filters']):
        data = build_filter_data(raw, fltr, args.objname,
                                 args.phaseshift, args.excludedates)
        run_filter(data, args)


if __name__ == '__main__':
    main()
