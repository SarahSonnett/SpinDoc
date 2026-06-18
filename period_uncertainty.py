"""
Bootstrap period uncertainty estimator.

Randomly perturbs photometry within its error bars (Gaussian noise) and refits
the rotation period with the same Fourier order as the original solution.
The FWHM of the resulting period distribution is the period uncertainty.

Usage
-----
python period_uncertainty.py --infile TARGET.txt --objname 3923 \
    --period 26.463 --order 3 --ntrials 1000
"""

import os
import warnings
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

from spindoc import (HGfunction, fourier, chisqrpdf, makenewdir,
                     lighttimecorrection, read_photometry)

warnings.simplefilter(action='ignore', category=FutureWarning)


# ---------------------------------------------------------------------------
# H-G and period fitting helpers
# ---------------------------------------------------------------------------

def fit_hg(alpha, modmags, modelmags, xmodel, merr):
    """Fit H and G after subtracting rotational modulation."""
    modelmean = np.average(xmodel)
    residuals = modmags - (modelmags - modelmean)
    try:
        popt, pcov = curve_fit(
            HGfunction, alpha, residuals,
            sigma=merr, bounds=((-np.inf, -0.25), (np.inf, 0.8))
        )
        H, G = popt
        try:
            Hsigma, Gsigma = np.sqrt(np.diag(pcov))
        except ValueError:
            Hsigma, Gsigma = 10., 10.
        return H, Hsigma, G, Gsigma
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def fourier_chi2(phase, rmags, merr, starterarray):
    """Fit a Fourier model and return reduced chi-squared."""
    idx = np.argsort(phase)
    rp_s, mg_s, me_s = phase[idx], rmags[idx], merr[idx]
    try:
        popt, _ = curve_fit(fourier, rp_s, mg_s, starterarray, sigma=me_s)
        return chisqrpdf(mg_s, fourier(rp_s, *popt), me_s)
    except Exception:
        return np.nan


def fourier_coeffs(phase, rmags, merr, starterarray):
    """Return best-fit Fourier coefficients."""
    idx = np.argsort(phase)
    rp_s, mg_s, me_s = phase[idx], rmags[idx], merr[idx]
    popt, _ = curve_fit(fourier, rp_s, mg_s, starterarray, sigma=me_s)
    return popt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Bootstrap period uncertainty via Monte Carlo photometry perturbation.'
    )
    p.add_argument('--infile',    dest='infile',    help='Input photometry file')
    p.add_argument('--objname',   dest='objname',   help='Object name')
    p.add_argument('--format',    dest='format',    default=None,
                   help='File format (None=default, anything else=compact)')
    p.add_argument('--period',    dest='period',    type=float, help='Best-fit period (hours)')
    p.add_argument('--order',     dest='order',     type=int,   help='Fourier order of best fit')
    p.add_argument('--ntrials',   dest='ntrials',   type=int,   default=100,
                   help='Number of Monte Carlo trials')
    p.add_argument('--sepfilter', dest='sepfilter', default='True',
                   help='Analyze filters separately (True/False)')
    p.add_argument('--phaseshift',dest='phaseshift',type=float, default=0.,
                   help='Additive phase shift applied to rotation phase')
    return p.parse_args()


def main():
    args = parse_args()
    raw = read_photometry(args.infile, fmt=args.format)

    period = args.period
    order  = args.order
    dP     = 0.01
    prange = np.arange(period - 50. * dP, period + 50. * dP + dP, dP)
    starterarray = [1.] * (order * 2 + 2)
    modelx = np.linspace(0., 1., 10000)

    for fltr in np.unique(raw['filters']):
        sel = raw['filters'] == fltr
        time  = raw['time'][sel].copy()
        helio = raw['helio'][sel].copy()
        geo   = raw['geo'][sel].copy()
        alpha = raw['alpha'][sel].copy()
        mags  = raw['mags'][sel].copy()
        merr  = raw['merr'][sel].copy()

        time += lighttimecorrection(geo)
        time  = (time - time[0]) * 24.
        reducedmagsdistance = mags - 5. * np.log10(helio * geo)

        H, G = 15., 0.4
        perarray, amparray, Harray, Garray = [], [], [], []

        sumfile = open(f'PeriodUncertaintyResults_{fltr}.txt', 'w')

        for trial in range(args.ntrials):
            print(f'Running trial {trial+1} of {args.ntrials}')

            modmags = reducedmagsdistance + merr * np.random.normal(0, 1, len(merr))

            # Coarse period search
            chi2array = []
            for p in prange:
                phase = ((time % p) / p + args.phaseshift) % 1.
                chi2array.append(fourier_chi2(phase, modmags, merr, starterarray))
            topper = prange[np.argsort(chi2array)]

            # Refit H-G at coarse period
            phase = ((time % topper[0]) / topper[0] + args.phaseshift) % 1.
            popt = fourier_coeffs(phase, modmags, merr, starterarray)
            modelmags = fourier(phase[np.argsort(phase)], *popt)
            fullmodelmags = fourier(modelx, *popt)
            H, Hsigma, G, Gsigma = fit_hg(alpha, modmags, modelmags, fullmodelmags, merr)
            rmags = modmags - HGfunction(alpha, H, G) + H

            # Refined period search
            prange2 = np.arange(topper[0] - 20. * (dP/10.),
                                 topper[0] + 20. * (dP/10.) + (dP/10.), dP/10.)
            chi2array = []
            for p in prange2:
                phase = ((time % p) / p + args.phaseshift) % 1.
                chi2array.append(fourier_chi2(phase, rmags, merr, starterarray))
            topper = prange2[np.argsort(chi2array)]

            # Final H-G refit
            phase = ((time % topper[0]) / topper[0] + args.phaseshift) % 1.
            try:
                popt = fourier_coeffs(phase, modmags, merr, starterarray)
                modelmags = fourier(phase[np.argsort(phase)], *popt)
                fullmodelmags = fourier(modelx, *popt)
                H, Hsigma, G, Gsigma = fit_hg(alpha, modmags, modelmags, fullmodelmags, merr)
                rmags = modmags - HGfunction(alpha, H, G) + H

                phase = ((time % topper[0]) / topper[0] + args.phaseshift) % 1.
                popt = fourier_coeffs(phase, rmags, merr, starterarray)
                modelmags_full = fourier(modelx, *popt)
                amp = max(modelmags_full) - min(modelmags_full)

                perarray.append(topper[0])
                amparray.append(amp)
                Harray.append(H)
                Garray.append(G)
                sumfile.write(f'{topper[0]}  {amp}  {H}  {G}\n')
                sumfile.flush()
            except Exception:
                pass

        sumfile.close()

        # Summary histograms
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        datasets = [
            (perarray, ax1, 'Period (hrs)', 'b'),
            (amparray, ax2, 'Amplitude (mags)', 'g'),
            (Harray,   ax3, 'H (mags)', 'r'),
            (Garray,   ax4, 'G', 'y'),
        ]
        for arr, ax, xlabel, color in datasets:
            if not arr:
                continue
            mu, sigma = norm.fit(arr)
            print(f'{xlabel}: {mu:.4f} +/- {sigma:.4f}')
            n, bins, _ = ax.hist(arr, 20, density=True, histtype='stepfilled',
                                 color=color, alpha=0.7)
            ax.plot(bins, norm.pdf(bins, mu, sigma), 'k--', linewidth=2.)
            ax.text(min(bins), 0.9 * max(n),
                    f'{mu:.4f} ' + r'$\pm$' + f' {sigma:.4f}')
            ax.set_xlabel(xlabel)

        fig.suptitle(f'{args.objname}, {len(perarray)} trials')
        plt.tight_layout()
        plt.savefig(f'PeriodUncertaintyResults_{fltr}.png')
        plt.close()


if __name__ == '__main__':
    main()
