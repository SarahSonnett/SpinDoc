# SpinDoc

Tools for determining asteroid rotation periods, light-curve amplitudes, and H-G phase function parameters from calibrated photometry.

---

## Background

### Rotation period and H-G phase function

An asteroid light curve's rotation period and amplitude are fairly intuitive physical properties. The solar phase function is less so, but can also deliver rich information about the surfaces of asteroids.

The solar phase function is the measure of how an asteroid's brightness changes as the observer–asteroid–Sun angle (solar phase angle; α) changes. An asteroid with high porosity, lots of craters, or a very rough surface will get much dimmer as α increases because more shadows appear:

![Phase angle diagram](docs/images/image1.jpg)

*Figure from Buchheim (2010).*

The parameters **H** and **G** describe the y-intercept and steepness of the phase curve, respectively. **H** is the absolute magnitude — the brightness an object would have if it were 1 AU from the Earth, 1 AU from the Sun, and at a phase angle of 0°. This is an unphysical situation, but it is useful for comparing solar system bodies. Since surface reflectivity changes with wavelength, H-magnitude differs for different broadband filters (V-filter H is standard). Lower H means brighter.

The phase function looks something like this:

![Phase function curve](docs/images/image2.jpg)

*Figure from Buchheim (2010).*

![Reduced magnitude vs phase angle](docs/images/image3.png)

Over a small range of phase angles the reduced magnitude may not change much. Most asteroids move only a few degrees in phase angle over a month — much shorter than a typical rotation period — so asteroids with large-amplitude light curves will show rotational brightness variation much larger than their phase-curve effect.

To see the phase curve, we must account for rotational modulation. One further complication: light-curve amplitude can vary slightly with phase angle:

![Amplitude vs phase angle](docs/images/image4.jpg)

*Figure from Buchheim (2010).*

### Algorithm

Fit the rotation period, amplitude, H, and G iteratively by phasing the data at different period, H, and G values and minimizing the reduced chi-squared statistic until convergence.

The rotation solution is sensitive to the H-G parameters (incorrect H-G values can make the data align better at the wrong period), and the H-G parameters are sensitive to the rotation period. For now we ignore the amplitude dependence on phase angle, since the phase angles sampled in typical datasets rarely span more than 10°.

---

## Installation

```bash
git clone https://github.com/ssonnett/SpinDoc.git
cd SpinDoc
pip install numpy matplotlib scipy
```

No package installation step is required — import from `spindoc` works from the repo root.

---

## Package structure

```
SpinDoc/
├── spindoc/
│   ├── __init__.py        # public API
│   ├── hg.py              # IAU H-G phase function
│   ├── fourier.py         # Fourier series models
│   ├── utils.py           # date conversion, chi-squared, directory helpers
│   └── io.py              # photometry file reader
├── period_search.py       # iterative period + H-G fitter
├── period_uncertainty.py  # bootstrap period uncertainty
├── tests/                 # test suite
└── docs/
    ├── images/                                  # figures referenced in this README
    └── Target_Calibrated_FinalErr_rp_cleaned.txt  # sample photometry dataset
```

---

## Usage

### Step 1 — Period and H-G search

Run the code that fits period, H, and G iteratively using a broad period range first:

```bash
python period_search.py \
    --infile  docs/Target_Calibrated_FinalErr_rp_cleaned.txt \
    --object  3923 \
    --minper  2. \
    --maxper  300.
```

**Examine the output.** Look first at the periodograms in the newly created `PeriodHGSearch_<filter>/Periodograms/` directory. The code computes reduced chi-squared periodograms across several Fourier series orders. Look at the 1st iteration to see the broadest range of periods. The minima represent the best period solutions:

![Chi-squared periodogram](docs/images/Chisq_order2_iter2.png)

If reduced chi-squared never falls below ~3, you have poor fits and may not have a good period solution in that plot.

**Examine the summary.** Look at `PeriodHGSearch_<filter>/Summary.*` to identify at which Fourier order the four parameters (period, amplitude, H, G) begin to converge. Select the *lowest* order that achieves convergence — this protects against over-fitting.

![Summary convergence plot](docs/images/image7.png)

**Examine the phased light curve** for the chosen order (3rd iteration). A typical asteroid light curve has two maxima per rotation. If the best-fit solution shows only one maximum, the true period is probably twice the best-fit value:

![Phased light curve](docs/images/Chi2_iter3_fit1.png)

In that case, rerun with a tighter period range centred on the double-period solution:

```bash
python period_search.py \
    --infile  docs/Target_Calibrated_FinalErr_rp_cleaned.txt \
    --object  3923 \
    --minper  24. \
    --maxper  28.
```

Repeat until the period solution is isolated.

**Examine the solar phase function.** Once the rotational modulation has been removed, the code fits the IAU H-G phase function to the reduced magnitudes and writes the result to `PeriodHGSearch_<filter>/`. This is the solar phase function computed by SpinDoc:

![Solar phase function H-G fit](docs/images/HGFit_order2_iter3.png)

### Step 2 — Period and amplitude uncertainties

Use a bootstrapping technique: randomly vary the photometry within its error bars (Gaussian random factor) and refit the light curve for a user-defined number of trials. The FWHMs of the resulting period and amplitude distributions are the uncertainties.

```bash
python period_uncertainty.py \
    --infile   docs/Target_Calibrated_FinalErr_rp_cleaned.txt \
    --objname  3923 \
    --period   26.463 \
    --order    3 \
    --ntrials  1000
```

The code outputs the period and amplitude distributions over all trials, along with the fitted FWHMs that give the uncertainties:

![Period uncertainty results](docs/images/PeriodUncertaintyResults_rp.png)

> **Note:** Uncertainties on H and G should come from the final solar phase function plot, not from this uncertainty algorithm.

---

## Command-line reference

### `period_search.py`

| Argument | Default | Description |
|---|---|---|
| `--infile` | — | Input photometry file |
| `--object` | — | Object name (used in plot titles) |
| `--format` | `None` | Column layout (`None` = default; any other value = compact 7-column format) |
| `--minper` | `2.0` | Minimum search period (hours) |
| `--maxper` | `300.0` | Maximum search period (hours) |
| `--dPstart` | `0.1` | Initial period grid step (hours) |
| `--niterations` | `3` | Number of period–H-G convergence iterations |
| `--sepfilter` | `True` | Analyze each filter separately |
| `--exactrange` | `False` | Keep search range fixed across iterations |
| `--phaseshift` | `0.0` | Additive shift to rotation phase |
| `--writesubtracteddata` | `False` | Write model-subtracted data file |
| `--excludedates` | `None` | Date range to exclude from fitting (e.g. `20100829_20100831` or `56789.123_56789.567`) |

### `period_uncertainty.py`

| Argument | Default | Description |
|---|---|---|
| `--infile` | — | Input photometry file |
| `--objname` | — | Object name |
| `--period` | — | Best-fit period from `period_search.py` (hours) |
| `--order` | — | Fourier order that best fit the data |
| `--ntrials` | `100` | Number of Monte Carlo trials |
| `--sepfilter` | `True` | Analyze each filter separately |
| `--phaseshift` | `0.0` | Additive shift to rotation phase |

---

## Input file format

A sample photometry file is provided at [`docs/Target_Calibrated_FinalErr_rp_cleaned.txt`](docs/Target_Calibrated_FinalErr_rp_cleaned.txt) — the r'-band data used to generate the example figures in this README. It is also exercised as a fixture by the test suite (see [Tests](#tests)).

Default format: whitespace-delimited, 11 columns, with a single header row. The first few lines look like:

```
Frame                           Rhelio  Delta  alpha  Amass Filter Exptime MJD    TmagCorr  TmagErr  TmagFinalErr
lsc1m004-fa03-20190805-0113-e91 5.0334 4.0624 3.731 1.541 rp 154 58701.1115602 18.89380 0.022 0.032
lsc1m004-fa03-20190805-0114-e91 5.0334 4.0624 3.731 1.520 rp 154 58701.1136581 18.85950 0.021 0.031
```

The reader uses fixed column *positions* (header names are ignored), so every column must be present to keep the positions aligned. The columns it reads are:

| Col | Header | Content |
|---|---|---|
| 2 | `Rhelio` | heliocentric distance (AU) |
| 3 | `Delta` | geocentric distance (AU) |
| 4 | `alpha` | solar phase angle (degrees) |
| 6 | `Filter` | filter |
| 8 | `MJD` | observation time (MJD) |
| 9 | `TmagCorr` | calibrated magnitude |
| 11 | `TmagFinalErr` | magnitude uncertainty |

Columns 1, 5, 7, and 10 (`Frame`, `Amass`, `Exptime`, `TmagErr`) are placeholders that must be present but are not read.

Compact format (`--format compact`, 7 data columns): MJD, helio, geo, alpha, mag, merr, filter.

---

## Tests

The test suite reads the sample dataset through the public reader and checks that the default input format is parsed correctly. The tests have no dependencies beyond `numpy`, and can be run either with pytest or directly:

```bash
pytest tests/
# or, without pytest:
python tests/test_read_photometry.py
```

---

## Author

This code package was written by S. Sonnett. It was first referenced publicly in the two Minor Planet Bulletin papers listed below, which were led by students using these tools.

---

## Citing this software

If you use SpinDoc in your work, please cite the papers in which it was first applied:

- Williamson, B., Sonnett, S., Witry, J., Chatelain, J., Grav, T., Reddy, V., Lejoly, C., Kramer, E., Mainzer, A., Masiero, J., Gritsevich, M., & Bauer, J. (2019). "Physical Properties of Hilda Binary Asteroid Candidates." *Minor Planet Bulletin*, 46(3), 332–335. [2019MPBu...46..332W](https://ui.adsabs.harvard.edu/abs/2019MPBu...46..332W/abstract)
- Witry, J., Sonnett, S., Williamson, B., Chatelain, J., Grav, T., Reddy, V., Lejoly, C., Kramer, E., Mainzer, A., Masiero, J., Gritsevich, M., & Bauer, J. (2019). "Rotation Properties of Large-Amplitude Hilda Asteroids." *Minor Planet Bulletin*, 46(3), 335–338. [2019MPBu...46..335W](https://ui.adsabs.harvard.edu/abs/2019MPBu...46..335W/abstract)

---

## References

- Buchheim, R. K. (2010). "Methods and Lessons Learned Determining the H-G Parameters of Asteroid Phase Curves." *Proceedings of the Society for Astronomical Sciences Annual Symposium*, 29, 101–115. [2010SASS...29..101B](https://ui.adsabs.harvard.edu/abs/2010SASS...29..101B/abstract)

The phase-angle, phase-function, and amplitude-vs-phase-angle figures in the **Background** section above are reproduced from Buchheim (2010).
