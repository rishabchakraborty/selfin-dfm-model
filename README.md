# Selfin tail engine

DFM + jump-diffusion + EVT for a book of event contracts.

    pip install numpy pandas scipy statsmodels
    python dfm.py        # estimate the factor model, writes Fhat.pkl
    python engine.py     # 3 models x 200k paths, EVT tail
    python analysis.py   # capital allocation, pricing gaps, hedgeable share

| file | what it does |
|---|---|
| `panel.py`    | the 14-indicator observable panel (swap in real series here) |
| `dfm.py`      | Kalman/MLE dynamic factor model + varimax rotation |
| `book.py`     | the contract book and its factor loadings |
| `engine.py`   | jump-diffusion simulation, hazard model, GPD tail fit |
| `analysis.py` | Euler ES allocation, required loadings, threshold stability |

## To run it on a real book

1. `panel.py` — replace `simulate_panel()` with a loader for the real indicator panel.
   Everything downstream is agnostic to where the data came from.
2. `book.py` — replace `BOOK` with live positions. Loadings `b1..b3` should be
   estimated by regressing each contract's underlying index on the fitted factors
   rather than assumed.
3. `engine.py` — `LAM_J`, `GAMMA`, `JMEAN` are the jump parameters. With enough
   history these should be estimated jointly with the factor dynamics rather than set.

## Design notes

- The no-jump benchmark is variance-matched: its innovation covariance absorbs the
  jump variance, so tail differences are dependence structure, not scale.
- Every model's baseline intensities are recalibrated so all marginals equal the
  market-implied probabilities (to within 0.2pp). Verified in `engine.py`.
- The synthetic panel is generated from a known 3-factor DGP so the estimator can be
  checked against ground truth; it recovers all three factors with R^2 0.90-0.91.
