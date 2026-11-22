"""
Selfin event-risk engine - Step 1: the observable panel and the dynamic factor model.

The panel is 14 monthly indicators Selfin can pull from public/vendor sources.
Data here is SYNTHETIC, generated from a known 3-factor structure so we can
verify the estimator recovers it. Swap in real series and nothing else changes.
"""
import numpy as np, pandas as pd

RNG = np.random.default_rng(20251121)
T, K = 192, 3          # 16 years monthly, 3 latent factors

SERIES = [
    # name                        F1    F2    F3     idio sd
    ("VIX",                      0.88, 0.10, 0.05,  0.45),
    ("MOVE (rates vol)",         0.80, 0.08, 0.02,  0.55),
    ("HY OAS",                   0.85,-0.05, 0.15,  0.50),
    ("10y-3m term spread",      -0.62, 0.05, 0.20,  0.65),
    ("Broad USD index",          0.55,-0.25,-0.10,  0.70),
    ("Equity drawdown",          0.78, 0.05, 0.25,  0.55),
    ("Brent",                    0.12, 0.84,-0.05,  0.50),
    ("Henry Hub",                0.05, 0.78, 0.02,  0.58),
    ("ERCOT on-peak power",      0.02, 0.72, 0.18,  0.60),
    ("ONI (ENSO)",              -0.05, 0.58,-0.08,  0.72),
    ("US drought index",         0.00, 0.52,-0.05,  0.78),
    ("Drewry container index",   0.22, 0.30, 0.70,  0.52),
    ("H100 spot rental",        -0.08,-0.05, 0.86,  0.46),
    ("Semi capex / DC vacancy",  0.15, 0.02, 0.80,  0.50),
]
NAMES   = [s[0] for s in SERIES]
LAMBDA  = np.array([[s[1], s[2], s[3]] for s in SERIES])   # N x K loadings
IDIO_SD = np.array([s[4] for s in SERIES])
N = len(SERIES)

# --- true factor dynamics: VAR(1), persistent, mildly cross-correlated -------
PHI = np.array([[0.82, 0.06, 0.04],
                [0.03, 0.74, 0.05],
                [0.05, 0.02, 0.86]])
Q   = np.array([[1.00, 0.22, 0.14],
                [0.22, 1.00, 0.10],
                [0.14, 0.10, 1.00]]) * 0.35**2

def simulate_panel(T=T, rng=RNG):
    F = np.zeros((T, K))
    L = np.linalg.cholesky(Q)
    for t in range(1, T):
        F[t] = PHI @ F[t-1] + L @ rng.standard_normal(K)
    F = (F - F.mean(0)) / F.std(0)                      # standardise factors
    idio = np.zeros((T, N))
    for i in range(N):                                   # AR(1) idiosyncratic
        for t in range(1, T):
            idio[t, i] = 0.35*idio[t-1, i] + IDIO_SD[i]*rng.standard_normal()
    X = F @ LAMBDA.T + idio
    idx = pd.period_range("2010-01", periods=T, freq="M")
    return pd.DataFrame(X, index=idx, columns=NAMES), pd.DataFrame(F, index=idx,
                        columns=["F1_financial", "F2_energy_climate", "F3_logistics_tech"])

if __name__ == "__main__":
    X, Ftrue = simulate_panel()
    print("panel:", X.shape, "| factors:", Ftrue.shape)
    print(X.iloc[:3, :5].round(2))
    print("\nshare of panel variance from factors (true):",
          round(float((LAMBDA**2).sum() / ((LAMBDA**2).sum() + (IDIO_SD**2).sum()/(1-0.35**2))), 3))
