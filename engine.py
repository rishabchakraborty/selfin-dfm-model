"""Step 4: jump-diffusion factor simulation, event generation, EVT tail, risk allocation.

Three models, deliberately made comparable:
  (1) INDEPENDENT   - same marginals, zero correlation  (the naive book view)
  (2) DIFFUSION     - factor correlation, Gaussian innovations only
  (3) JUMP-DIFFUSION- factor correlation + common Poisson jumps
(2) and (3) are VARIANCE-MATCHED: the no-jump model's innovation covariance is
inflated so both have identical one-step factor variance. And every model has its
baseline intensities recalibrated so all 12 marginal probabilities equal p_mkt.
So any difference in the tail is pure dependence structure, not scale.
"""
import numpy as np, pandas as pd
from scipy import stats, optimize
from book import load_book

H       = 12            # horizon, months
NPATH   = 200_000
LOADING = 0.25          # premium charged = p_mkt * (1 + LOADING)
LAM_J   = 0.5 / 12      # systemic jump arrivals per month (~0.5/yr)
GAMMA   = np.array([0.60, 1.40, -0.50])   # jump loading on F1/F2/F3
JMEAN   = 1.20          # Exp mean of jump magnitude

def var1(F):
    """VAR(1) on the filtered factors -> transition matrix and innovation cov."""
    Y, Xl = F.values[1:], F.values[:-1]
    Phi = np.linalg.lstsq(Xl, Y, rcond=None)[0].T
    E   = Y - Xl @ Phi.T
    return Phi, np.cov(E.T)

def sim_factors(Phi, Sig, F0, n, jumps, rng):
    """Returns (n, H, 3) factor paths."""
    L = np.linalg.cholesky(Sig + 1e-10*np.eye(3))
    out = np.empty((n, H, 3)); F = np.tile(F0, (n, 1))
    for t in range(H):
        F = F @ Phi.T + rng.standard_normal((n, 3)) @ L.T
        if jumps:
            hit = rng.random(n) < LAM_J
            S   = rng.exponential(JMEAN, n) * hit
            F   = F + S[:, None] * GAMMA
        out[:, t] = F
    return out

def cum_hazard(Fp, a, B):
    """Sum_t exp(a_i + b_i . F_t) -> (n, ncontract) cumulative hazard."""
    n = Fp.shape[0]; out = np.zeros((n, len(a)))
    for t in range(Fp.shape[1]):
        out += np.exp(a + Fp[:, t] @ B.T)
    return out

def calibrate(Fp, B, p_mkt, a0):
    """Shift each contract's baseline so its SIMULATED 12m prob equals p_mkt."""
    a = a0.copy()
    for i in range(len(a)):
        s = np.zeros(Fp.shape[0])
        for t in range(Fp.shape[1]):
            s += np.exp(Fp[:, t] @ B[i])
        f = lambda ai: np.mean(1 - np.exp(-np.exp(ai)*s)) - p_mkt[i]
        a[i] = optimize.brentq(f, a0[i]-6, a0[i]+6, xtol=1e-10)
    return a

def run(model, bk, Phi, Sig, F0, rng):
    B, p_mkt = bk[["b1","b2","b3"]].values, bk.p_mkt.values
    expo     = bk.exposure_m.values
    if model == "independent":
        ev = rng.random((NPATH, len(bk))) < p_mkt
    else:
        Fp = sim_factors(Phi, Sig, F0, NPATH, model == "jump", rng)
        a  = calibrate(Fp, B, p_mkt, bk.a.values)
        P  = 1 - np.exp(-cum_hazard(Fp, a, B))
        ev = rng.random(P.shape) < P
    prem = (expo * p_mkt * (1 + LOADING)).sum()
    loss_i = ev * expo
    return prem - loss_i.sum(1), loss_i, ev      # P&L, per-contract loss, events

def evt_tail(losses, q_u=0.95):
    """Peaks-over-threshold: fit a GPD to exceedances, read VaR/ES off the fit."""
    u    = np.quantile(losses, q_u)
    exc  = losses[losses > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    Nu_n = len(exc)/len(losses)
    def var_(q):
        return u + beta/xi*(((1-q)/Nu_n)**(-xi) - 1)
    def es_(q):
        v = var_(q); return (v + beta - xi*u)/(1-xi)
    return dict(xi=xi, beta=beta, u=u, n_exc=len(exc),
                var99=var_(.99), var995=var_(.995), var999=var_(.999),
                es99=es_(.99), es995=es_(.995), es999=es_(.999))

if __name__ == "__main__":
    bk  = load_book()
    Fh  = pd.read_pickle("Fhat.pkl")
    Phi, Sig = var1(Fh)
    F0  = Fh.values[-1]
    # variance-match: no-jump innovation cov absorbs the jump variance
    jvar = LAM_J * (GAMMA**2) * 2*JMEAN**2
    Sig_nojump = Sig + np.diag(jvar)
    print("VAR(1) eigenvalues:", np.round(np.abs(np.linalg.eigvals(Phi)), 3))
    print("per-step jump variance added to each factor:", np.round(jvar, 4))

    res, store = {}, {}
    for name, sig, mdl in [("independent", Sig, "independent"),
                           ("diffusion",   Sig_nojump, "diff"),
                           ("jump",        Sig, "jump")]:
        rng = np.random.default_rng(42)
        pnl, loss_i, ev = run(mdl, bk, Phi, sig, F0, rng)
        losses = -pnl
        res[name] = dict(mean_pnl=pnl.mean(), sd=losses.std(),
                         emp_var995=np.quantile(losses, .995),
                         emp_es995=losses[losses >= np.quantile(losses, .995)].mean(),
                         **evt_tail(losses))
        store[name] = (losses, loss_i, ev)
        np.save(f"losses_{name}.npy", losses.astype(np.float32))

    out = pd.DataFrame(res).T
    print("\n=== book P&L, $m, 12-month horizon ===")
    print(out[["mean_pnl","sd","emp_var995","emp_es995","es995","xi"]].round(3).to_string())
    print("\nmarginal check (simulated 12m prob vs market):")
    for n in store: print(f"  {n:12s} max abs dev = {np.abs(store[n][2].mean(0)-bk.p_mkt.values).max():.4f}")
    pd.to_pickle({k:(v[1].astype(np.float32), v[2]) for k,v in store.items()}, "sim.pkl")
    out.to_pickle("results.pkl")
