"""Step 2: estimate the dynamic factor model, rotate it so the factors are nameable,
and check it recovers the true structure."""
import numpy as np, pandas as pd, warnings
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from panel import simulate_panel, NAMES
warnings.filterwarnings("ignore")

def varimax(L, tol=1e-9, itmax=500):
    """Kaiser varimax. A rotation inside the factor space: it changes nothing
    about fit, only which basis we report, so the factors can be named."""
    L = np.asarray(L, float); p, k = L.shape; R = np.eye(k); d = 0
    for _ in range(itmax):
        Lam = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lam**3 - Lam @ np.diag((Lam**2).sum(0))/p))
        R = u @ vt; d_new = s.sum()
        if d_new < d*(1+tol): break
        d = d_new
    return L @ R, R

def fit_dfm(X, k_factors=3, factor_order=1, error_order=1, maxiter=2000):
    Xs = (X - X.mean()) / X.std()
    mod = DynamicFactor(Xs, k_factors=k_factors, factor_order=factor_order,
                        error_order=error_order)
    start = mod.fit(disp=False, maxiter=150, method="lbfgs")        # warm start
    res   = mod.fit(start.params, disp=False, maxiter=maxiter, method="nm")
    res   = mod.fit(res.params,   disp=False, maxiter=maxiter, method="lbfgs")

    F = res.factors.filtered.T
    L = np.column_stack([[res.params[f"loading.f{j+1}.{c}"] for c in Xs.columns]
                         for j in range(k_factors)])
    Lrot, R = varimax(L)                       # rotate loadings ...
    Frot = F @ R                               # ... and counter-rotate factors
    sgn  = np.sign(Lrot[np.abs(Lrot).argmax(0), range(k_factors)])  # fix sign
    Lrot, Frot = Lrot*sgn, Frot*sgn
    order = np.argsort(-(Lrot**2).sum(0))      # strongest factor first
    Lrot, Frot = Lrot[:, order], Frot[:, order]

    cols = ["F1", "F2", "F3"][:k_factors]
    Fhat = pd.DataFrame(Frot, index=X.index, columns=cols)
    Fhat = (Fhat - Fhat.mean()) / Fhat.std()
    return res, Fhat, pd.DataFrame(Lrot, index=Xs.columns, columns=cols)

def canon_r2(Ftrue, Fhat):
    A = np.c_[np.ones(len(Fhat)), Fhat.values]
    return pd.Series({c: 1 - ((Ftrue[c].values - A@np.linalg.lstsq(A, Ftrue[c].values, rcond=None)[0])**2).sum()
                         / ((Ftrue[c].values - Ftrue[c].values.mean())**2).sum() for c in Ftrue.columns})

def common_share(X, Fhat):
    Xs = (X - X.mean())/X.std()
    fit = Fhat.values @ np.linalg.lstsq(Fhat.values, Xs.values, rcond=None)[0]
    return pd.Series(1 - ((Xs.values-fit)**2).sum(0)/(Xs.values**2).sum(0), index=X.columns)

if __name__ == "__main__":
    X, Ftrue = simulate_panel()
    res, Fhat, load = fit_dfm(X)
    print(f"converged: {res.mle_retvals['converged']}   loglik {res.llf:,.1f}   AIC {res.aic:,.1f}")
    print("\n--- recovery of true factors (R^2 on estimated factor space) ---")
    print(canon_r2(Ftrue, Fhat).round(3).to_string())
    print("\n--- varimax-rotated loadings: the factors are now nameable ---")
    print(load.round(2).to_string())
    cs = common_share(X, Fhat)
    print("\n--- share of each series' variance explained by the 3 factors ---")
    print(cs.sort_values(ascending=False).round(2).to_string())
    print(f"\npanel-average common share: {cs.mean():.3f}")
    Fhat.to_pickle("Fhat.pkl"); X.to_pickle("X.pkl"); load.to_pickle("load.pkl")
    cs.to_pickle("common_share.pkl"); Ftrue.to_pickle("Ftrue.pkl")
