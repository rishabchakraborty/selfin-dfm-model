"""Step 5: turn the simulation into things a desk can act on -
capital allocation, pricing loadings, concentration flags, hedgeable share."""
import numpy as np, pandas as pd
from scipy import stats
from book import load_book
from engine import (var1, sim_factors, cum_hazard, calibrate, evt_tail,
                    NPATH, H, LOADING, LAM_J, GAMMA, JMEAN)

bk  = load_book()
Fh  = pd.read_pickle("Fhat.pkl")
Phi, Sig = var1(Fh); F0 = Fh.values[-1]
B, p_mkt, expo = bk[["b1","b2","b3"]].values, bk.p_mkt.values, bk.exposure_m.values
prem_total = (expo * p_mkt * (1 + LOADING)).sum()

rng = np.random.default_rng(42)
Fp  = sim_factors(Phi, Sig, F0, NPATH, True, rng)
a   = calibrate(Fp, B, p_mkt, bk.a.values)
P   = 1 - np.exp(-cum_hazard(Fp, a, B))
ev  = rng.random(P.shape) < P
loss_i = ev * expo
Lgross = loss_i.sum(1)

# ---------- 1. Euler / marginal ES allocation at 99.5% ----------------------
q   = np.quantile(Lgross, .995)
tail = Lgross >= q
mES = loss_i[tail].mean(0)                       # E[loss_i | L in tail]
bk["ES_contrib_m"]  = mES
bk["ES_share_pct"]  = 100*mES/mES.sum()
bk["standalone_EL_m"] = expo*p_mkt
bk["premium_m"]     = expo*p_mkt*(1+LOADING)

# ---------- 2. capital intensity and the pricing loading it implies ---------
COST_OF_CAPITAL = 0.15
bk["capital_m"]      = mES - bk.standalone_EL_m          # unexpected loss
bk["req_loading_pct"]= 100*COST_OF_CAPITAL*bk.capital_m/bk.standalone_EL_m
bk["charged_pct"]    = 100*LOADING
bk["pricing_gap_pp"] = bk.req_loading_pct - bk.charged_pct

# ---------- 3. hedgeable share: freeze the factors, keep idiosyncratic ------
rng2 = np.random.default_rng(7)
Pbar = 1 - np.exp(-np.exp(a)*H)                           # factors at zero
ev_id = rng2.random((NPATH, len(bk))) < Pbar
var_total = Lgross.var()
var_idio  = (ev_id*expo).sum(1).var()
hedgeable = 1 - var_idio/var_total

# ---------- 4. how much of the tail is jump-driven --------------------------
rngj = np.random.default_rng(42)
_ = sim_factors(Phi, Sig, F0, NPATH, True, np.random.default_rng(42))  # same stream
rngc = np.random.default_rng(42); jumped = np.zeros(NPATH, bool)
Ftmp = np.tile(F0,(NPATH,1)); Lc = np.linalg.cholesky(Sig+1e-10*np.eye(3))
for t in range(H):
    Ftmp = Ftmp @ Phi.T + rngc.standard_normal((NPATH,3)) @ Lc.T
    hit  = rngc.random(NPATH) < LAM_J
    jumped |= hit
    Ftmp = Ftmp + (rngc.exponential(JMEAN, NPATH)*hit)[:,None]*GAMMA

print("="*78)
print("CAPITAL ALLOCATION AND PRICING  (99.5% ES, 12m, $m)")
print("="*78)
show = bk[["contract","sector","exposure_m","p_mkt","standalone_EL_m","premium_m",
           "ES_contrib_m","ES_share_pct","req_loading_pct","pricing_gap_pp"]]
print(show.sort_values("ES_share_pct", ascending=False).round(2).to_string(index=False))

print(f"\ngross ES(99.5) on payouts      ${Lgross[tail].mean():.2f}m")
print(f"premium collected              ${prem_total:.2f}m")
print(f"net capital at risk            ${Lgross[tail].mean()-prem_total:.2f}m")
print(f"\nportfolio variance explained by factors (hedgeable): {100*hedgeable:.1f}%")
print(f"purely idiosyncratic (diversifiable):                {100*(1-hedgeable):.1f}%")
print(f"share of 99.5% tail scenarios containing a jump:     {100*jumped[tail].mean():.1f}%")
print(f"unconditional share of paths with a jump:            {100*jumped.mean():.1f}%")

# ---------- 5. EVT diagnostics: threshold stability --------------------------
print("\n" + "="*78); print("EVT THRESHOLD STABILITY (GPD fit to book losses)"); print("="*78)
losses = Lgross - prem_total
rows=[]
for qu in (0.90, 0.925, 0.95, 0.975, 0.99):
    d = evt_tail(losses, qu)
    rows.append([qu, d["u"], d["n_exc"], d["xi"], d["beta"], d["es995"], d["es999"]])
print(pd.DataFrame(rows, columns=["thresh_q","u","n_exc","xi","beta","ES99.5","ES99.9"]).round(3).to_string(index=False))
bk.to_pickle("book_out.pkl")
np.save("Lgross.npy", Lgross.astype(np.float32)); np.save("jumped.npy", jumped)
print(f"\nhedgeable={hedgeable:.4f}")
