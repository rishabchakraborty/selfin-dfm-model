"""Step 3: Selfin's book of event contracts, and how each loads on the factors.

SYNTHETIC but shaped to match the deals Selfin describes publicly (JFK closure,
GPU rental prices, relegation). Replace with the real book and nothing changes.
Loadings b are on the ROTATED factors: F1 energy/climate, F2 financial, F3 logistics/AI-capex.
"""
import numpy as np, pandas as pd

# name, client sector, notional $m, mkt-implied 12m prob, retained %, b1, b2, b3, idio sd
BOOK = [
 ("JFK ground stop >6h",          "Airline",        12.0, 0.18, 0.30,  0.45,  0.20,  0.35, 1.00),
 ("H100 rental -25%",             "AI cloud",       15.0, 0.22, 0.35, -0.10,  0.35, -0.95, 1.00),
 ("Premier League relegation",    "Football club",   6.0, 0.25, 0.40,  0.00,  0.05,  0.00, 1.00),
 ("ERCOT neg-price hrs >800",     "Solar developer", 9.0, 0.30, 0.25,  0.85, -0.10,  0.15, 1.00),
 ("Panama transit restriction",   "Shipping",        8.0, 0.20, 0.30,  0.70,  0.15,  0.40, 1.00),
 ("Taiwan disruption breach",     "Semiconductor",  14.0, 0.07, 0.20,  0.15,  0.80,  0.55, 1.00),
 ("Midwest drought D3+",          "Ag processor",    7.0, 0.28, 0.35,  0.90, -0.05, -0.05, 1.00),
 ("Hashprice < $40/PH/day",       "Crypto miner",    5.0, 0.35, 0.40, -0.15,  0.60, -0.45, 1.00),
 ("West Coast port strike >5d",   "Retailer",       10.0, 0.12, 0.25,  0.10,  0.35,  0.65, 1.00),
 ("FL hurricane Cat3+ landfall",  "Insurer",        13.0, 0.24, 0.20,  0.75,  0.10, -0.05, 1.00),
 ("PJM capacity >$400/MW-day",    "Datacenter",     11.0, 0.16, 0.30,  0.60,  0.15,  0.50, 1.00),
 ("Lithium carbonate -30%",       "EV maker",        6.0, 0.19, 0.35, -0.05,  0.40, -0.60, 1.00),
]
COLS = ["contract","sector","notional_m","p_mkt","retained","b1","b2","b3","idio_sd"]

def load_book():
    df = pd.DataFrame(BOOK, columns=COLS)
    df["exposure_m"] = df.notional_m * df.retained          # what Selfin warehouses
    # calibrate baseline log-intensity so that at F=0 the 12m prob equals p_mkt
    df["a"] = np.log(-np.log(1 - df.p_mkt) / 12.0)
    return df

if __name__ == "__main__":
    b = load_book()
    print(b[["contract","sector","notional_m","retained","exposure_m","p_mkt","b1","b2","b3"]].to_string(index=False))
    print(f"\ngross notional ${b.notional_m.sum():.0f}m | retained exposure ${b.exposure_m.sum():.1f}m")
    print(f"expected payout (standalone) ${(b.exposure_m*b.p_mkt).sum():.2f}m")
