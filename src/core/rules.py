import numpy as np

def rule_burst_score(df):
    comps = []
    for c in ["vel_5m","same_amt_5m","fanout_5m","z_amount"]:
        x = df[c].to_numpy(float)
        mx, mn = np.nanmax(x), np.nanmin(x)
        comps.append(np.zeros_like(x) if mx==mn else (x-mn)/(mx-mn))
    v,s,f,z = comps
    return 0.35*v + 0.35*s + 0.2*f + 0.1*z
