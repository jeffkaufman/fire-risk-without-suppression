"""Fit structure-to-structure spread curve to LA 2025 fires by pseudo-likelihood.

P(v burned) = 1 - (1 - b_v) * prod_{u destroyed, gap<40m} (1 - p_uv)
  b_v  = sigmoid(c_fire + c1 * logit(ring_v))          background (ember/wildland) exposure
  p_uv = pmax * sigmoid(-(gap - d50_uv) / s)
  d50_uv = d50 + k * cos(bearing_uv - downwind) + a * clip(log2(area_u/120), -2, 2)
"""
import pickle, numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

D = pickle.load(open("data/la/calib.pkl", "rb"))
st, fire, targets, nbrs, ring = D["status"], D["fire"], D["targets"], D["nbrs"], D["ring"]
tg = np.array([v for v in targets if not np.isnan(ring[v])])
y = st[tg].astype(float)
lr = logit(np.clip(ring[tg], 0.01, 0.99))
fi = fire[tg]

# flatten neighbor pairs
rows, gaps, brg, sa = [], [], [], []
for j, v in enumerate(tg):
    idx, g, b, a = nbrs[v]
    rows += [j] * len(idx); gaps += list(g); brg += list(b); sa += list(a)
rows = np.array(rows); gaps = np.array(gaps); brg = np.array(brg)
lsa = np.clip(np.log2(np.array(sa) / 120.0), -2, 2)
WIND_FROM = 40.0                                   # Santa Ana NE winds
cosang = np.cos(np.radians(brg - (WIND_FROM + 180)))

def unpack(th):
    c0, c1e, c1, d50, ls, lpm, k, a = th
    return c0, c1e, c1, d50, np.exp(ls), expit(lpm), k, a

def nll(th, return_p=False):
    c0, c1e, c1, d50, s, pmax, k, a = unpack(th)
    b = expit(np.where(fi == 0, c0, c1e) + c1 * lr)
    dd = d50 + k * cosang + a * lsa
    p = pmax * expit(-(gaps - dd) / s)
    logsurv = np.bincount(rows, weights=np.log1p(-np.clip(p, 0, 1 - 1e-9)), minlength=len(tg))
    P = 1 - (1 - b) * np.exp(logsurv)
    P = np.clip(P, 1e-9, 1 - 1e-9)
    if return_p: return P, b
    return -np.sum(y * np.log(P) + (1 - y) * np.log(1 - P))

th0 = np.array([-2.0, -2.0, 1.0, 5.0, np.log(2.0), 2.0, 0.0, 0.5])
r = minimize(nll, th0, method="Nelder-Mead", options=dict(maxiter=20000, maxfev=20000, xatol=1e-4, fatol=1e-3))
r = minimize(nll, r.x, method="BFGS")
c0, c1e, c1, d50, s, pmax, k, a = unpack(r.x)
print(f"NLL {r.fun:.1f}  n={len(tg)}  burned={y.mean():.3f}")
print(f"background: Palisades c0={c0:.2f}, Eaton c0={c1e:.2f}, ring slope={c1:.2f}")
print(f"spread: d50={d50:.2f} m, width s={s:.2f} m, pmax={pmax:.3f}, wind shift k={k:.2f} m, source-size a={a:.2f} m/doubling")

# bootstrap-ish uncertainty from inverse Hessian (BFGS approx)
se = np.sqrt(np.diag(r.hess_inv))
print("approx SE (raw params):", np.round(se, 3))

# curve
print("\nper-neighbor ignition prob (typical 120 m2 source):")
for g in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]:
    pc = pmax * expit(-(g - d50) / s)
    pdn = pmax * expit(-(g - d50 - k) / s); pup = pmax * expit(-(g - d50 + k) / s)
    print(f"  gap {g:3d} m: cross {pc:.3f}  downwind {pdn:.3f}  upwind {pup:.3f}")

# calibration check: predicted vs observed by decile
P, b = nll(r.x, True)
q = np.quantile(P, np.linspace(0, 1, 11))
print("\ncalibration (pred vs obs by decile):")
for lo, hi in zip(q[:-1], q[1:]):
    m = (P >= lo) & (P <= hi)
    print(f"  {P[m].mean():.3f} vs {y[m].mean():.3f}  (n={m.sum()})")
# share of burns attributed to neighbors vs background
print("\nmean background b among burned:", b[y == 1].mean().round(3), " mean P among burned:", P[y == 1].mean().round(3))
pickle.dump(dict(theta=r.x, d50=d50, s=s, pmax=pmax, k=k, a=a), open("data/la/fit.pkl", "wb"))
