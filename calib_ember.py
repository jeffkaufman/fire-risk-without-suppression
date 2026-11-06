"""Calibrate the model's structure-to-structure EMBER kernel on LA 2025 (Eaton + Palisades).

calib_fit2 absorbed all non-adjacent exposure into a 'ring burned fraction' background, so the forward
model's ember term (e0 * exp(-d/L), downwind cone, <= 250 m) was never fitted.  Here the background is
replaced by the model's own kernel summed over destroyed buildings:

  P(v burned) = 1 - (1 - b0_fire) * exp(-e0 * S_v) * prod_{destroyed u, gap<40m} (1 - p_flame(gap_uv))
  S_v = sum_{destroyed u, 0 < gap <= RMAX, [bearing within +/-HALF of downwind]} exp(-gap/L)

Caveat: wildland embers are correlated with S_v, so e0 is an upper-ish estimate of structure-borne spotting
in extreme (Santa Ana, 60-100 mph gust) wind.
"""
import pickle, sys, numpy as np, shapely
from shapely import STRtree
from scipy.optimize import minimize
from scipy.special import expit, logit
import calib_recovery as CR

RMAX = 250.0
def pairs250():
    import os
    f = "data/la/pairs250.npz"
    if os.path.exists(f):
        z = np.load(f); return z["src"], z["dst"], z["gap"], z["brg"]
    tree = STRtree(CR.polys)
    burned = np.where(CR.status_real == 1)[0]
    s, d = tree.query(CR.polys[burned], predicate="dwithin", distance=RMAX)   # s indexes burned
    s = burned[s]; k = s != d; s, d = s[k], d[k]
    gap = shapely.distance(CR.polys[s], CR.polys[d])
    dx = CR.C[d, 0] - CR.C[s, 0]; dy = CR.C[d, 1] - CR.C[s, 1]
    brg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    np.savez(f, src=s, dst=d, gap=gap.astype(np.float32), brg=brg.astype(np.float32))
    return s, d, gap, brg

def main():
    D = CR.D; st = D["status"]; n = CR.n
    insp = st >= 0
    is_res = np.array(["Residence" in s or "Mobile Home" in s for s in D["stype"]])
    ring = CR.ring_frac((st == 1).astype(int), insp)
    tg = np.where(is_res & insp & ~np.isnan(ring))[0]
    pos = -np.ones(n, dtype=int); pos[tg] = np.arange(len(tg))
    y = st[tg].astype(float); fi = CR.fire[tg]; lr = logit(np.clip(ring[tg], .01, .99))
    # radiant pairs
    e = (pos[CR.dst] >= 0) & (st[CR.src] == 1)
    rrow, rg, rl = pos[CR.dst[e]], CR.gap[e], CR.lsa[e]
    # ember pairs
    s, d, gap, brg = pairs250()
    k = pos[d] >= 0
    erow, egap, ebrg = pos[d[k]], gap[k].astype(float), brg[k].astype(float)
    print("targets", len(tg), "ember pairs", len(erow), flush=True)

    def S(L, wind_from, half):
        w = np.exp(-egap / L)
        if half < 180:
            w = w * (np.cos(np.radians(ebrg - (wind_from + 180))) > np.cos(np.radians(half)))
        return np.bincount(erow, weights=w, minlength=len(tg))

    def fit(Sv, use_ring=False, th0=None):
        def nll(th):
            c0, c1e, le0, d50, ls, a, c1 = th
            lin = np.where(fi == 0, c0, c1e) + (c1 * lr if use_ring else 0)
            lb = np.log1p(-expit(lin))                     # log(1-b0)
            lp = np.bincount(rrow, weights=np.log1p(-np.clip(expit(-(rg - d50 - a * rl) / np.exp(ls)), 0, 1 - 1e-9)), minlength=len(tg))
            P = np.clip(1 - np.exp(lb - (np.exp(le0) * Sv if Sv is not None else 0) + lp), 1e-9, 1 - 1e-9)
            return -np.sum(y * np.log(P) + (1 - y) * np.log(1 - P))
        th0 = np.array([-2.0, -2.0, np.log(0.02), 4.4, np.log(4.0), 0.6, 0.85]) if th0 is None else th0
        r = minimize(nll, th0, method="L-BFGS-B")
        return r.x, r.fun

    th, f = fit(None, use_ring=True)
    print(f"ring background (calib_fit2 form): NLL {f:.0f}  d50 {th[3]:.2f} s {np.exp(th[4]):.2f}")
    best = None
    print("kernel-only fits:  half-angle  wind_from  L   -> NLL   e0   d50   s   b0(Pal) b0(Eat)")
    for half, wf in [(180, 0), (35, 0), (35, 20), (35, 40), (35, 60), (60, 20), (60, 40), (90, 20), (90, 40)]:
        for L in (40, 90, 200, 1e6):
            th, f = fit(S(L, wf, half))
            print(f"   {half:4d} {wf:4d} {L:8.0f} -> {f:8.0f}  e0 {np.exp(th[2]):.4f}  d50 {th[3]:.2f} s {np.exp(th[4]):.2f}  b0 {expit(th[0]):.3f} {expit(th[1]):.3f}", flush=True)
            if best is None or f < best[0]: best = (f, half, wf, L, th)
    print("best:", best[:4])
    pickle.dump(dict(best=best), open("data/la/fit_ember.pkl", "wb"))

if __name__ == "__main__":
    main()
