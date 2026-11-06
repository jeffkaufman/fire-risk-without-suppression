"""Parameter-recovery test for the LA pseudo-likelihood fit.

Simulate fires on the real LA footprints with the forward model's own mechanics
(background/ember ignition b_v, then independent structure-to-structure edges with
p(gap) = sigmoid(-(gap - d50 - a*lsa)/s)), then run the calib_fit2-style
pseudo-likelihood on the simulated outcome and see whether it returns the
transmission parameters that generated the data.
"""
import json, pickle, sys, numpy as np, shapely
from shapely.geometry import shape
from shapely import STRtree
from pyproj import Transformer
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

D = pickle.load(open("data/la/calib.pkl", "rb"))
C, areas, status_real, fire = D["cents"], D["areas"], D["status"], D["fire"]
n = len(C)
# rebuild polygons in the same order as calib_prep.py
d = json.load(open("data/la/buildings_dins.geojson"))
tf = Transformer.from_crs("EPSG:4326", "+proj=aeqd +lat_0=34.1 +lon_0=-118.35 +units=m", always_xy=True)
polys = []
for f in d["features"]:
    g = f["geometry"]
    if g is None: continue
    geom = shapely.transform(shape(g), lambda c: np.column_stack(tf.transform(c[:, 0], c[:, 1])))
    if not geom.is_valid: geom = geom.buffer(0)
    if geom.is_empty or geom.area < 8: continue
    polys.append(geom)
assert len(polys) == n
polys = np.array(polys, dtype=object)
tree = STRtree(polys)
src, dst = tree.query(polys, predicate="dwithin", distance=40.0)
keep = src != dst
src, dst = src[keep], dst[keep]
gap = shapely.distance(polys[src], polys[dst])
lsa = np.clip(np.log2(areas[src] / 120.0), -2, 2)
print("buildings", n, "directed pairs <40m", len(src), flush=True)

# ring structure 50-250 m (centroids)
kt = cKDTree(C)
A250 = kt.sparse_distance_matrix(kt, 250.0, output_type="coo_matrix")
m = A250.data > 50.0
RING = csr_matrix((np.ones(m.sum()), (A250.row[m], A250.col[m])), shape=(n, n))

def ring_frac(y, insp):
    num = RING @ (y * insp); den = RING @ insp.astype(float)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)

def fit(y, insp, tg, th0):
    """pseudo-likelihood as in calib_fit2 (area term, no heights, single intercept unless fi given)"""
    ring = ring_frac(y, insp)
    tg = tg[~np.isnan(ring[tg])]
    pos = -np.ones(n, dtype=int); pos[tg] = np.arange(len(tg))
    e = (pos[dst] >= 0) & (y[src] == 1)
    rows, g, l = pos[dst[e]], gap[e], lsa[e]
    yy = y[tg].astype(float); lr = logit(np.clip(ring[tg], .01, .99)); fi = fire[tg]
    def nll(th):
        c0, c1e, c1, d50, ls, a = th
        b = expit(np.where(fi == 0, c0, c1e) + c1 * lr)
        p = expit(-(g - d50 - a * l) / np.exp(ls))
        s_ = np.bincount(rows, weights=np.log1p(-np.clip(p, 0, 1 - 1e-9)), minlength=len(tg))
        P = np.clip(1 - (1 - b) * np.exp(s_), 1e-9, 1 - 1e-9)
        return -np.sum(yy * np.log(P) + (1 - yy) * np.log(1 - P))
    r = minimize(nll, th0, method="L-BFGS-B")
    return r.x

def simulate(rng, b, d50, s, a, pmax=1.0):
    p = pmax * expit(-(gap - d50 - a * lsa) / s)
    open_ = rng.random(len(src)) < p
    G = csr_matrix((np.ones(open_.sum()), (src[open_], dst[open_])), shape=(n, n))
    seeds = np.where(rng.random(n) < b)[0]
    # multi-source reachability: add a super-source
    S = csr_matrix((np.ones(len(seeds)), (np.full(len(seeds), n), seeds)), shape=(n + 1, n + 1))
    G2 = csr_matrix((G.data, G.indices, np.append(G.indptr, G.indptr[-1])), shape=(n + 1, n + 1)) + S
    order = breadth_first_order(G2, n, directed=True, return_predecessors=False)
    y = np.zeros(n, dtype=int); y[order[order < n]] = 1
    return y, len(seeds)

if __name__ == "__main__":
    th0 = np.array([-1.7, -2.3, 0.86, 4.4, np.log(4.0), 0.6])
    insp = status_real >= 0
    is_res = np.array(["Residence" in s or "Mobile Home" in s for s in D["stype"]])
    tg_real = np.where(is_res & insp)[0]
    th = fit((status_real == 1).astype(int), insp, tg_real, th0)
    print("REAL DATA refit: d50 %.2f s %.2f a %.2f  (c: %.2f %.2f %.2f)  burned frac of inspected %.3f" %
          (th[3], np.exp(th[4]), th[5], th[0], th[1], th[2], (status_real[insp] == 1).mean()), flush=True)
    # smooth background field from real burn pattern at ~400 m scale
    K = kt.sparse_distance_matrix(kt, 400.0, output_type="coo_matrix")
    Kc = csr_matrix((np.ones(len(K.data)), (K.row, K.col)), shape=(n, n))
    smooth = (Kc @ (status_real == 1).astype(float)) / np.maximum(Kc @ insp.astype(float), 1)
    rng = np.random.default_rng(0)
    allinsp = np.ones(n, dtype=bool); tg_all = np.arange(n)
    print("\ntruth: d50, s, a | background scale -> seeds, burned frac | recovered d50, s, a | p(2m) p(5m) p(10m) truth vs recovered")
    for (d50, s, a, pmax) in [(4.34, 3.97, 0.61, 1.0), (8.0, 3.0, 0.61, 1.0), (2.0, 2.0, 0.61, 1.0), (12.0, 4.0, 0.61, 1.0), (6.0, 2.0, 0.0, 1.0)]:
        for bscale in (0.1, 0.3, 0.7):
            y, ns = simulate(rng, np.clip(smooth * bscale, 0, 1), d50, s, a, pmax)
            th = fit(y, allinsp, tg_all, th0)
            pt = [expit(-(g_ - d50) / s) for g_ in (2, 5, 10)]
            pr = [expit(-(g_ - th[3]) / np.exp(th[4])) for g_ in (2, 5, 10)]
            print(f"  {d50:5.2f} {s:4.2f} {a:4.2f} | {bscale:.1f} -> {ns:6d} {y.mean():.3f} | {th[3]:6.2f} {np.exp(th[4]):5.2f} {th[5]:5.2f} | "
                  + " ".join(f"{x:.2f}/{z:.2f}" for x, z in zip(pt, pr)), flush=True)
