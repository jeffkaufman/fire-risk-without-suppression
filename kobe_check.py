"""Calm/ordinary-day check against Kobe 1995 (Hokugo 1997, Fire Safety Science 5:947-958).

Kobe, 17 Jan 1995: mean wind 2.6 m/s, RH 66-70%, no hydrant water.  Of 150 investigated fire sites,
73 involved a single building (49%), 77 several (51%); 44 (29%) burned >= 1,000 m2 of site area and
7 (4.7%) >= 33,000 m2.  Spread was 20-30 m/h.  Caveats: brigades did suppress many fires early (pushes sizes
down); collapsed buildings bridged streets and the big fires were in fabric denser than anything here (pushes up).

Here: forward fire-size distribution from a single ignition on calm/ordinary days in dense fabric
(present-day Chelsea MA between the railroad and the waterfront).
Site area is converted to footprint with a 60% coverage ratio: 1,000 m2 -> 600 m2, 33,000 m2 -> 20,000 m2.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
import model, weather, hindcast_chelsea as H

def sizes(G, pool, m, e0, L, scale, wx, rng, nreal=40, nseed=25):
    n = len(G["area"]); out = []
    for _ in range(nreal):
        wf = rng.choice(model.WIND_DIRS, p=wx) + rng.uniform(-22.5, 22.5)
        p = H.edge_p(G, wf, m, e0, L, scale)
        o = rng.random(len(p)) < p
        A = csr_matrix((np.ones(o.sum(), dtype=np.int8), (G["src"][o], G["dst"][o])), shape=(n, n))
        for s in rng.choice(pool, nseed, replace=False):
            b = breadth_first_order(A, s, directed=True, return_predecessors=False)
            out.append((len(b), G["area"][b].sum()))
    return np.array(out)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    G = H.setup(); district, _ = H.zones(G)
    pool = np.where(district & ~G["noncomb"])[0]
    rose = {k: np.array(v["dirs"]) for k, v in weather.day_types("BOS")["types"].items()}
    print("Kobe 1995 (light wind, partial suppression):      P(>1 bldg) 0.51   P(>=600 m2 footprint) 0.29   P(>=20,000 m2) 0.047")
    print(f"\nChelsea district (nearest-gap median 2.4 m): {len(pool)} candidate ignition buildings")
    for nm, dt, m, e0, L, sc in [("calm/damp as-is (m 0.8)", "calm/damp", 0.8, 0.0, 40, 1.0), ("ordinary as-is (m 0.9)", "ordinary", 0.9, 0.002, 40, 1.0),
                                 ("ordinary, m 1.0", "ordinary", 1.0, 0.002, 40, 1.0), ("ordinary, m 1.0, spread x1.25", "ordinary", 1.0, 0.002, 40, 1.25),
                                 ("ordinary, m 1.0, spread x1.5", "ordinary", 1.0, 0.002, 40, 1.5)]:
        r = sizes(G, pool, m, e0, L, sc, rose[dt], rng)
        print(f"  {nm:32s} P(>1 bldg) {np.mean(r[:,0] > 1):.2f}   P(>=600 m2) {np.mean(r[:,1] >= 600):.2f}   P(>=20,000 m2) {np.mean(r[:,1] >= 20000):.3f}"
              f"   median/mean/p90 bldgs {np.median(r[:,0]):.0f}/{r[:,0].mean():.1f}/{np.percentile(r[:,0], 90):.0f}", flush=True)
