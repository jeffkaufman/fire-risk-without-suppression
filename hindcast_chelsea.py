"""Chelsea MA 1908 hindcast on the present-day footprint (density proxy; see README caveats).

Graph setup + forward-simulation helpers; hindcast_sweep.py runs the scenarios and compares the burned set
with the historical burned district (492 acres, ~2,800+ buildings, 1.5 mi x 0.75 mi; north flank held
at the B&M railroad, south flank at Winnisimmet Sq, burned out at Chelsea Creek / the marshes).
"""
import json, pickle, sys, os, numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
import model, osm

LAT, LON = 42.3918, -71.0328
WIND_FROM = 300.0            # origin -> far end of the burned district bears ~120 deg
ORIGIN_1908 = (-1270.0, 1020.0)   # West Third St near the Everett line (Boston Blacking Co.)

def setup():
    cache = "data/chelsea_graph.pkl"
    if os.path.exists(cache): return pickle.load(open(cache, "rb"))
    osm.ensure("chelsea", LAT, LON, 2800)
    polys, info = model.build(model.load("chelsea"), LAT, LON)
    nb, cents = model.neighbor_lists(polys)
    dst = np.concatenate([np.full(len(nb[v][0]), v, dtype=np.int32) for v in range(len(nb))])
    src = np.concatenate([nb[v][0] for v in range(len(nb))])
    gap = np.concatenate([nb[v][1] for v in range(len(nb))])
    brg = np.concatenate([nb[v][2] for v in range(len(nb))])
    G = dict(cents=cents, area=np.array([x["area"] for x in info]), noncomb=np.array([x["noncomb"] for x in info]),
             src=src, dst=dst, gap=gap, brg=brg)
    pickle.dump(G, open(cache, "wb"))
    return G

def edge_p(G, wind_from, m, e0, L, scale=1.0):
    old = model.SPREAD_SCALE; model.SPREAD_SCALE = scale
    pf, pe = model.edge_probs(G["gap"], G["brg"], G["area"][G["src"]], False, wind_from, m, e0, L)
    model.SPREAD_SCALE = old
    sus = np.where(G["noncomb"][G["dst"]], 0.35, 1.0)
    return 1 - (1 - pf * sus) * (1 - pe * sus)

def burn(G, p, seeds, rng):
    n = len(G["area"])
    o = rng.random(len(p)) < p
    A = csr_matrix((np.ones(o.sum(), dtype=np.int8), (G["src"][o], G["dst"][o])), shape=(n, n))
    y = np.zeros(n, dtype=bool)
    for s in seeds:
        if not y[s]:
            y[breadth_first_order(A, s, directed=True, return_predecessors=False)] = True
    return y


# B&M (now MBTA Newburyport/Rockport) main line through Chelsea, local metres, read off OSM
RAIL_X = np.array([-2500, -2000, -1400, -500, 0, 300, 500, 750, 1000, 2000])
RAIL_Y = np.array([ 1050,  1090,   790,  520, 370, 330, 345, 450,  620, 1240])
def zones(G):
    """district = today's dense fabric inside the 1908 burned district (south of the railroad, Everett Ave to
    Chelsea Creek); north = 500 m band across the tracks, which did not burn in 1908 (one building lost)."""
    x, y = G["cents"].T
    ry = np.interp(x, RAIL_X, RAIL_Y)
    inx = (x > -650) & (x < 850)
    district = inx & (y < ry - 15) & (y > -720)
    north = inx & (y > ry + 15) & (y < ry + 500)
    return district, north

if __name__ == "__main__":
    G = setup()
    print("buildings", len(G["area"]), "edges", len(G["src"]), "-- run hindcast_sweep.py for the hindcast itself")
