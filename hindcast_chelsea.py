"""Chelsea MA 1908 hindcast on the present-day footprint (density proxy; see README caveats).

Graph setup + forward-simulation helpers; hindcast_sweep.py runs the scenarios and compares the burned set
with the historical burned district (492 acres, ~2,800+ buildings, 1.5 mi x 0.75 mi; north flank held
at the B&M railroad, south flank at Winnisimmet Sq, burned out at Chelsea Creek / the marshes).
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
import model, osm

LAT, LON = 42.3918, -71.0328
WIND_FROM = 300.0            # origin -> far end of the burned district bears ~120 deg

def setup():
    osm.ensure("chelsea", LAT, LON, 2800)
    polys, info = model.build(model.load("chelsea"), LAT, LON)
    G = model.graph(polys, info, cache="data/graph_chelsea.npz")
    G["dst"] = np.repeat(np.arange(len(G["area"])), np.diff(G["indptr"])).astype(np.int32)
    return G

def edge_p(G, wind_from, day, scale=1.0):
    """per-edge spread probability for a day type tuple (name, m, wind weight, e0, L)"""
    old = model.SPREAD_SCALE; model.SPREAD_SCALE = scale
    pf, pe = model.edge_probs(G["gap"], G["brg"], G["area"][G["src"]], G["noncomb"][G["dst"]], wind_from, day)
    model.SPREAD_SCALE = old
    return 1 - (1 - pf) * (1 - pe)

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
