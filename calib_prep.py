"""Prepare LA 2025 (Eaton + Palisades) structure data for calibrating structure-to-structure spread."""
import json, pickle, numpy as np, shapely, shapely.ops
from shapely.geometry import shape
from shapely import STRtree
from pyproj import Transformer

d = json.load(open("data/la/buildings_dins.geojson"))
tf = Transformer.from_crs("EPSG:4326", "+proj=aeqd +lat_0=34.1 +lon_0=-118.35 +units=m", always_xy=True)
BURN = {"Destroyed (>50%)", "Major (26-50%)"}
SURV = {"No Damage", "Affected (1-9%)", "Minor (10-25%)"}
polys, status, stype, lon, hts = [], [], [], [], []
for f in d["features"]:
    g = f["geometry"]
    if g is None: continue
    geom = shape(g)
    geom = shapely.transform(geom, lambda c: np.column_stack(tf.transform(c[:,0], c[:,1])))
    if not geom.is_valid: geom = geom.buffer(0)
    if geom.is_empty or geom.area < 8: continue
    p = f["properties"]
    dm = p["DAMAGE"]
    status.append(1 if dm in BURN else (0 if dm in SURV else -1))   # -1: not inspected / unknown
    stype.append(p["STRUCTURET"] or "")
    hts.append(p["HEIGHT"] or np.nan)
    polys.append(geom)
    lon.append(shape(g).centroid.x)
status = np.array(status); lon = np.array(lon)
fire = np.where(lon < -118.4, 0, 1)   # 0 Palisades, 1 Eaton
areas = np.array([p.area for p in polys])
cents = np.array([[p.centroid.x, p.centroid.y] for p in polys])
tree = STRtree(polys)

# targets: inspected residences
is_res = np.array(["Residence" in s or "Mobile Home" in s for s in stype])
targets = np.where(is_res & (status >= 0))[0]

R = 40.0
nbrs = {}
ring = np.zeros(len(polys)); ring_n = np.zeros(len(polys))
ctree = STRtree([shapely.Point(c) for c in cents])
for v in targets:
    idx = tree.query(polys[v].buffer(R), predicate="intersects")
    idx = idx[(idx != v) & (status[idx] == 1)]          # destroyed neighbors only
    gaps = shapely.distance(polys[v], [polys[i] for i in idx]) if len(idx) else np.array([])
    dx = cents[v, 0] - cents[idx, 0]; dy = cents[v, 1] - cents[idx, 1]
    brg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360  # direction of travel u->v
    nbrs[v] = (idx, np.asarray(gaps), brg, areas[idx])
    # ring 50-250 m: fraction of inspected buildings burned
    ri = ctree.query(shapely.Point(cents[v]).buffer(250))
    dd = np.hypot(*(cents[ri] - cents[v]).T)
    ri = ri[(dd > 50) & (status[ri] >= 0)]
    ring_n[v] = len(ri); ring[v] = status[ri].mean() if len(ri) else np.nan

pickle.dump(dict(heights=np.array(hts,dtype=float), cents=cents, status=status, fire=fire, areas=areas, targets=targets, nbrs=nbrs, ring=ring, ring_n=ring_n,
                 stype=stype), open("data/la/calib.pkl", "wb"))

# --- non-parametric look: burn rate vs nearest destroyed neighbor gap, stratified by ring burn fraction
y = status[targets]; rg = ring[targets]
ng = np.array([nbrs[v][1].min() if len(nbrs[v][1]) else 99 for v in targets])
print("targets", len(targets), "burned frac", y.mean().round(3))
bins = [0, 2, 4, 6, 8, 10, 14, 20, 30, 41, 100]
for lo, hi in [(0, .2), (.2, .5), (.5, .8), (.8, 1.01)]:
    m = (rg >= lo) & (rg < hi)
    row = []
    for a, b in zip(bins[:-1], bins[1:]):
        mm = m & (ng >= a) & (ng < b)
        row.append(f"{y[mm].mean():.2f}({mm.sum()})" if mm.sum() >= 15 else "   -    ")
    print(f"ring {lo:.1f}-{hi:.1f}: " + " ".join(row))
print("gap bins:", bins)
