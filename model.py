"""
Urban fire spread when municipal fire suppression is unavailable.

Model
-----
* Buildings (OpenStreetMap footprints) are nodes.  Fire passes u -> v with probability
  p(u->v) = 1 - (1 - p_flame)(1 - p_ember), sampled independently per fire.
    - p_flame: radiant / direct-flame spread, logistic in the edge-to-edge gap, isotropic.
      Fitted to the 2025 Eaton and Palisades fires (calib_prep.py, calib_fit2.py).
    - p_ember: spotting, downwind only (+/-35 degrees), decaying exponentially with distance, up to 250 m.
      Not fitted; see README.
* Each day has a weather type (weather.py) that sets a multiplier m on the radiant distances and the
  ember strength, and a wind direction drawn from that type's observed wind rose.
* For a target building we sample the set of buildings whose fire would reach it (reverse breadth-first
  search over lazily sampled edges) and record W, their total footprint in units of 100 m2.
* Escaped ignitions (fires that grow to involve the whole building) occur at rate q0 per 100 m2 of footprint
  per day.  The target survives a day with probability (1 - q0*a_t*own) * exp(-q0*W), so results can be
  recomputed for any q0 and duration from the stored W samples (report.py).
* Defense levels modify the edges INTO a defended building and its own escaped-ignition rate.
"""
import json, math
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, Point
from shapely import STRtree
import shapely

R_FLAME = 30.0    # m, largest gap considered for radiant/flame spread
R_EMBER = 250.0   # m, largest distance for ember spotting

# Radiant curve fitted on LA 2025 (calib_fit2.py):
#   p_flame(gap) = sigmoid(-(gap - d50)/s), d50 = 4.34 m + 0.61 m per doubling of source footprint over 120 m2, s = 3.97 m
D50_LA, S_LA, AREA_COEF = 4.34, 3.97, 0.61
SPREAD_SCALE = 1.0   # sensitivity knob: multiplies d50 and s on every day type
M_FLOOR = None       # sensitivity knob: if set, every day type uses at least this multiplier m

# Day types: (name, m, ember e0, ember length scale L in m).  The classes are defined on observed weather in
# weather.py, which also supplies their frequencies and wind roses per station.  m, e0 and L are judgment
# calls: the red-flag row reproduces the 1908 Chelsea conflagration on today's footprint (hindcast_sweep.py)
# and the first two rows give Kobe-1995-like fire sizes in dense fabric (kobe_check.py).
DAY_TYPES = [
    ("calm/damp",   0.8, 0.000, 40),
    ("ordinary",    0.9, 0.002, 40),
    ("windy & dry", 1.0, 0.010, 60),
    ("red-flag",    1.0, 0.030, 90),
]
WIND_DIRS = np.array([0, 45, 90, 135, 180, 225, 270, 315])   # wind-from bins used by the wind roses

# Defense levels.  (flame multiplier at gap < 3 m, flame multiplier at gap >= 8 m (interpolated between),
# ember multiplier, own escaped-ignition multiplier).  Judgment calls; see README.
DEFENSE = {
    "do nothing":            (1.00, 1.00, 1.00, 1.00),
    "attentive, no prep":    (0.95, 0.80, 0.45, 0.55),
    "attentive + prepared":  (0.85, 0.45, 0.12, 0.25),
}

# Buildings treated as less ignitable (susceptibility 0.35): by tag, by size, by material or by height.
NONCOMB_TAGS = {"commercial", "industrial", "retail", "school", "university", "college", "hospital",
                "church", "civic", "public", "office", "warehouse", "parking", "government", "train_station",
                "fire_station", "supermarket", "dormitory", "hotel"}
NONCOMB_SUSCEPTIBILITY = 0.35


def load(name):
    """OSM building elements saved by osm.py"""
    return json.load(open(f"data/osm_{name}.json"))["elements"]


def build(elements, lat0, lon0):
    """-> (polygons in local metres centred on lat0/lon0, per-building info dicts)"""
    tf = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +units=m", always_xy=True)
    polys, info = [], []
    for e in elements:
        if e["type"] != "way" or "geometry" not in e:   # multipolygon relations are skipped (rare)
            continue
        xs, ys = tf.transform([p["lon"] for p in e["geometry"]], [p["lat"] for p in e["geometry"]])
        if len(xs) < 4: continue
        pg = Polygon(zip(xs, ys))
        if not pg.is_valid: pg = pg.buffer(0)
        if pg.is_empty or pg.area < 8: continue
        t = e.get("tags", {})
        b = t.get("building", "yes")
        if b in ("roof", "carport", "ruins", "construction"):   # open structures
            continue
        area = pg.area
        try: lv = float(t.get("building:levels"))
        except (TypeError, ValueError): lv = None
        mat = t.get("building:material", "")
        noncomb = (b in NONCOMB_TAGS or area > 900 or mat in ("brick", "concrete", "stone", "granite", "steel")
                   or (lv is not None and lv >= 5))
        info.append(dict(id=e["id"], area=area, noncomb=noncomb, tag=b))
        polys.append(pg)
    return polys, info


def find_target(polys, info, osm_way=None, max_dist=25.0):
    """index of the site's building.  With osm_way, the footprint with that OpenStreetMap way id.  Otherwise
    the footprint at the site's own coordinates (the local origin): the one containing the point, else the
    nearest within max_dist metres.  Geocoded addresses often land on the street or the lot rather than the
    building, so check the printed description or pass osm_way."""
    if osm_way is not None:
        for i, x in enumerate(info):
            if x["id"] == osm_way: return i
        raise SystemExit(f"OSM way {osm_way} is not among the buildings in the window")
    d = shapely.distance(Point(0, 0), polys)
    i = int(np.argmin(d))
    if d[i] > max_dist:
        raise SystemExit(f"no building within {max_dist} m of the site coordinates (nearest is {d[i]:.0f} m away); "
                         "check lat/lon, or that OpenStreetMap has the building")
    return i


def neighbor_lists(polys):
    """for each building v: candidate sources u within R_EMBER, with edge-to-edge gap and bearing u->v"""
    tree = STRtree(polys)
    cents = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    nb = [None] * len(polys)
    for v in range(len(polys)):
        idx = tree.query(polys[v].buffer(R_EMBER), predicate="intersects")
        idx = idx[idx != v]
        gaps = shapely.distance(polys[v], [polys[i] for i in idx])
        dx = cents[v, 0] - cents[idx, 0]; dy = cents[v, 1] - cents[idx, 1]
        brg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360      # compass bearing of travel u->v
        nb[v] = (idx.astype(np.int32), gaps.astype(np.float32), brg.astype(np.float32))
    return nb, cents


def edge_probs(gaps, brg, src_area, dst_noncomb, wind_from, m, e0, L):
    """-> (p_flame, p_ember) for edges into one destination building"""
    toward = (wind_from + 180) % 360               # direction the wind blows toward
    cosang = np.cos(np.radians(brg - toward))      # +1 = spread is straight downwind
    sc = m * SPREAD_SCALE
    d50 = (D50_LA + AREA_COEF * np.clip(np.log2(src_area / 120.0), -2, 2)) * sc
    p_flame = 1.0 / (1.0 + np.exp((gaps - d50) / (S_LA * sc)))
    p_flame = np.where(gaps <= R_FLAME, p_flame, 0.0)
    p_ember = np.where(cosang > math.cos(math.radians(35)), e0 * np.exp(-gaps / L), 0.0)
    sus = NONCOMB_SUSCEPTIBILITY if dst_noncomb else 1.0
    return p_flame * sus, p_ember * sus


_AREAS = {}
def sample_W(target, nb, info, cents, rng, day, wind_from, defense, boundary_r, block=None):
    """One sample of the reverse reach set.  Returns (W, hit_boundary, n):
      W = footprint/100 m2 of the buildings whose fire would reach the target, hit_boundary = whether that
      set reaches the edge of the data window, n = number of buildings in it.
    block: optional set of building indices (besides the target) at the same defense level: their incoming
      edges get the defense multipliers and their own escaped-ignition rate is scaled by `own`."""
    _, m, e0, L = day
    if M_FLOOR is not None: m = max(m, M_FLOOR)
    if id(info) not in _AREAS: _AREAS.clear(); _AREAS[id(info)] = np.array([x["area"] for x in info])
    areas = _AREAS[id(info)]
    seen = {target}
    frontier = [target]
    W = 0.0
    hit = False
    fm_near, fm_far, em, own = defense
    while frontier:
        v = frontier.pop()
        idx, gaps, brg = nb[v]
        if len(idx) == 0: continue
        pf, pe = edge_probs(gaps, brg, areas[idx], info[v]["noncomb"], wind_from, m, e0, L)
        if v == target or (block is not None and v in block):
            fm = np.interp(gaps, [3, 8], [fm_near, fm_far])
            pf = pf * fm; pe = pe * em
        p = 1 - (1 - pf) * (1 - pe)
        open_ = rng.random(len(idx)) < p
        for u in idx[open_]:
            if u not in seen:
                seen.add(u); frontier.append(u)
                W += areas[u] / 100.0 * (own if (block is not None and u in block) else 1.0)
                if math.hypot(*cents[u]) > boundary_r: hit = True
    return W, hit, len(seen) - 1
