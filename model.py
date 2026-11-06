"""
Urban fire spread when municipal fire suppression is unavailable.

Model
-----
* Buildings (OpenStreetMap footprints) are nodes.  Fire passes u -> v with probability
  p(u->v) = 1 - (1 - p_flame)(1 - p_ember), sampled independently per fire.
    - p_flame: short-range spread (radiant heat, flame contact, brands), isotropic, a function of the
      edge-to-edge gap and the source's footprint.  Fitted to the 2025 Eaton and Palisades fires with "any
      damage" as the outcome (calib_fit3.py), then stretched for taller New England houses (radiant_physics.py).
      On days without strong wind the far end of the curve is cut off, because radiant heat alone does not
      ignite beyond about 15 m.
    - p_ember: long-range spotting, downwind only (+/-35 degrees), up to 250 m.  Not fitted; see README.
* Each day has a weather type (weather.py) that sets the distance multiplier m, the weight of the wind-driven
  tail, and the ember strength, and a wind direction drawn from that type's observed wind rose.
* For a target building we sample the set of buildings whose fire would reach it (reverse breadth-first
  search over lazily sampled edges) and record W, their total footprint in units of 100 m2.  On wind-driven
  day types the set is limited to where a fire could travel from during the wind event; otherwise a reach
  set is everything a fire would eventually burn with nobody stopping it.
* Escaped ignitions (fires that grow to involve the whole building) occur at rate q0 per 100 m2 of footprint
  per day.  The target survives a day with probability (1 - q0*a_t*own) * exp(-q0*W), so results can be
  recomputed for any q0 and duration from the stored W samples (report.py).
* Defense levels modify the edges INTO a defended building and its own escaped-ignition rate.
"""
import json, math, os
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, Point
from shapely import STRtree
import shapely

R_FLAME = 40.0    # m, largest gap considered for short-range spread (the range of the LA fit)
R_EMBER = 250.0   # m, largest distance for ember spotting

# Short-range curve fitted on LA 2025, outcome "any damage" (calib_fit3.py):
#   p(gap) = exp(-ln2 * (gap / d50)^K),  d50 = D50_LA * 2^(AREA_COEF * log2(source footprint / 120 m2))
# 90% intervals from a spatial block bootstrap: D50_LA 4.9-5.6 m, K 0.97-1.10, AREA_COEF 0.22-0.34.
D50_LA, K_LA, AREA_COEF = 5.22, 1.03, 0.28
# LA houses are mostly one storey with stucco walls; New England houses are mostly 2-3 storeys with wood or
# vinyl siding.  A radiant-heat calculation (radiant_physics.py) puts the 50% distance 1.25-1.6x further for a
# 2.5-storey house than for a one-storey LA house.  The LA data themselves show no height effect, so this
# factor is a judgment between 1.0 (no adjustment) and 1.6.
HOUSE_FACTOR = 1.3
# Without strong wind, one burning house cannot ignite another by radiant heat beyond roughly 15 m
# (radiant_physics.py), and the LA tail at that range reflects wind-driven brands.  The curve is multiplied by
# tail_w + (1 - tail_w) / (1 + (gap / ENV_D)^ENV_K), where tail_w is the day type's wind weight.
ENV_D, ENV_K = 14.0, 6.0

SPREAD_SCALE = 1.0   # sensitivity knob: multiplies the 50% distance on every day type
M_FLOOR = None       # sensitivity knob: if set, every day type uses at least this distance multiplier m
TAIL_W = {}          # sensitivity knob: {day type: wind weight} overriding DAY_TYPES

# Day types: (name, distance multiplier m, wind weight of the tail, ember e0, ember length scale L in m).
# The classes are defined on observed weather in weather.py, which also supplies their frequencies and wind
# roses per station.  m, the wind weights and the ember numbers are judgment calls; see README.
DAY_TYPES = [
    ("calm/damp",   0.8, 0.0, 0.000, 40),
    ("ordinary",    0.9, 0.1, 0.002, 40),
    ("windy & dry", 1.0, 0.5, 0.010, 60),
    ("red-flag",    1.0, 1.0, 0.030, 90),
]
WIND_DIRS = np.array([0, 45, 90, 135, 180, 225, 270, 315])   # wind-from bins used by the wind roses

# Travel-time limit on wind-driven days.  A wind event lasts hours, and a fire cannot reach the house from
# further than it can travel in that time; afterwards the weather is usually an ordinary day, when fires in
# these neighbourhoods stay within a block or so.  {day type: (downwind spread rate m/h, event hours)}.
# Red-flag rate: Chelsea 1908 covered 2.4 km in 5 h.  Windy & dry: Kobe 1995 measured about 20 m/h of extra
# downwind spread per m/s of wind above 2 m/s (Hokugo 1997), so ~180 m/h at 10 m/s.  Sideways spread is taken
# as a quarter of the downwind rate (the 1908 burn was half as wide as long) and spread against the wind as
# 30 m/h (Kobe's light-wind rate).  Calm and ordinary days have no limit.
EVENT = {"windy & dry": (180.0, 10.0), "red-flag": (480.0, 10.0)}
BACKING_RATE = 30.0
LATERAL_FRACTION = 0.25

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


def graph(polys, info, cache=None):
    """Edge list in CSR form, grouped by destination building v: for each v, the candidate sources u within
    R_EMBER with their edge-to-edge gap and the compass bearing of travel u->v.  Cached in `cache` if given."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if len(z["area"]) == len(polys): return {k: z[k] for k in z.files}
    P = np.array(polys, dtype=object)
    cents = np.array([[p.centroid.x, p.centroid.y] for p in polys])
    tree = STRtree(P)
    dst, src = tree.query(P, predicate="dwithin", distance=R_EMBER)
    k = dst != src; dst, src = dst[k], src[k]
    o = np.argsort(dst, kind="stable"); dst, src = dst[o], src[o]
    gap = np.empty(len(src), dtype=np.float32)
    for a in range(0, len(src), 2_000_000):                       # in chunks to bound memory
        gap[a:a + 2_000_000] = shapely.distance(P[src[a:a + 2_000_000]], P[dst[a:a + 2_000_000]])
    brg = ((np.degrees(np.arctan2(cents[dst, 0] - cents[src, 0], cents[dst, 1] - cents[src, 1])) + 360) % 360).astype(np.float32)
    G = dict(indptr=np.concatenate([[0], np.cumsum(np.bincount(dst, minlength=len(polys)))]).astype(np.int64),
             src=src.astype(np.int32), gap=gap, brg=brg, cents=cents,
             area=np.array([x["area"] for x in info]), noncomb=np.array([x["noncomb"] for x in info]))
    if cache: np.savez(cache, **G)
    return G


def edge_probs(gaps, brg, src_area, dst_noncomb, wind_from, day):
    """-> (p_flame, p_ember) for edges; dst_noncomb may be a bool or an array aligned with the edges"""
    name, m, tail_w, e0, L = day
    if M_FLOOR is not None: m = max(m, M_FLOOR)
    tail_w = TAIL_W.get(name, tail_w)
    d50 = D50_LA * HOUSE_FACTOR * SPREAD_SCALE * m * 2.0 ** (AREA_COEF * np.clip(np.log2(src_area / 120.0), -2, 2))
    g = np.maximum(gaps, 0.05)
    p_flame = np.exp(-math.log(2) * (g / d50) ** K_LA)
    p_flame = p_flame * (tail_w + (1 - tail_w) / (1 + (g / ENV_D) ** ENV_K))
    p_flame = np.where(gaps <= R_FLAME, p_flame, 0.0)
    toward = (wind_from + 180) % 360               # direction the wind blows toward
    cosang = np.cos(np.radians(brg - toward))      # +1 = spread is straight downwind
    p_ember = np.where(cosang > math.cos(math.radians(35)), e0 * np.exp(-gaps / L), 0.0) if e0 > 0 else np.zeros(len(gaps))
    sus = np.where(dst_noncomb, NONCOMB_SUSCEPTIBILITY, 1.0)
    return p_flame * sus, p_ember * sus


def reachable(G, target, day, wind_from):
    """mask of buildings a fire could travel from to the target within the day type's wind event, or None"""
    ev = EVENT.get(day[0])
    if ev is None: return None
    v, hours = ev
    a = math.radians(wind_from + 180); ux, uy = math.sin(a), math.cos(a)      # unit vector the wind blows toward
    d = G["cents"] - G["cents"][target]
    upwind = -(d[:, 0] * ux + d[:, 1] * uy)                                  # > 0: the source is upwind of the target
    cross = np.abs(d[:, 0] * uy - d[:, 1] * ux)
    t = np.where(upwind > 0, upwind / v, -upwind / BACKING_RATE) + cross / max(BACKING_RATE, LATERAL_FRACTION * v)
    return t <= hours


def sample_W(target, G, rng, day, wind_from, defense, boundary_r, block=None, wild=None):
    """One sample of the reverse reach set.  Returns (W, hit_boundary, n, X):
      W = footprint/100 m2 of the buildings whose fire would reach the target, hit_boundary = whether that
      set reaches the edge of the data window, n = number of buildings in it, X = wildland exposure (below).
    wild: optional (patch index per building or -1, ignition probability if that forest patch burns), from
      wildland.exposure.  X = sum over forest patches of the probability that, if the patch burned today, it
      would ignite the target or a building in its reach set.
    block: optional boolean mask of buildings (besides the target) at the same defense level: their incoming
      edges get the defense multipliers and their own escaped-ignition rate is scaled by `own`."""
    indptr, src, gap, brg, area, noncomb = G["indptr"], G["src"], G["gap"], G["brg"], G["area"], G["noncomb"]
    near_only = day[3] == 0                       # no embers: only edges within R_FLAME matter
    fm_near, fm_far, em, own = defense
    defended = np.zeros(len(area), dtype=bool) if block is None else block.copy()
    defended[target] = True
    seen = np.zeros(len(area), dtype=bool); seen[target] = True
    frontier = np.array([target])
    W = 0.0; hit = False; n = 0
    r = np.hypot(*G["cents"].T)
    allowed = reachable(G, target, day, wind_from)
    while len(frontier):
        starts = indptr[frontier]; lens = indptr[frontier + 1] - starts
        tot = int(lens.sum())
        if tot == 0: break
        e = np.repeat(starts - (np.cumsum(lens) - lens), lens) + np.arange(tot)   # indices of all edges into the frontier
        v = np.repeat(frontier, lens)
        keep = ~seen[src[e]]
        if near_only: keep &= gap[e] <= R_FLAME
        if allowed is not None: keep &= allowed[src[e]]
        e, v = e[keep], v[keep]
        if len(e) == 0: break
        u = src[e]
        pf, pe = edge_probs(gap[e], brg[e], area[u], noncomb[v], wind_from, day)
        d = defended[v]
        if d.any():
            pf = np.where(d, pf * np.interp(gap[e], [3, 8], [fm_near, fm_far]), pf); pe = np.where(d, pe * em, pe)
        new = np.unique(u[rng.random(len(e)) < 1 - (1 - pf) * (1 - pe)])
        seen[new] = True
        W += float((area[new] / 100.0 * (np.where(block[new], own, 1.0) if block is not None else 1.0)).sum())
        hit = hit or bool((r[new] > boundary_r).any())
        n += len(new)
        frontier = new
    X = 0.0
    if wild is not None:
        patch, pw = wild
        m = seen & (patch >= 0)
        if m.any():
            pw_eff = np.where(defended[m], pw[m] * em, pw[m])
            X = float((1 - np.exp(np.bincount(patch[m], weights=np.log1p(-pw_eff)))).sum())
    return W, hit, n, X
