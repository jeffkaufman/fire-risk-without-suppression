"""Chelsea 1908 hindcast on the present-day footprint, and what ember strength / spread scale reproduces it.
History: WNW gale (40-45 mph); essentially every building between the railroad and the waterfront burned,
from the Everett line to Chelsea Creek; one building lost north of the tracks.
Tests: (a) single ignition inside today's dense fabric; (b) a fire front arriving at its upwind edge."""
import sys, numpy as np, hindcast_chelsea as H
G = H.setup(); district, north = H.zones(G)
c = G["cents"]; res = ~G["noncomb"]
# single ignitions inside the dense fabric (upwind third), and a "front" = every district building west of x=-250
PTS = [(-150, 300), (-100, 150), (0, 350), (-200, 50), (50, 250), (-50, 0), (100, 150), (-150, -150)]
seeds = [int(np.where(res & district)[0][np.argmin(np.hypot(*(c[res & district] - p).T))]) for p in PTS]
front = np.where(district & (c[:, 0] < -250))[0]
down = district & (c[:, 0] > -150)          # fabric the 1908 front then burned through to the creek
print("district", district.sum(), "front seeds", len(front), "downwind fabric", down.sum())
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
rng = np.random.default_rng(1)
print("scenario (m, e0, L, scale)                 single ignition: P(>=10) P(>=100) P(dist>50%) | front arriving: mean downwind fabric burned, P(>50%), north-of-tracks burned")
for name, (m, e0, L, sc) in {
    "as-is red-flag":      (1.0, 0.03, 90, 1.0),
    "e0 x2":               (1.0, 0.06, 90, 1.0),
    "e0 x4":               (1.0, 0.12, 90, 1.0),
    "L 150":               (1.0, 0.03, 150, 1.0),
    "scale 0.8":           (1.0, 0.03, 90, 0.8),
    "scale 1.25":          (1.0, 0.03, 90, 1.25),
    "scale 1.5":           (1.0, 0.03, 90, 1.5),
    "scale 1.5 no embers": (1.0, 0.00, 90, 1.5),
    "scale 2.0 no embers": (1.0, 0.00, 90, 2.0),
    "windy&dry as-is":     (1.0, 0.01, 60, 1.0),
    "windy&dry scale 1.5": (1.0, 0.01, 60, 1.5),
    "calm scale 1.5 (m.8)":(0.8, 0.00, 40, 1.5),
    "calm scale 2.0 (m.8)":(0.8, 0.00, 40, 2.0),
}.items():
    r = []; fr = []
    for i in range(N):
        p = H.edge_p(G, H.WIND_FROM + rng.uniform(-15, 15), m, e0, L, sc)
        y = H.burn(G, p, [seeds[i % len(seeds)]], rng)
        r.append((y.sum(), y[district].mean()))
        if i < N // 3:
            y = H.burn(G, p, front, rng); fr.append((y[down].mean(), y[north].mean()))
    r = np.array(r); fr = np.array(fr)
    print(f"{name:22s} {str((m, e0, L, sc)):24s} {np.mean(r[:,0]>=10):8.2f} {np.mean(r[:,0]>=100):8.2f} {np.mean(r[:,1]>.5):8.2f}   | {fr[:,0].mean():8.2f} {np.mean(fr[:,0]>.5):8.2f} {fr[:,1].mean():8.2f}", flush=True)
