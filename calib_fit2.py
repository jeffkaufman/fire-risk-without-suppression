"""Fit v2: adds source & target height terms; spatial block bootstrap for uncertainty."""
import pickle, numpy as np, sys
from scipy.optimize import minimize
from scipy.special import expit, logit
D = pickle.load(open("data/la/calib.pkl","rb"))
st, fire, targets, nbrs, ring, H, C = D["status"], D["fire"], D["targets"], D["nbrs"], D["ring"], D["heights"], D["cents"]
H = np.where(np.isnan(H), np.nanmedian(H), H)
tg_all = np.array([v for v in targets if not np.isnan(ring[v])])

def build(tg):
    rows, gaps, src = [], [], []
    for j, v in enumerate(tg):
        idx, g, b, a = nbrs[v]
        rows += [j]*len(idx); gaps += list(g); src += list(idx)
    rows=np.array(rows,dtype=int); gaps=np.array(gaps); src=np.array(src,dtype=int)
    return dict(n=len(tg), y=st[tg].astype(float), lr=logit(np.clip(ring[tg],.01,.99)), fi=fire[tg], rows=rows, gaps=gaps,
                lsa=np.clip(np.log2(D["areas"][src]/120),-2,2), lsh=np.clip(np.log2(H[src]/16),-1.5,1.5),
                lth=np.clip(np.log2(H[tg]/16),-1.5,1.5)[rows])
def nll(th, X):
    c0,c1e,c1,d50,ls,a,ah,at = th
    b = expit(np.where(X["fi"]==0,c0,c1e) + c1*X["lr"])
    dd = d50 + a*X["lsa"] + ah*X["lsh"] + at*X["lth"]
    p = expit(-(X["gaps"]-dd)/np.exp(ls))
    ls_ = np.bincount(X["rows"], weights=np.log1p(-np.clip(p,0,1-1e-9)), minlength=X["n"])
    P = np.clip(1-(1-b)*np.exp(ls_),1e-9,1-1e-9)
    return -np.sum(X["y"]*np.log(P)+(1-X["y"])*np.log(1-P))
def fit(X, th0):
    r = minimize(nll, th0, args=(X,), method="L-BFGS-B")
    return r.x, r.fun
X = build(tg_all)
th0 = np.array([-1.7,-2.3,0.86,4.4,np.log(4.0),0.6,0.0,0.0])
th, f = fit(X, th0)
names = ["c0_pal","c0_eat","ring_slope","d50","log_s","a_area","a_srcheight","a_tgtheight"]
print("NLL", round(f,1)); print(dict(zip(names, np.round(th,3))), "s=",round(np.exp(th[4]),2))
# block bootstrap (500 m blocks)
blk = (np.floor(C[tg_all,0]/500).astype(int)*100003 + np.floor(C[tg_all,1]/500).astype(int))
ub = np.unique(blk); rng = np.random.default_rng(0); B=[]
for i in range(40):
    pick = rng.choice(ub, len(ub), replace=True)
    tg = np.concatenate([tg_all[blk==b] for b in pick])
    B.append(fit(build(tg), th)[0])
B = np.array(B)
lo, hi = np.percentile(B, [5,95], axis=0)
for n_, m_, l_, h_ in zip(names, th, lo, hi): print(f"  {n_:12s} {m_:6.2f}  90% CI [{l_:.2f}, {h_:.2f}]")
print(f"  s: {np.exp(th[4]):.2f} [{np.exp(lo[4]):.2f}, {np.exp(hi[4]):.2f}]")
pickle.dump(dict(theta=th, boot=B, names=names), open("data/la/fit2.pkl","wb"))
# implied curves for a 1-story LA house vs a 3-story (~35 ft) New England house, both 120 m2
for lbl, hs in [("1-story src&tgt (16ft)",16),("2.5-story (28ft)",28),("3-story (35ft)",35)]:
    l = np.clip(np.log2(hs/16),-1.5,1.5); dd = th[3]+th[6]*l+th[7]*l; s=np.exp(th[4])
    print(lbl, "d50=%.1f"%dd, " ".join(f"{g}m:{expit(-(g-dd)/s):.2f}" for g in [2,4,6,8,10,15,20]))
