"""Turn the team's CAD (STL) into solid-cell masks on the Model 2 grid. Run once here; the masks are shipped
as .npz so the WPI machine does not need trimesh.
  market_assembly1.stl : Onshape model at 1:10 (sheds 1.65 x 0.61 x 0.51 -> 16.5 x 6.1 x 5.1 m) [scale assumed]
  cricket_stadium_1.stl: scale unknown; x1500 chosen so the stands rise ~18 m and the bowl is ~230 m across
                         (Chepauk-sized) (0.152 x 0.174 x 0.018 -> 228 x 261 x 27 m incl. the base under the field)
A cell is solid if a solid body contains its centre OR any part of a body's surface passes through it
(so thin roofs and posts are not lost; they become one cell thick)."""
import sys, numpy as np, trimesh

SPEC = {
    'market2': dict(stl='cad/market_assembly1.stl', scale=10.0, lo=(-22.0, -22.0, 0.0), hi=(22.0, 22.0, 9.0),
                    keep=lambda e: (e.max() > 3.0 and e.min() > 1.0) or (abs(e.max() - 1.82) < 0.05 and e.min() > 0.5)),
    'stands': dict(stl='cad/cricket_stadium_1.stl', scale=1500.0, lo=(-135.0, -130.0, 0.0), hi=(135.0, 170.0, 60.0),
                   keep=lambda e: True),
}

def voxelize(site, h):
    S = SPEC[site]; m = trimesh.load(S['stl'], force='mesh'); m.apply_scale(S['scale'])
    lo = np.array(S['lo']); hi = np.array(S['hi']); N = np.round((hi - lo) / h).astype(int)
    solid = np.zeros(N, bool)
    ax = [lo[i] + (np.arange(N[i]) + 0.5) * h for i in range(3)]
    bodies = [b for b in m.split(only_watertight=False) if S['keep'](np.sort(b.extents))]
    print(site, 'h', h, 'bodies kept', len(bodies), 'of', len(m.split(only_watertight=False)), 'grid', N)
    X = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    for b in bodies:
        bl, bh = b.bounds
        sel = np.all((X >= bl - h) & (X <= bh + h), 1)
        if b.is_watertight and sel.any():
            inside = np.zeros(len(X), bool); inside[sel] = b.contains(X[sel]); solid |= inside.reshape(N)
        n = int(max(2000, b.area / (h / 4) ** 2))
        pts = b.sample(n)
        ijk = np.floor((pts - lo) / h).astype(int)
        ok = np.all((ijk >= 0) & (ijk < N), 1)
        solid[tuple(ijk[ok].T)] = True
    solid[:, :, :] &= True
    np.savez_compressed(f'cad/{site}_h{h:g}.npz', solid=solid, lo=lo, hi=hi, h=h)
    print('  solid cells', solid.sum(), 'of', solid.size)
    return solid

if __name__ == '__main__':
    todo = (('market2', 0.5), ('market2', 0.25), ('stands', 2.5), ('stands', 2.0))
    for site, h in [t for t in todo if len(sys.argv) < 2 or t[0] == sys.argv[1]]:
        voxelize(site, h)
