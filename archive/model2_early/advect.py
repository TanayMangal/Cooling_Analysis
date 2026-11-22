"""
Semi-Lagrangian advection on a cell-centered grid, written as sparse matrices so the
scalar update stays linear (T_new = S @ T + s_bc) and the transfer-matrix method still works.

Three schemes:
  'fwd_idw'  - the team's scheme: every cell-center point moves forward p_future = p + u*dt;
               each (fixed) cell center then takes the inverse-distance-weighted average of
               the moved points within radius r_c. Cells no point lands near keep their value.
  'bwd_idw'  - standard backward semi-Lagrangian: trace each center back, x_d = x - u*dt,
               and interpolate the old field at x_d by inverse-distance weighting of the 8
               surrounding centers.
  'bwd_tri'  - same backtrace, trilinear interpolation (weights linear in distance per axis).

Walls are thin axis-aligned rectangles. Back-traced paths are clipped at walls, interpolation
stencils never reach across a wall, and a path that leaves through an inflow patch (domain
inflow face or a cooler outlet) takes that patch's boundary value instead.
"""
import numpy as np
import scipy.sparse as sp


class Walls:
    """List of thin rectangles: (axis, coord, (lo1, hi1), (lo2, hi2)) where the other two
    axes in increasing order are bounded by lo/hi. 'patches' are open sub-rectangles of a
    wall through which supply air enters, with a boundary-value key."""

    def __init__(self, rects=(), patches=()):
        self.rects = list(rects)
        self.patches = list(patches)        # (axis, coord, (lo1,hi1), (lo2,hi2), key)

    @staticmethod
    def _hit(a, b, axis, c, r1, r2):
        """Segments a->b (n,3). Returns t in (0,1] of crossing plane x_axis=c within the
        rectangle, else inf."""
        o = [i for i in range(3) if i != axis]
        da, db = a[:, axis] - c, b[:, axis] - c
        cross = (da * db < 0) | ((db == 0) & (da != 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(cross, da / (da - db), np.inf)
        p = a + np.clip(t, 0, 1)[:, None] * (b - a)
        inside = ((p[:, o[0]] >= r1[0]) & (p[:, o[0]] <= r1[1]) &
                  (p[:, o[1]] >= r2[0]) & (p[:, o[1]] <= r2[1]))
        return np.where(cross & inside, t, np.inf)

    def first_hit(self, a, b):
        """Earliest wall crossing on each segment: (t, patch_key or None)."""
        t = np.full(len(a), np.inf); key = np.full(len(a), -1)
        for (ax, c, r1, r2) in self.rects:
            th = self._hit(a, b, ax, c, r1, r2)
            better = th < t; t[better] = th[better]; key[better] = -1
        for k, (ax, c, r1, r2, _) in enumerate(self.patches):
            th = self._hit(a, b, ax, c, r1, r2)
            better = th <= t; t[better] = th[better]; key[better] = k
        return t, key

    def blocked(self, a, b):
        t, _ = self.first_hit(a, b)
        return np.isfinite(t)


class Grid:
    def __init__(self, origin, d, N, walls=None, inflow=None):
        self.o = np.asarray(origin, float); self.d = np.asarray(d, float)
        self.N = np.asarray(N); self.n = int(np.prod(N))
        g = np.meshgrid(*[self.o[i] + (np.arange(N[i]) + 0.5) * self.d[i] for i in range(3)],
                        indexing='ij')
        self.X = np.stack([a.ravel() for a in g], 1)       # (n,3) cell centers
        self.walls = walls or Walls()
        self.lo = self.o; self.hi = self.o + self.d * self.N
        # inflow: list of (axis, side 0/1) domain faces whose value is the far-field value
        self.inflow = inflow or []

    def idx(self, i, j, k):
        return (i * self.N[1] + j) * self.N[2] + k


def _stencil_8(G, P):
    """8 surrounding cell centers of points P (n,3). Returns idx (n,8), centers (n,8,3),
    trilinear weights (n,8)."""
    f = (P - G.o) / G.d - 0.5
    i0 = np.floor(f).astype(int)
    fr = f - i0
    idx = np.zeros((len(P), 8), int); w = np.zeros((len(P), 8)); C = np.zeros((len(P), 8, 3))
    m = 0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                ii = np.clip(i0 + np.array([a, b, c]), 0, G.N - 1)
                idx[:, m] = G.idx(ii[:, 0], ii[:, 1], ii[:, 2])
                C[:, m] = G.o + (ii + 0.5) * G.d
                w[:, m] = ((fr[:, 0] if a else 1 - fr[:, 0]) * (fr[:, 1] if b else 1 - fr[:, 1])
                           * (fr[:, 2] if c else 1 - fr[:, 2]))
                m += 1
    return idx, C, w


def backtrace(G, U, dt, n_sub=2):
    """Departure points with walls: midpoint (RK2) backtrace, clipped at walls.
    Returns x_d (n,3), source key per point: -1 interior, -2 far-field inflow,
    k>=0 wall patch k."""
    X = G.X
    Ucen = U
    # RK2: velocity at the half-way point (trilinear interp of U)
    Xm = X - 0.5 * dt * Ucen
    idx, _, w = _stencil_8(G, np.clip(Xm, G.lo, G.hi))
    Um = np.einsum('nm,nmk->nk', w, Ucen[idx])
    Xd = X - dt * Um
    key = np.full(G.n, -1)
    # walls / patches
    t, k = G.walls.first_hit(X, Xd)
    hit = np.isfinite(t)
    Xd[hit] = X[hit] + (np.maximum(t[hit] - 1e-6, 0))[:, None] * (Xd[hit] - X[hit])
    key[hit & (k >= 0)] = k[hit & (k >= 0)]
    # leaving the domain through an inflow face
    for ax, side in G.inflow:
        out = (Xd[:, ax] < G.lo[ax]) if side == 0 else (Xd[:, ax] > G.hi[ax])
        key[out & (key == -1)] = -2
    Xd = np.clip(Xd, G.lo + 1e-9, G.hi - 1e-9)
    return Xd, key


def advection_matrix(G, U, dt, scheme='bwd_tri', r_c=None, p=2.0):
    """Return (S, B) with T_new = S @ T + B @ bc_values, bc_values = [far-field, patch0, patch1...]."""
    n = G.n
    nbc = 1 + len(G.walls.patches)
    if scheme in ('bwd_tri', 'bwd_idw'):
        Xd, key = backtrace(G, U, dt)
        idx, C, w = _stencil_8(G, Xd)
        if scheme == 'bwd_idw':
            dist = np.linalg.norm(C - Xd[:, None, :], axis=2)
            w = 1.0 / np.maximum(dist, 1e-9) ** p
            exact = dist < 1e-9
            w[exact.any(1)] = exact[exact.any(1)].astype(float)
        # no stencil across walls
        A = np.repeat(Xd, 8, axis=0); Bp = C.reshape(-1, 3)
        blk = G.walls.blocked(A, Bp).reshape(n, 8)
        w = np.where(blk, 0.0, w)
        s = w.sum(1, keepdims=True)
        lonely = s[:, 0] == 0
        w[~lonely] /= s[~lonely]
        rows = np.repeat(np.arange(n), 8)
        interior = (key == -1)
        S = sp.csr_matrix(((w * interior[:, None]).ravel(), (rows, idx.ravel())), shape=(n, n))
        S = S + sp.diags((interior & lonely).astype(float))   # fallback: keep own value
        bcol = np.where(key == -2, 0, key + 1)
        B = sp.csr_matrix(((~interior).astype(float), (np.arange(n), np.clip(bcol, 0, nbc - 1))),
                          shape=(n, nbc))
        return S.tocsr(), B.tocsr()
    if scheme == 'fwd_idw':
        # move every center forward (RK2), clip at walls
        X = G.X
        Xm = X + 0.5 * dt * U
        idx, _, w = _stencil_8(G, np.clip(Xm, G.lo, G.hi))
        Um = np.einsum('nm,nmk->nk', w, U[idx])
        Pf = X + dt * Um
        t, _ = G.walls.first_hit(X, Pf)
        hit = np.isfinite(t)
        Pf[hit] = X[hit] + np.maximum(t[hit] - 1e-6, 0)[:, None] * (Pf[hit] - X[hit])
        r_c = r_c if r_c is not None else float(np.max(G.d))
        from scipy.spatial import cKDTree
        tree = cKDTree(Pf / G.d)                          # distances in cell units
        pairs = tree.query_ball_point(X / G.d, r=r_c / np.max(G.d) * 1.0 + 1e-9)
        rows, cols, vals = [], [], []
        Bvals = np.zeros(n); has_pt = np.zeros(n, bool)
        for c, lst in enumerate(pairs):
            if not lst:
                continue
            lst = np.array(lst)
            ok = ~G.walls.blocked(np.repeat(X[c][None], len(lst), 0), Pf[lst])
            lst = lst[ok]
            if len(lst) == 0:
                continue
            dist = np.linalg.norm((Pf[lst] - X[c]) / G.d * np.max(G.d), axis=1)
            if np.any(dist < 1e-9):
                wt = (dist < 1e-9).astype(float)
            else:
                wt = 1.0 / dist ** p
            wt /= wt.sum()
            rows += [c] * len(lst); cols += list(lst); vals += list(wt); has_pt[c] = True
        # cells no moved point reaches: upstream of an inflow -> boundary value, else keep
        S = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        Xd, key = backtrace(G, U, dt)
        empty = ~has_pt
        S = S + sp.diags((empty & (key == -1)).astype(float))
        bcol = np.where(key == -2, 0, key + 1)
        bc_rows = empty & (key != -1)
        B = sp.csr_matrix((bc_rows.astype(float), (np.arange(n), np.clip(bcol, 0, nbc - 1))),
                          shape=(n, nbc))
        return S.tocsr(), B.tocsr()
    raise ValueError(scheme)
