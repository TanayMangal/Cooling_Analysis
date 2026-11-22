"""
Stable-Fluids (Stam 1999) wind solver around and inside the bus shelter.

Extended domain (outside air around the shelter) on a MAC grid:
  normal velocities live on cell faces, pressure at cell centers.
Each step:  advect (semi-Lagrangian) -> eddy-viscosity diffusion (implicit)
            -> impose inflow / jets / walls -> pressure projection (div u = 0).
Thin walls (back, sides, roof) block faces; the ground blocks the bottom face.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from advect import Grid, Walls, advection_matrix

# shelter geometry (same as Model 1) inside an extended domain
SH = dict(x0=0.0, x1=3.0, y0=0.0, y1=1.5, H=2.5)
DOMAIN = dict(origin=(-1.5, -0.6, 0.0), d=(0.3, 0.15, 0.25), N=(20, 24, 14))
UTCI_V10 = np.log(10 / 0.01) / np.log(1.1 / 0.01)      # 1.1 m -> 10 m wind factor (UTCI convention)


def shelter_walls(patches=()):
    s = SH
    rects = [(1, s['y0'], (s['x0'], s['x1']), (0, s['H'])),          # back wall  (y = y0)
             (0, s['x0'], (s['y0'], s['y1']), (0, s['H'])),          # left side  (x = x0)
             (0, s['x1'], (s['y0'], s['y1']), (0, s['H'])),          # right side (x = x1)
             (2, s['H'], (s['x0'], s['x1']), (s['y0'], s['y1']))]    # roof       (z = H)
    return Walls(rects, patches)


class Domain:
    def __init__(self, inflow_dir=None, U10=0.0, z0=0.1, nu_t=0.02, jets=()):
        """inflow_dir: 'front' (wind blowing into the open front), 'along' (parallel to
        the front, +x), 'behind' (onto the back wall), or None (calm).
        jets: list of dict(kind='supply'|'fan', x=(lo,hi), z=(lo,hi), V=m/s)."""
        o, d, N = DOMAIN['origin'], DOMAIN['d'], DOMAIN['N']
        self.d = np.array(d); self.N = np.array(N); self.o = np.array(o)
        self.nu_t, self.U10, self.z0 = nu_t, U10, z0
        self.jets = list(jets)
        patches = [(1, SH['y0'], j['x'], j['z'], i) for i, j in enumerate(self.jets)
                   if j['kind'] == 'supply']
        self.patch_jet = [i for i, j in enumerate(self.jets) if j['kind'] == 'supply']
        # every outer face except the ground: air entering from outside carries far-field values
        inflow = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)]
        self.inflow_dir = inflow_dir
        self.G = Grid(o, d, N, shelter_walls(patches), inflow)
        Nx, Ny, Nz = N
        xf = o[0] + np.arange(Nx + 1) * d[0]; yf = o[1] + np.arange(Ny + 1) * d[1]
        zf = o[2] + np.arange(Nz + 1) * d[2]
        xc = o[0] + (np.arange(Nx) + .5) * d[0]; yc = o[1] + (np.arange(Ny) + .5) * d[1]
        zc = o[2] + (np.arange(Nz) + .5) * d[2]
        self.xc, self.yc, self.zc = xc, yc, zc
        # ---- face masks: 0 open (solved), 1 blocked (u=0), 2 fixed value
        self.mu = np.zeros((Nx + 1, Ny, Nz), int); self.mv = np.zeros((Nx, Ny + 1, Nz), int)
        self.mw = np.zeros((Nx, Ny, Nz + 1), int)
        self.fu = np.zeros((Nx + 1, Ny, Nz)); self.fv = np.zeros((Nx, Ny + 1, Nz))
        self.fw = np.zeros((Nx, Ny, Nz + 1))
        inX = lambda a, b: (xc >= a) & (xc <= b)
        inY = lambda a, b: (yc >= a) & (yc <= b)
        inZ = lambda a, b: (zc >= a) & (zc <= b)
        s = SH
        iw0 = int(round((s['x0'] - o[0]) / d[0])); iw1 = int(round((s['x1'] - o[0]) / d[0]))
        jb = int(round((s['y0'] - o[1]) / d[1])); kr = int(round((s['H'] - o[2]) / d[2]))
        self.ish = (iw0, iw1); self.jsh = (jb, int(round((s['y1'] - o[1]) / d[1]))); self.ksh = kr
        Ysh, Zsh = inY(s['y0'], s['y1']), inZ(0, s['H'])
        for i in (iw0, iw1):
            self.mu[i][np.ix_(Ysh, Zsh)] = 1
        self.mv[:, jb, :][np.ix_(inX(s['x0'], s['x1']), Zsh)] = 1
        self.mw[:, :, kr][np.ix_(inX(s['x0'], s['x1']), Ysh)] = 1
        self.mw[:, :, 0] = 1                                           # ground
        # ---- inflow profile (log law)
        prof = lambda z: U10 * np.log(np.maximum(z, z0) / z0) / np.log(10 / z0)
        self.prof = prof
        self.U_far = np.zeros(3)
        if inflow_dir == 'front':
            self.mv[:, Ny, :] = 2; self.fv[:, Ny, :] = -prof(zc)[None, :]
        elif inflow_dir == 'along':
            self.mu[0, :, :] = 2; self.fu[0, :, :] = prof(zc)[None, :]
        elif inflow_dir == 'behind':
            self.mv[:, 0, :] = 2; self.fv[:, 0, :] = prof(zc)[None, :]
        # ---- jets: supply = air enters through the back wall; fan = interior face 1 cell in
        for jt in self.jets:
            X, Z = inX(*jt['x']), inZ(*jt['z'])
            jj = jb if jt['kind'] == 'supply' else jb + 1
            sub = self.mv[:, jj, :]; sub[np.ix_(X, Z)] = 2; self.mv[:, jj, :] = sub
            sub = self.fv[:, jj, :]; sub[np.ix_(X, Z)] = jt['V']; self.fv[:, jj, :] = sub
            jt['area'] = X.sum() * d[0] * Z.sum() * d[2]
        self.u = np.where(self.mu == 2, self.fu, 0.0); self.v = np.where(self.mv == 2, self.fv, 0.0)
        self.w = np.where(self.mw == 2, self.fw, 0.0)
        self._build_poisson(); self._build_visc_cache = {}

    # ------------------------------------------------------------- pressure projection
    def _build_poisson(self):
        Nx, Ny, Nz = self.N; d = self.d
        n = Nx * Ny * Nz
        I = lambda i, j, k: (i * Ny + j) * Nz + k
        rows, cols, vals = [], [], []; diag = np.zeros(n)
        def faces(mask, axis):
            sh = list(mask.shape)
            for idx in np.argwhere(mask == 0):
                a = idx.copy(); a[axis] -= 1; b = idx.copy()
                lo_in = a[axis] >= 0; hi_in = b[axis] < self.N[axis]
                c = 1.0 / d[axis] ** 2
                if lo_in and hi_in:
                    p, q = I(*a), I(*b)
                    rows.extend([p, q]); cols.extend([q, p]); vals.extend([-c, -c])
                    diag[p] += c; diag[q] += c
                elif lo_in:                       # open boundary face, p_ghost = 0
                    diag[I(*a)] += 2 * c
                elif hi_in:
                    diag[I(*b)] += 2 * c
        faces(self.mu, 0); faces(self.mv, 1); faces(self.mw, 2)
        diag = np.where(diag == 0, 1.0, diag)
        rows += list(range(n)); cols += list(range(n)); vals += list(diag)
        self.Lp = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
        self.Lp_lu = spla.splu(self.Lp)

    def divergence(self, u=None, v=None, w=None):
        u = self.u if u is None else u; v = self.v if v is None else v; w = self.w if w is None else w
        return ((u[1:] - u[:-1]) / self.d[0] + (v[:, 1:] - v[:, :-1]) / self.d[1]
                + (w[:, :, 1:] - w[:, :, :-1]) / self.d[2])

    def project(self):
        div = self.divergence().ravel()
        phi = self.Lp_lu.solve(div).reshape(self.N)       # phi = dt*p/rho
        d = self.d
        pad = lambda a, ax: np.concatenate([np.take(a, [0], ax) * 0 - np.take(a, [0], ax),
                                            a, -np.take(a, [-1], ax)], ax)
        # gradient on faces, ghost = -phi (so boundary face value sits at p=0)
        gx = np.diff(pad(phi, 0), axis=0) / d[0]
        gy = np.diff(pad(phi, 1), axis=1) / d[1]
        gz = np.diff(pad(phi, 2), axis=2) / d[2]
        self.u = np.where(self.mu == 0, self.u + gx, self.u)
        self.v = np.where(self.mv == 0, self.v + gy, self.v)
        self.w = np.where(self.mw == 0, self.w + gz, self.w)
        self._enforce()

    def _enforce(self):
        for m, f, a in ((self.mu, self.fu, 'u'), (self.mv, self.fv, 'v'), (self.mw, self.fw, 'w')):
            arr = getattr(self, a)
            arr = np.where(m == 1, 0.0, np.where(m == 2, f, arr)); setattr(self, a, arr)

    # ------------------------------------------------------------- helpers
    def centers(self):
        U = np.stack([0.5 * (self.u[1:] + self.u[:-1]), 0.5 * (self.v[:, 1:] + self.v[:, :-1]),
                      0.5 * (self.w[:, :, 1:] + self.w[:, :, :-1])], -1)
        return U.reshape(-1, 3)

    def to_faces(self, Uc):
        U = Uc.reshape(*self.N, 3)
        u = np.empty_like(self.u); v = np.empty_like(self.v); w = np.empty_like(self.w)
        u[1:-1] = 0.5 * (U[1:, :, :, 0] + U[:-1, :, :, 0]); u[0] = U[0, :, :, 0]; u[-1] = U[-1, :, :, 0]
        v[:, 1:-1] = 0.5 * (U[:, 1:, :, 1] + U[:, :-1, :, 1]); v[:, 0] = U[:, 0, :, 1]; v[:, -1] = U[:, -1, :, 1]
        w[:, :, 1:-1] = 0.5 * (U[:, :, 1:, 2] + U[:, :, :-1, 2]); w[:, :, 0] = 0; w[:, :, -1] = U[:, :, -1, 2]
        self.u, self.v, self.w = u, v, w
        self._enforce()

    def _visc(self, dt):
        if dt not in self._build_visc_cache:
            from model2 import diffusion_operator
            A, _ = diffusion_operator(self, kind='momentum')
            M = sp.identity(self.G.n) / dt
            self._build_visc_cache[dt] = spla.splu((M + self.nu_t * A).tocsc())
        return self._build_visc_cache[dt]

    def step(self, dt):
        Uc = self.centers()
        S, B = advection_matrix(self.G, Uc, dt, 'bwd_tri')
        bc = np.zeros((B.shape[1], 3)); bc[0] = self.far_velocity()
        for k, jid in enumerate(self.patch_jet):
            bc[k + 1] = [0, self.jets[jid]['V'], 0]
        Ua = S @ Uc + B @ bc
        lu = self._visc(dt)
        Ud = np.stack([lu.solve(Ua[:, c] / dt) for c in range(3)], 1)
        self.to_faces(Ud)
        self.project()

    def far_velocity(self):
        U = self.U10 * 0.55                                 # representative far-field value
        return {'front': [0, -U, 0], 'along': [U, 0, 0], 'behind': [0, U, 0], None: [0, 0, 0]}[self.inflow_dir]

    def solve_steady(self, dt=0.05, t_end=30.0, tol=1e-2, verbose=False):
        prev = self.centers()
        for n in range(int(t_end / dt)):
            self.step(dt)
            if n % 50 == 49:
                cur = self.centers(); ch = np.abs(cur - prev).max(); prev = cur
                if verbose:
                    print(f't={dt*(n+1):5.1f}s  max change/2.5s={ch:.4f}  max|div|={np.abs(self.divergence()).max():.2e}')
                if ch < tol:
                    break
        return self

    def interior_mask(self):
        X = self.G.X
        return ((X[:, 0] > SH['x0']) & (X[:, 0] < SH['x1']) & (X[:, 1] > SH['y0'])
                & (X[:, 1] < SH['y1']) & (X[:, 2] < SH['H']))
