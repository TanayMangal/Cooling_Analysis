"""Compiled (numba) kernels for Model 2's semi-Lagrangian steps.

backward_gather : standard backward semi-Lagrangian. Each cell center is traced back
                  along the wind (substeps < h/2, stopped at blocked faces); fields are
                  read there by trilinear (distance-based) interpolation that ignores
                  cells across a wall. Leaving through an open outer face -> far field.
forward_scatter : the team's forward scheme. Each cell center (and a layer of ghost
                  outside-air points) moves forward to p_future; each FIXED center then
                  takes the weighted average of the moved points, weight = tent function
                  of distance (1 - |dx|/h)(1 - |dy|/h)(1 - |dz|/h), zero beyond one cell.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, inline='always')
def _cell(p, lo, h, N):
    i = int(np.floor((p[0] - lo[0]) / h)); j = int(np.floor((p[1] - lo[1]) / h)); k = int(np.floor((p[2] - lo[2]) / h))
    i = min(max(i, 0), N[0] - 1); j = min(max(j, 0), N[1] - 1); k = min(max(k, 0), N[2] - 1)
    return i, j, k


@njit(cache=True)
def _weights(p, lo, h, N, tab, idx, w):
    """8 corner indices/weights (wall-aware) for point p. Returns number of valid weights>0."""
    fx = (p[0] - lo[0]) / h - 0.5; fy = (p[1] - lo[1]) / h - 0.5; fz = (p[2] - lo[2]) / h - 0.5
    i0 = int(np.floor(fx)); j0 = int(np.floor(fy)); k0 = int(np.floor(fz))
    rx = fx - i0; ry = fy - j0; rz = fz - k0
    oi, oj, ok = _cell(p, lo, h, N)
    own = (oi * N[1] + oj) * N[2] + ok
    s = 0.0; m = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                ii = min(max(i0 + a, 0), N[0] - 1); jj = min(max(j0 + b, 0), N[1] - 1); kk = min(max(k0 + c, 0), N[2] - 1)
                wt = (rx if a else 1 - rx) * (ry if b else 1 - ry) * (rz if c else 1 - rz)
                dx = min(max(ii - oi, -1), 1) + 1; dy = min(max(jj - oj, -1), 1) + 1; dz = min(max(kk - ok, -1), 1) + 1
                if tab[own, dx * 9 + dy * 3 + dz]:
                    wt = 0.0
                idx[m] = (ii * N[1] + jj) * N[2] + kk; w[m] = wt; s += wt; m += 1
    if s <= 0:
        for q in range(8):
            w[q] = 0.0
        idx[0] = own; w[0] = 1.0
    else:
        for q in range(8):
            w[q] /= s


@njit(cache=True)
def _weights_ext(p, lo, h, N, tab, idx, w):
    """Like _weights, but corners outside the domain (sides/top; the ground is a wall and
    is clamped) get idx = -1 so the caller can use the outside-air (far-field) value:
    the backward-scheme equivalent of ghost points."""
    fx = (p[0] - lo[0]) / h - 0.5; fy = (p[1] - lo[1]) / h - 0.5; fz = (p[2] - lo[2]) / h - 0.5
    i0 = int(np.floor(fx)); j0 = int(np.floor(fy)); k0 = int(np.floor(fz))
    rx = fx - i0; ry = fy - j0; rz = fz - k0
    oi, oj, ok = _cell(p, lo, h, N)
    own = (oi * N[1] + oj) * N[2] + ok
    s = 0.0; m = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                ii = i0 + a; jj = j0 + b; kk = max(k0 + c, 0)
                wt = (rx if a else 1 - rx) * (ry if b else 1 - ry) * (rz if c else 1 - rz)
                if ii < 0 or jj < 0 or ii >= N[0] or jj >= N[1] or kk >= N[2]:
                    idx[m] = -1
                else:
                    dx = min(max(ii - oi, -1), 1) + 1; dy = min(max(jj - oj, -1), 1) + 1; dz = min(max(kk - ok, -1), 1) + 1
                    if tab[own, dx * 9 + dy * 3 + dz]:
                        wt = 0.0
                    idx[m] = (ii * N[1] + jj) * N[2] + kk
                w[m] = wt; s += wt; m += 1
    if s <= 0:
        for q in range(8):
            w[q] = 0.0
        idx[0] = own; w[0] = 1.0
    else:
        for q in range(8):
            w[q] /= s


@njit(cache=True)
def _dev_codes(p, lo, h, N, idx, dvu, dvv, dvw, dcode):
    """For each of the 8 stencil corners: is it on the other side of a supply-device outlet
    face from the point's own cell? If so dcode = device id (the corner reads supply air),
    so the outlet acts as an inflow boundary with the device's supply state."""
    oi, oj, ok = _cell(p, lo, h, N)
    for q in range(8):
        dcode[q] = 0
        if idx[q] < 0:
            continue
        c = idx[q]; ci = c // (N[1] * N[2]); cj = (c // N[2]) % N[1]; ck = c % N[2]
        if ci != oi:
            f = max(ci, oi)
            if dvu[f, oj, ok] > 0:
                dcode[q] = dvu[f, oj, ok]
        if cj != oj:
            f = max(cj, oj)
            if dvv[oi, f, ok] > 0:
                dcode[q] = dvv[oi, f, ok]
        if ck != ok:
            f = max(ck, ok)
            if dvw[oi, oj, f] > 0:
                dcode[q] = dvw[oi, oj, f]


@njit(cache=True)
def _vel(p, lo, h, N, tab, U, idx, w):
    _weights(p, lo, h, N, tab, idx, w)
    vx = 0.0; vy = 0.0; vz = 0.0
    for q in range(8):
        vx += w[q] * U[idx[q], 0]; vy += w[q] * U[idx[q], 1]; vz += w[q] * U[idx[q], 2]
    return vx, vy, vz


@njit(cache=True)
def _move(p, sgn, dt, ns, lo, hi, h, N, tab, U, mu, mv, mw, dvu, dvv, dvw, idx, w):
    """Move point p by sgn*u*dt in ns substeps; stop at blocked faces.
    Returns 0 (still inside), -1 (left through an open outer face) or k > 0 (crossed the
    outlet plane of supply device k: the air there is that device's supply air)."""
    ds = dt / ns
    for _ in range(ns):
        vx, vy, vz = _vel(p, lo, h, N, tab, U, idx, w)
        i0, j0, k0 = _cell(p, lo, h, N)
        q0 = p[0] + sgn * ds * vx; q1 = p[1] + sgn * ds * vy; q2 = p[2] + sgn * ds * vz
        if q2 < lo[2]:
            q2 = lo[2] + 1e-6
        if q0 < lo[0] or q0 > hi[0] or q1 < lo[1] or q1 > hi[1] or q2 > hi[2]:
            return -1
        i1 = min(max(int(np.floor((q0 - lo[0]) / h)), 0), N[0] - 1)
        j1 = min(max(int(np.floor((q1 - lo[1]) / h)), 0), N[1] - 1)
        k1 = min(max(int(np.floor((q2 - lo[2]) / h)), 0), N[2] - 1)
        if i1 != i0:
            f = i0 + 1 if i1 > i0 else i0
            if dvu[f, j0, k0] > 0:
                return dvu[f, j0, k0]
            if mu[f, j0, k0] == 1:
                q0 = lo[0] + f * h + (-1e-6 if i1 > i0 else 1e-6)
        if j1 != j0:
            f = j0 + 1 if j1 > j0 else j0
            if dvv[i0, f, k0] > 0:
                return dvv[i0, f, k0]
            if mv[i0, f, k0] == 1:
                q1 = lo[1] + f * h + (-1e-6 if j1 > j0 else 1e-6)
        if k1 != k0:
            f = k0 + 1 if k1 > k0 else k0
            if dvw[i0, j0, f] > 0:
                return dvw[i0, j0, f]
            if mw[i0, j0, f] == 1:
                q2 = lo[2] + f * h + (-1e-6 if k1 > k0 else 1e-6)
        p[0] = q0; p[1] = q1; p[2] = q2
    return 0


@njit(parallel=True, cache=True)
def backward_gather(X, dt, ns, lo, hi, h, N, tab, U, mu, mv, mw, dvu, dvv, dvw, F, far_vals, dev_vals, sgn=-1.0):
    """F: (n, nf) fields at centers; far_vals: (n, nf) far-field value for each cell's
    departure (computed outside from the departure height). Returns new fields and the
    departure points."""
    n = X.shape[0]; nf = F.shape[1]
    out = np.empty_like(F); far = np.zeros(n, np.bool_); P = np.empty_like(X)
    for c in prange(n):
        idx = np.empty(8, np.int64); w = np.empty(8); dcode = np.zeros(8, np.int64)
        p = X[c].copy()
        code = _move(p, sgn, dt, ns, lo, hi, h, N, tab, U, mu, mv, mw, dvu, dvv, dvw, idx, w)
        far[c] = code == -1; P[c] = p
        if code == -1:
            for k in range(nf):
                out[c, k] = far_vals[c, k]
        elif code > 0:
            for k in range(nf):
                out[c, k] = dev_vals[code - 1, k]
        else:
            _weights_ext(p, lo, h, N, tab, idx, w)
            _dev_codes(p, lo, h, N, idx, dvu, dvv, dvw, dcode)
            for k in range(nf):
                s = 0.0
                for q in range(8):
                    if dcode[q] > 0:
                        s += w[q] * dev_vals[dcode[q] - 1, k]
                    elif idx[q] < 0:
                        s += w[q] * far_vals[c, k]
                    else:
                        s += w[q] * F[idx[q], k]
                out[c, k] = s
    return out, far, P


@njit(cache=True)
def _tent(p, lo, h, N, tab, is_real, idx, w):
    """Raw tent (trilinear) weights of point p onto the 8 surrounding centers. Corners outside
    the domain are dropped. For real points, corners across a wall are dropped and the rest
    renormalized (so the point's heat is not lost); ghost points are not renormalized."""
    fx = (p[0] - lo[0]) / h - 0.5; fy = (p[1] - lo[1]) / h - 0.5; fz = (p[2] - lo[2]) / h - 0.5
    i0 = int(np.floor(fx)); j0 = int(np.floor(fy)); k0 = int(np.floor(fz))
    rx = fx - i0; ry = fy - j0; rz = fz - k0
    oi, oj, ok = _cell(p, lo, h, N); own = (oi * N[1] + oj) * N[2] + ok
    s = 0.0; m = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                ii = i0 + a; jj = j0 + b; kk = k0 + c
                wt = (rx if a else 1 - rx) * (ry if b else 1 - ry) * (rz if c else 1 - rz)
                if ii < 0 or jj < 0 or kk < 0 or ii >= N[0] or jj >= N[1] or kk >= N[2]:
                    wt = 0.0; ii = 0; jj = 0; kk = 0
                elif is_real:
                    dx = min(max(ii - oi, -1), 1) + 1; dy = min(max(jj - oj, -1), 1) + 1; dz = min(max(kk - ok, -1), 1) + 1
                    if tab[own, dx * 9 + dy * 3 + dz]:
                        wt = 0.0
                idx[m] = (ii * N[1] + jj) * N[2] + kk; w[m] = wt; s += wt; m += 1
    if is_real and s > 0:
        for q in range(8):
            w[q] /= s


@njit(cache=True)
def forward_scatter(Xp, Fp, n_real, dt, ns, lo, hi, h, N, tab, U, mu, mv, mw, dvu, dvv, dvw, F_old, dev_vals):
    """Xp: (m,3) points = the n_real cell centers followed by ghost outside-air points;
    Fp: (m,nf) their values. Each point moves forward to p_future; each FIXED center takes
    the tent-weighted average of the moved points. Centers nothing lands on keep F_old."""
    n = N[0] * N[1] * N[2]; m = Xp.shape[0]; nf = Fp.shape[1]
    num = np.zeros((n, nf)); den = np.zeros(n)
    idx = np.empty(8, np.int64); w = np.empty(8)
    for j in range(m):
        p = Xp[j].copy()
        real = j < n_real
        val = Fp[j].copy()
        if real:
            code = _move(p, 1.0, dt, ns, lo, hi, h, N, tab, U, mu, mv, mw, dvu, dvv, dvw, idx, w)
            if code == -1:
                continue                               # left the domain
            if code > 0:                               # passed through a device: now supply air
                val = dev_vals[code - 1].copy()
        else:
            i, jj, k = _cell(p, lo, h, N); c = (i * N[1] + jj) * N[2] + k
            p[0] += dt * U[c, 0]; p[1] += dt * U[c, 1]; p[2] += dt * U[c, 2]
        _tent(p, lo, h, N, tab, real, idx, w)
        for q in range(8):
            if w[q] > 0:
                den[idx[q]] += w[q]
                for k in range(nf):
                    num[idx[q], k] += w[q] * val[k]
    out = F_old.copy()
    for c in range(n):
        if den[c] > 1e-12:
            for k in range(nf):
                out[c, k] = num[c, k] / den[c]
    return out


@njit(parallel=True, cache=True)
def face_gather(Xf, dt, ns, lo, hi, h, N, tab, U, mu, mv, mw, dvu, dvv, dvw, F, origin, farv, sgn=-1.0):
    """MAC velocity advection: each face of one velocity component is traced back along the
    (cell-centered) wind; the component is read there by trilinear interpolation on its own
    staggered grid F (origin = position of F[0,0,0]). Leaving the domain -> farv[i]."""
    n = Xf.shape[0]; out = np.empty(n)
    S0, S1, S2 = F.shape
    for c in prange(n):
        idx = np.empty(8, np.int64); w = np.empty(8)
        p = Xf[c].copy()
        p[0] = min(max(p[0], lo[0] + 1e-9), hi[0] - 1e-9)
        p[1] = min(max(p[1], lo[1] + 1e-9), hi[1] - 1e-9)
        p[2] = min(max(p[2], lo[2] + 1e-9), hi[2] - 1e-9)
        if _move(p, sgn, dt, ns, lo, hi, h, N, tab, U, mu, mv, mw, dvu, dvv, dvw, idx, w) == -1:
            out[c] = farv[c]; continue
        fx = (p[0] - origin[0]) / h; fy = (p[1] - origin[1]) / h; fz = (p[2] - origin[2]) / h
        i0 = int(np.floor(fx)); j0 = int(np.floor(fy)); k0 = int(np.floor(fz))
        rx = fx - i0; ry = fy - j0; rz = fz - k0
        s = 0.0
        for a in range(2):
            for b in range(2):
                for cc in range(2):
                    ii = i0 + a; jj = j0 + b; kk = max(k0 + cc, 0)
                    wt = (rx if a else 1 - rx) * (ry if b else 1 - ry) * (rz if cc else 1 - rz)
                    if ii < 0 or jj < 0 or ii >= S0 or jj >= S1 or kk >= S2:
                        s += wt * farv[c]
                    else:
                        s += wt * F[ii, jj, kk]
        out[c] = s
    return out
