"""
MODEL 2 (ADVANCED) - core solver.

Builds on Model 1 (same psychrometrics, same no-penetration walls with U-values, same
sun/shade radiation, same 17-segment JOS-3 bodies and comfort weights, same efficiency
metric) and adds:
  * a resolved, time-dependent WIND field (Stable Fluids: semi-Lagrangian advection,
    eddy viscosity, Boussinesq BUOYANCY on virtual temperature, pressure projection),
    with outside wind entering the domain;
  * heat and humidity carried by that wind (semi-Lagrangian, wall-aware) + eddy diffusion;
  * every cell holds T, humidity, wind velocity and Tmrt -> a FULL UTCI per cell;
  * JOS-3 bodies read the same cell values (incl. local wind) and put their own heat
    and moisture back into those cells every 30 s; flow stays coupled while people
    are present (their plumes rise);
  * real devices driven by their spec-sheet outputs (fan, evaporative cooler,
    high-pressure misting, spot AC, air curtain).

Grid: uniform cubes of side h. Velocities on faces (MAC) for the projection,
collocated copies at centers for advection/buoyancy.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import pyamg
from scipy.ndimage import minimum_filter, maximum_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'model1_basic'))
from shelter_model import (W_from_rh, rh_from_W, wet_bulb_stull, p_sat, RHO, CP, H_FG,
                           SIGMA, P_ATM)

G_ACC = 9.81
UTCI_V10 = np.log(10 / 0.01) / np.log(1.1 / 0.01)     # body-height wind -> 10 m (UTCI convention)


def utci_full(Ta, Tmrt, rh, speed):
    """Official UTCI (Broede et al. 2012 polynomial via pythermalcomfort) with all four
    inputs taken from the cell: air temp, Tmrt, humidity and the cell's own wind speed."""
    from pythermalcomfort.models import utci
    v10 = np.clip(np.asarray(speed) * UTCI_V10, 0.5, 17.0)
    Ta = np.clip(Ta, -50, 50); dT = np.clip(Tmrt - Ta, -30, 70)
    return np.asarray(utci(tdb=Ta, tr=Ta + dT, v=v10, rh=rh, limit_inputs=False,
                           round_output=False).utci, float)


def sun_tmrt(climate, dTmrt_shade=3.0):
    """Mean radiant temperature of a person standing in the open sun for this weather (same formula as a
    sunlit cell). Used for the walk to the space, so it never depends on heat sources inside the venue."""
    Tsh = climate['T_out'] + dTmrt_shade + 273.15
    return float((Tsh ** 4 + 0.25 * 0.7 * climate['I_dir'] / (0.97 * SIGMA)) ** 0.25 - 273.15)


def W_sat(T):
    ps = p_sat(T)
    return 0.622 * ps / (P_ATM - ps)


# ====================================================================== geometry
class Wall:
    """Thin axis-aligned panel on a cell-face plane. axis = normal axis, c = plane coord,
    r1/r2 = extents along the other two axes (in increasing axis order).
    kind 'wall' | 'roof' | 'glass'. U = panel U-value [W/m2K], absorptance for sun."""

    def __init__(self, axis, c, r1, r2, kind='wall', U=5.0, absorb=0.6):
        self.axis, self.c, self.r1, self.r2 = axis, c, r1, r2
        self.kind, self.U, self.absorb = kind, U, absorb


class Device:
    """Spec-driven device = a 'fan plane' of fixed face velocities.
    kind: 'fan' | 'evap' | 'spotac' | 'curtain' | 'mist'
    axis/sign: blowing direction; center (x,y,z); size (w,h) of the outlet in the two
    other axes; flow [m3/s] from the spec sheet; power [kW]; extra spec fields."""

    def __init__(self, name, kind, center, axis=1, sign=+1, size=(0.6, 0.6), flow=0.0,
                 power_kW=0.0, **spec):
        self.name, self.kind, self.center = name, kind, np.array(center, float)
        self.axis, self.sign, self.size, self.flow, self.power_kW = axis, sign, size, flow, power_kW
        self.spec = spec


class Geometry:
    def __init__(self, name, h, lo, hi, walls, interior_box, occupied_z=(0.5, 1.75),
                 sun_dir=(0.0, 1.0, 0.0), sun_elev_deg=40.0):
        self.name, self.h = name, h
        self.lo = np.array(lo, float); self.hi = np.array(hi, float)
        self.N = np.round((self.hi - self.lo) / h).astype(int)
        self.walls = walls
        self.box = interior_box                     # ((x0,x1),(y0,y1),(z0,z1)) sheltered space
        self.occ_z = occupied_z
        e = np.radians(sun_elev_deg)
        d = np.array(sun_dir, float); d[2] = 0; d /= np.linalg.norm(d)
        self.sun = np.array([d[0] * np.cos(e), d[1] * np.cos(e), np.sin(e)])
        self.solid = None          # optional (N) bool: solid cells from a CAD model (blocks air, casts shade)
        self.occ_mask = None       # optional (N) bool: custom occupied zone (e.g. the seating of a stadium)


def bus_shelter(h=0.125):
    """Part 2a geometry (Model 1): 3.0 x 1.5 x 2.5 m, back wall + side panels + roof, open front."""
    W = [Wall(1, 0.0, (0, 3), (0, 2.5)), Wall(0, 0.0, (0, 1.5), (0, 2.5), 'glass'),
         Wall(0, 3.0, (0, 1.5), (0, 2.5), 'glass'), Wall(2, 2.5, (0, 3), (0, 1.5), 'roof')]
    return Geometry('Bus shelter (2a)', h, (-1.5, -0.75, 0), (4.5, 3.0, 3.5), W,
                    ((0, 3), (0, 1.5), (0, 2.5)))


def rail_canopy(h=0.25):
    """Part 2b: semi-open rail-platform waiting area (designed, assumptions stated).
    20 m long x 4 m deep, canopy 3.5 m high with 0.5 m overhang over the track edge,
    solid back wall (station building side), glass wind screens at both ends covering
    the back half of the depth to 2.5 m height, track side fully open."""
    W = [Wall(1, 0.0, (0, 20), (0, 3.5)),                      # back wall
         Wall(0, 0.0, (0, 2.0), (0, 2.5), 'glass'),            # end wind screens
         Wall(0, 20.0, (0, 2.0), (0, 2.5), 'glass'),
         Wall(2, 3.5, (0, 20), (0, 4.5), 'roof')]              # canopy roof (+0.5 m overhang)
    return Geometry('Rail platform canopy (2b)', h, (-3.0, -2.0, 0), (23.0, 10.0, 6.0), W,
                    ((0, 20), (0, 4), (0, 3.5)))


# ====================================================================== solver
class Model2:
    def __init__(self, geom, climate, wind_dir='front', U10=None, devices=(), alpha_t=0.02,
                 nu_t=0.02, buoyancy=True, z0=0.1, ground_dT_sun=15.0, ground_dT_shade=2.0,
                 dTmrt_shade=3.0, scheme='bwd', bfecc=True, bfecc_vel=False, verbose=False):
        self.g = g = geom; self.cl = climate; self.verbose = verbose
        self.T_out, self.rh_out = climate['T_out'], climate['rh_out']
        self.W_out = W_from_rh(self.T_out, self.rh_out)
        self.U10 = climate['U10'] if U10 is None else U10
        self.alpha_t, self.nu_t, self.buoy, self.z0 = alpha_t, nu_t, buoyancy, z0
        self.devices = list(devices); self.wind_dir = wind_dir; self.scheme = scheme; self.bfecc = bfecc; self.bfecc_vel = bfecc_vel
        N = g.N; h = g.h; self.n = int(np.prod(N))
        ax = [g.lo[i] + (np.arange(N[i]) + 0.5) * h for i in range(3)]
        self.xc, self.yc, self.zc = ax
        Xg = np.meshgrid(*ax, indexing='ij'); self.X = np.stack([a.ravel() for a in Xg], 1)
        self._build_faces()
        self._build_wall_links()
        self._build_radiation(ground_dT_sun, ground_dT_shade, dTmrt_shade)
        self._build_devices()
        self._build_poisson()
        # state
        self.T = np.full(tuple(N), self.T_out); self.W = np.full(tuple(N), self.W_out)
        self.U = np.zeros(tuple(N) + (3,))
        self.u = np.where(self.mu == 2, self.fu, 0.0); self.v = np.where(self.mv == 2, self.fv, 0.0)
        self.w = np.where(self.mw == 2, self.fw, 0.0)
        self.U = self._centers()
        self.time = 0.0; self.phi = np.zeros(self.n)
        self.extra_T = np.zeros(tuple(N)); self.extra_W = np.zeros(tuple(N))   # W/cell, kg/s/cell
        self.water_used = 0.0; self.water_evap = 0.0
        self.dev_vals = np.tile([self.T_out, self.W_out], (max(len(self.supply_ids), 1), 1)).astype(float)

    # ------------------------------------------------------------------ faces
    def _face_index(self, axis, coord):
        return int(round((coord - self.g.lo[axis]) / self.g.h))

    def _cells_in(self, axis_vals, r):
        return (axis_vals >= r[0] - 1e-9) & (axis_vals <= r[1] + 1e-9)

    def _build_faces(self):
        g = self.g; N = g.N
        self.mu = np.zeros((N[0] + 1, N[1], N[2]), np.int8); self.fu = np.zeros(self.mu.shape)
        self.mv = np.zeros((N[0], N[1] + 1, N[2]), np.int8); self.fv = np.zeros(self.mv.shape)
        self.mw = np.zeros((N[0], N[1], N[2] + 1), np.int8); self.fw = np.zeros(self.mw.shape)
        cen = [self.xc, self.yc, self.zc]
        for wl in g.walls:
            o = [i for i in range(3) if i != wl.axis]
            k = self._face_index(wl.axis, wl.c)
            A = self._cells_in(cen[o[0]], wl.r1); B = self._cells_in(cen[o[1]], wl.r2)
            m = [self.mu, self.mv, self.mw][wl.axis]
            sl = [slice(None)] * 3; sl[wl.axis] = k
            sub = m[tuple(sl)]; sub[np.ix_(A, B)] = 1
        self.mw[:, :, 0] = 1                                            # ground
        S = g.solid
        if S is not None:                                               # every face of a solid cell is blocked
            self.mu[1:-1][S[1:] | S[:-1]] = 1; self.mu[0][S[0]] = 1; self.mu[-1][S[-1]] = 1
            self.mv[:, 1:-1][S[:, 1:] | S[:, :-1]] = 1; self.mv[:, 0][S[:, 0]] = 1; self.mv[:, -1][S[:, -1]] = 1
            self.mw[:, :, 1:-1][S[:, :, 1:] | S[:, :, :-1]] = 1; self.mw[:, :, -1][S[:, :, -1]] = 1
        # outside wind: log-law inflow through one outer face
        prof = lambda z: self.U10 * np.log(np.maximum(z, self.z0) / self.z0) / np.log(10 / self.z0)
        self.prof = prof; zc = self.zc
        d = self.wind_dir
        if self.U10 > 0 and d is not None:
            if d == 'front':      # wind blowing INTO the open side (open side faces +y)
                self.mv[:, -1, :] = 2; self.fv[:, -1, :] = -prof(zc)[None, :]
                self.U_far = np.array([0, -1, 0.])
            elif d == 'along':
                self.mu[0, :, :] = 2; self.fu[0, :, :] = prof(zc)[None, :]
                self.U_far = np.array([1, 0, 0.])
            elif d == 'against':  # from +x (opposite of 'along'): e.g. the sea breeze from the east at the stadium
                self.mu[-1, :, :] = 2; self.fu[-1, :, :] = -prof(zc)[None, :]
                self.U_far = np.array([-1, 0, 0.])
            elif d == 'behind':
                self.mv[:, 0, :] = 2; self.fv[:, 0, :] = prof(zc)[None, :]
                self.U_far = np.array([0, 1, 0.])
        else:
            self.U_far = np.zeros(3)

    # ------------------------------------------------------------------ wall heat links
    def _build_wall_links(self):
        """For every blocked face that belongs to a panel: the two cells either side and
        the panel properties (heat passes through panels by U-value, roof also absorbs sun)."""
        g = self.g; N = g.N; h = g.h; cen = [self.xc, self.yc, self.zc]
        links = []
        for wl in g.walls:
            o = [i for i in range(3) if i != wl.axis]
            k = self._face_index(wl.axis, wl.c)
            A = np.where(self._cells_in(cen[o[0]], wl.r1))[0]; B = np.where(self._cells_in(cen[o[1]], wl.r2))[0]
            aa, bb = np.meshgrid(A, B, indexing='ij'); aa = aa.ravel(); bb = bb.ravel()
            lo_ijk = np.zeros((len(aa), 3), int); lo_ijk[:, wl.axis] = k - 1
            lo_ijk[:, o[0]] = aa; lo_ijk[:, o[1]] = bb
            hi_ijk = lo_ijk.copy(); hi_ijk[:, wl.axis] = k
            ok = (lo_ijk[:, wl.axis] >= 0) & (hi_ijk[:, wl.axis] < N[wl.axis])
            for a, b in zip(lo_ijk[ok], hi_ijk[ok]):
                links.append((np.ravel_multi_index(a, N), np.ravel_multi_index(b, N), wl.U, wl.kind, wl.absorb, wl.axis))
        self.wl_a = np.array([l[0] for l in links], int); self.wl_b = np.array([l[1] for l in links], int)
        self.wl_U = np.array([l[2] for l in links]); self.wl_roof = np.array([l[3] == 'roof' for l in links])
        self.wl_abs = np.array([l[4] for l in links])

    # ------------------------------------------------------------------ radiation
    def _ray_blocked(self, P, d):
        """Does the ray P + t d (t>0) hit any panel?"""
        hit = np.zeros(len(P), bool)
        for wl in self.g.walls:
            a = wl.axis
            if abs(d[a]) < 1e-12:
                continue
            t = (wl.c - P[:, a]) / d[a]
            Q = P + t[:, None] * d
            o = [i for i in range(3) if i != a]
            inside = ((Q[:, o[0]] >= wl.r1[0]) & (Q[:, o[0]] <= wl.r1[1]) &
                      (Q[:, o[1]] >= wl.r2[0]) & (Q[:, o[1]] <= wl.r2[1]) & (t > 1e-9))
            hit |= inside & (wl.kind != 'glass')       # glass screens let direct sun through
        return hit

    def _build_radiation(self, dsun, dshade, dTmrt_shade):
        I_dir = self.cl['I_dir']
        sunlit = ~(self._ray_blocked(self.X, self.g.sun) | self._voxel_shadow(self.X)) if I_dir > 0 else np.zeros(self.n, bool)
        Tsh = self.T_out + dTmrt_shade + 273.15
        Tsun = (Tsh ** 4 + 0.25 * 0.7 * I_dir / (0.97 * SIGMA)) ** 0.25
        self.Tmrt = (np.where(sunlit, Tsun, Tsh) - 273.15).reshape(tuple(self.g.N))
        self.sunlit = sunlit.reshape(tuple(self.g.N))
        self.Tmrt_open_sun = float(Tsun - 273.15)   # weather only: the walk to the space (no venue heat sources)
        # ground surface temperature (sunlit pavement hotter) for the bottom cells
        ground = np.c_[self.X[:, :2], np.zeros(self.n)]
        if self.g.solid is not None:            # the floor under each air cell is the top of the solid below it
            S = self.g.solid.ravel(); below = np.r_[np.zeros(1, bool), S[:-1]].reshape(tuple(self.g.N))
            below[:, :, 0] = True; below = below & ~self.g.solid
            ground = self.X.copy(); ground[:, 2] -= 0.5 * self.g.h - 1e-3
            self.floor = below
        gsun = ~(self._ray_blocked(ground, self.g.sun) | self._voxel_shadow(ground)) if I_dir > 0 else np.zeros(self.n, bool)
        Tg = np.where(gsun, self.T_out + dsun, self.T_out + dshade).reshape(tuple(self.g.N))
        self.T_ground = Tg[:, :, 0]
        self.T_floor = Tg if self.g.solid is not None else None
        self.ceil = None
        if self.g.solid is not None:            # air cells directly under a solid (roof underside)
            S = self.g.solid; N = tuple(self.g.N)
            ceil = np.zeros(N, bool); ceil[:, :, :-1] = S[:, :, 1:] & ~S[:, :, :-1]
            # is the top of that column in the sun? (= the floor cell above the highest solid is sunlit)
            ktop = N[2] - 1 - np.argmax(S[:, :, ::-1], 2); has = S.any(2)
            kk = np.clip(ktop + 1, 0, N[2] - 1)
            gs = gsun.reshape(N); I_, J_ = np.indices(N[:2])
            expo = has & (ktop + 1 < N[2]) & gs[I_, J_, kk]
            self.ceil = ceil
            # sunlit roof panel temperature, same balance as a Wall roof: abs*I = h_o (Tp-T_out) + U (Tp-T_in)
            Ur = getattr(self.g, 'roof_U', 5.0); self.roof_U = Ur
            self.T_ceil_rise = np.where(expo, 0.6 * self.cl['I_roof'] / (15.0 + Ur), 0.0)[:, :, None] * np.ones(N)
            if getattr(self.g, 'longwave', True):
                self._longwave_tmrt(Tg, sunlit, I_dir, dsun, dTmrt_shade)

    def _longwave_tmrt(self, Tg, sunlit, I_dir, dsun, dTmrt_shade):
        """CAD venues: mean radiant temperature from the surfaces around each person, not a constant.
        Lower half of the view = the floor/seat surface underneath (sunlit concrete is hot, shaded is not);
        upper half = the underside of whatever is overhead (a sun-heated roof radiates down) or, in the open,
        the same 'shade' value used everywhere else (T_out + dTmrt_shade). Direct sun is then added exactly
        as before:  Tmrt^4 = 0.5 T_down^4 + 0.5 T_up^4 + 0.25*0.7*I_dir/(0.97 sigma)   (temperatures in K)"""
        S = self.g.solid; N = tuple(self.g.N); K = 273.15
        Tdown = np.empty(N); Tdown[:, :, 0] = Tg[:, :, 0]
        for k in range(1, N[2]):                     # surface below = nearest floor cell at or below
            Tdown[:, :, k] = np.where(self.floor[:, :, k], Tg[:, :, k], Tdown[:, :, k - 1])
        # roof underside radiating down: inside surface = T_in + U (Tp - T_in) / h_i, h_i = 8 W/m2K
        T_under = self.T_out + self.roof_U * self.T_ceil_rise / 8.0
        Tup = np.empty(N); Tup[:, :, -1] = self.T_out + dTmrt_shade
        for k in range(N[2] - 2, -1, -1):            # surface above = nearest solid overhead, else the open sky value
            Tup[:, :, k] = np.where(S[:, :, k + 1], T_under[:, :, k], Tup[:, :, k + 1])
        Tlw4 = 0.5 * (Tdown + K) ** 4 + 0.5 * (Tup + K) ** 4
        sun = 0.25 * 0.7 * I_dir / (0.97 * SIGMA)
        self.Tmrt = (np.where(sunlit.reshape(N), Tlw4 + sun, Tlw4) ** 0.25 - K)
        self.Tmrt_longwave = Tlw4 ** 0.25 - K
        # walk to the venue: open sun over sunlit pavement, same in every case (weather only)
        self.Tmrt_open_sun = float((0.5 * (self.T_out + dsun + K) ** 4 + 0.5 * (self.T_out + dTmrt_shade + K) ** 4 + sun) ** 0.25 - K)

    def _voxel_shadow(self, P):
        """Is the ray from P toward the sun blocked by a solid CAD cell? (marched in half-cell steps)"""
        S = self.g.solid
        if S is None:
            return np.zeros(len(P), bool)
        g = self.g; d = g.sun; h = g.h; hit = np.zeros(len(P), bool)
        smax = (g.hi[2] - g.lo[2]) / max(d[2], 0.05)
        for c in range(0, len(P), 200000):
            Q = P[c:c + 200000].copy(); hh = np.zeros(len(Q), bool)
            for s_ in np.arange(0.5 * h, smax, 0.5 * h):
                R = Q + s_ * d
                ijk = np.floor((R - g.lo) / h).astype(int)
                inside = np.all((ijk >= 0) & (ijk < g.N), 1)
                if not inside.any():
                    break
                hh[inside] |= S[tuple(ijk[inside].T)]
            hit[c:c + 200000] = hh
        return hit

    # ------------------------------------------------------------------ devices
    def _build_devices(self):
        g = self.g; h = g.h; cen = [self.xc, self.yc, self.zc]
        self.dev_state = []
        self.dvu = np.zeros(self.mu.shape, np.int64); self.dvv = np.zeros(self.mv.shape, np.int64)
        self.dvw = np.zeros(self.mw.shape, np.int64)
        self.supply_ids = []
        for dv in self.devices:
            a = dv.axis; o = [i for i in range(3) if i != a]
            k = self._face_index(a, dv.center[a])
            A = np.abs(cen[o[0]] - dv.center[o[0]]) <= dv.size[0] / 2 + 1e-9
            B = np.abs(cen[o[1]] - dv.center[o[1]]) <= dv.size[1] / 2 + 1e-9
            ia, ib = np.where(A)[0], np.where(B)[0]
            area = len(ia) * len(ib) * h * h
            V = dv.sign * dv.flow / area if area > 0 else 0.0
            if dv.kind == 'curtain':          # resolved slot: keep the spec momentum flux
                V = dv.sign * dv.spec['V_slot'] * np.sqrt(dv.spec['slot'] / h)
            m, f = [(self.mu, self.fu), (self.mv, self.fv), (self.mw, self.fw)][a]
            if dv.kind != 'mist':
                sl = [slice(None)] * 3; sl[a] = k
                sm = m[tuple(sl)]; sf = f[tuple(sl)]
                sm[np.ix_(ia, ib)] = 2; sf[np.ix_(ia, ib)] = V
            if dv.kind in ('evap', 'spotac'):
                self.supply_ids.append(len(self.dev_state))
                did = len(self.supply_ids)
                dvarr = [self.dvu, self.dvv, self.dvw][a]
                sl = [slice(None)] * 3; sl[a] = k
                sd = dvarr[tuple(sl)]; sd[np.ix_(ia, ib)] = did
            aa, bb = np.meshgrid(ia, ib, indexing='ij'); aa = aa.ravel(); bb = bb.ravel()
            up = np.zeros((len(aa), 3), int); up[:, o[0]] = aa; up[:, o[1]] = bb
            dn = up.copy()
            up[:, a] = k - 1 if dv.sign > 0 else k; dn[:, a] = k if dv.sign > 0 else k - 1
            st = dict(dev=dv, area=area, V=V,
                      up=np.ravel_multi_index(up.T, tuple(g.N)), dn=np.ravel_multi_index(dn.T, tuple(g.N)))
            if g.solid is not None:       # a device must take air from and blow into AIR cells, never a solid
                Sf = g.solid.ravel(); bad = dv.kind != 'mist' and (Sf[st['dn']].any() or Sf[st['up']].any())
                if bad:
                    raise ValueError(f'device {dv.name} at {tuple(dv.center)} draws from or blows into a solid cell')
            # downstream cell layers along the jet axis (layer 0 = outlet cells), used to place each
            # step's supply air volume Vdot*dt where it actually is
            layers = []
            for kk in range(24):
                q = dn.copy(); q[:, a] = dn[:, a] + (kk if dv.sign > 0 else -kk)
                if (q[:, a] < 0).any() or (q[:, a] >= g.N[a]).any():
                    break
                layers.append(np.ravel_multi_index(q.T, tuple(g.N)))
            st['layers'] = layers
            if dv.kind == 'mist':
                # nozzle line: cells along the line at the given height
                st['cells'] = st['dn']
                if dv.spec.get('follow_surface') and g.solid is not None:
                    # nozzles hang just above the stepped seating: use the first air cell above the surface
                    ijk = np.array(np.unravel_index(st['dn'], tuple(g.N))).T
                    kf = np.clip(np.ceil(g.top[ijk[:, 0], ijk[:, 1]] / h - 1e-9).astype(int), 0, g.N[2] - 1)
                    ijk[:, 2] = kf; st['cells'] = np.unique(np.ravel_multi_index(ijk.T, tuple(g.N)))
                    st['dn'] = st['cells']
                if g.solid is not None and g.solid.ravel()[st['cells']].any():
                    raise ValueError(f'misting line {dv.name} at {tuple(dv.center)} sprays into a solid cell')
            if dv.kind == 'spotac':
                # condenser exhaust ducted to outside air above the canopy roof
                ex = np.array(dv.spec['exhaust_at'], float)
                st['exhaust'] = np.ravel_multi_index(np.clip(((ex - g.lo) / h).astype(int), 0, g.N - 1), tuple(g.N))
            self.dev_state.append(st)

    # ------------------------------------------------------------------ pressure Poisson
    def _build_poisson(self):
        N = self.g.N; n = self.n; h = self.g.h
        idx = np.arange(n).reshape(tuple(N))
        rows, cols, vals = [], [], []; diag = np.zeros(n)
        c = 1.0 / h ** 2
        for axis, m in enumerate((self.mu, self.mv, self.mw)):
            sl_int = [slice(None)] * 3; sl_int[axis] = slice(1, -1)
            mi = m[tuple(sl_int)] == 0
            lo = [slice(None)] * 3; lo[axis] = slice(0, -1); hi = [slice(None)] * 3; hi[axis] = slice(1, None)
            p = idx[tuple(lo)][mi]; q = idx[tuple(hi)][mi]
            rows += [p, q]; cols += [q, p]; vals += [-c * np.ones(len(p))] * 2
            np.add.at(diag, p, c); np.add.at(diag, q, c)
            for side, fi, ci in ((0, 0, 0), (1, -1, -1)):          # open outer faces: p = 0
                sf = [slice(None)] * 3; sf[axis] = fi; sc = [slice(None)] * 3; sc[axis] = ci
                ob = m[tuple(sf)] == 0
                np.add.at(diag, idx[tuple(sc)][ob], 2 * c)
        diag[diag == 0] = 1.0
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
        L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        self.Lp = L
        self.amg = pyamg.smoothed_aggregation_solver(L, max_coarse=500)

    def _divergence(self):
        h = self.g.h
        return ((self.u[1:] - self.u[:-1]) + (self.v[:, 1:] - self.v[:, :-1]) + (self.w[:, :, 1:] - self.w[:, :, :-1])) / h

    def _project(self):
        div = self._divergence().ravel()
        self.phi = self.amg.solve(div, x0=self.phi, tol=1e-5, accel='cg', maxiter=60)
        phi = self.phi.reshape(tuple(self.g.N)); h = self.g.h
        def grad(axis):
            first = -np.take(phi, [0], axis); last = -np.take(phi, [-1], axis)
            return np.diff(np.concatenate([first, phi, last], axis), axis=axis) / h
        self.u = np.where(self.mu == 0, self.u + grad(0), self.u)
        self.v = np.where(self.mv == 0, self.v + grad(1), self.v)
        self.w = np.where(self.mw == 0, self.w + grad(2), self.w)
        self._enforce()

    def _enforce(self):
        self.u = np.where(self.mu == 1, 0.0, np.where(self.mu == 2, self.fu, self.u))
        self.v = np.where(self.mv == 1, 0.0, np.where(self.mv == 2, self.fv, self.v))
        self.w = np.where(self.mw == 1, 0.0, np.where(self.mw == 2, self.fw, self.w))

    def _centers(self):
        return np.stack([0.5 * (self.u[1:] + self.u[:-1]), 0.5 * (self.v[:, 1:] + self.v[:, :-1]),
                         0.5 * (self.w[:, :, 1:] + self.w[:, :, :-1])], -1)

    def _to_faces(self, U):
        u = self.u; v = self.v; w = self.w
        u[1:-1] = 0.5 * (U[1:, :, :, 0] + U[:-1, :, :, 0]); u[0] = U[0, :, :, 0]; u[-1] = U[-1, :, :, 0]
        v[:, 1:-1] = 0.5 * (U[:, 1:, :, 1] + U[:, :-1, :, 1]); v[:, 0] = U[:, 0, :, 1]; v[:, -1] = U[:, -1, :, 1]
        w[:, :, 1:-1] = 0.5 * (U[:, :, 1:, 2] + U[:, :, :-1, 2]); w[:, :, 0] = 0; w[:, :, -1] = U[:, :, -1, 2]
        self._enforce()

    # ------------------------------------------------------------------ semi-Lagrangian
    def _build_block_table(self):
        """blocked[c, code]: is the neighbour at offset (dx,dy,dz) in {-1,0,1}^3 of cell c on the
        other side of a blocked face? (static; built once)"""
        N = self.g.N; n = self.n
        own = np.stack(np.unravel_index(np.arange(n), tuple(N)), 1)
        tab = np.zeros((n, 27), bool)
        for code in range(27):
            off = np.array([code // 9 - 1, (code // 3) % 3 - 1, code % 3 - 1])
            ii = np.clip(own + off, 0, N - 1)
            blk = np.zeros(n, bool)
            for axis, mm in enumerate((self.mu, self.mv, self.mw)):
                d = ii[:, axis] - own[:, axis]
                cross = d != 0
                if not cross.any():
                    continue
                fidx = np.where(d > 0, ii[:, axis], own[:, axis])
                for base in (own, ii):
                    q = base.copy(); q[:, axis] = fidx
                    q = np.clip(q, 0, np.array(mm.shape) - 1)
                    blk |= cross & (mm[q[:, 0], q[:, 1], q[:, 2]] == 1)
            tab[:, code] = blk
        self.block_tab = tab

    def _interp_weights(self, P):
        """Trilinear weights from cell centers to points P; corners on the other side of a
        blocked face are dropped and the rest renormalized. Returns idx (n,8), w (n,8)."""
        g = self.g; N = g.N; h = g.h
        if not hasattr(self, 'block_tab'):
            self._build_block_table()
        f = (P - g.lo) / h - 0.5
        i0 = np.floor(f).astype(int); fr = f - i0
        own = np.clip(np.floor((P - g.lo) / h).astype(int), 0, N - 1)
        own_i = np.ravel_multi_index(own.T, tuple(N))
        idx = np.empty((len(P), 8), int); w = np.empty((len(P), 8))
        m = 0
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    ii = np.clip(i0 + np.array([a, b, c]), 0, N - 1)
                    wt = ((fr[:, 0] if a else 1 - fr[:, 0]) * (fr[:, 1] if b else 1 - fr[:, 1]) *
                          (fr[:, 2] if c else 1 - fr[:, 2]))
                    off = np.clip(ii - own, -1, 1) + 1
                    blk = self.block_tab[own_i, off[:, 0] * 9 + off[:, 1] * 3 + off[:, 2]]
                    idx[:, m] = np.ravel_multi_index(ii.T, tuple(N)); w[:, m] = np.where(blk, 0.0, wt)
                    m += 1
        s = w.sum(1, keepdims=True); bad = s[:, 0] == 0
        w[~bad] /= s[~bad]
        idx[bad, 0] = own_i[bad]; w[bad] = 0; w[bad, 0] = 1
        return idx, w

    def _backtrace(self, dt):
        """Departure points of all cell centers, substepped so each substep moves < h/2,
        stopped at blocked faces; points leaving through an open outer face -> far field."""
        g = self.g; N = g.N; h = g.h
        Uc = self.U.reshape(-1, 3)
        vmax = np.abs(Uc).max() + 1e-9
        ns = int(np.ceil(vmax * dt / (0.5 * h))); ns = max(ns, 1); ds = dt / ns
        P = self.X.copy(); far = np.zeros(self.n, bool)
        masks = (self.mu, self.mv, self.mw)
        for _ in range(ns):
            idx, w = self._interp_weights(np.clip(P, g.lo + 1e-9, g.hi - 1e-9))
            V = np.einsum('nm,nmk->nk', w, Uc[idx])
            Pn = P - ds * V
            c0 = np.floor((P - g.lo) / h).astype(int)
            Pn[:, 2] = np.maximum(Pn[:, 2], g.lo[2] + 1e-6)          # ground is solid
            out = (np.any((Pn[:, :2] < g.lo[:2]) | (Pn[:, :2] > g.hi[:2]), 1) | (Pn[:, 2] > g.hi[2])) & ~far
            far |= out
            c1 = np.clip(np.floor((Pn - g.lo) / h).astype(int), 0, N - 1)
            for axis, mm in enumerate(masks):
                d = c1[:, axis] - c0[:, axis]
                cr = (d != 0) & ~far
                if not cr.any():
                    continue
                fidx = np.where(d > 0, c0[:, axis] + 1, c0[:, axis])
                q = c0.copy(); q[:, axis] = fidx; q = np.clip(q, 0, np.array(mm.shape) - 1)
                blk = cr & (mm[q[:, 0], q[:, 1], q[:, 2]] == 1)
                face = g.lo[axis] + fidx * h
                Pn[blk, axis] = np.where(d[blk] > 0, face[blk] - 1e-6, face[blk] + 1e-6)
            P = np.where(far[:, None], P, Pn)
        return np.clip(P, g.lo + 1e-9, g.hi - 1e-9), far

    def _ghosts(self):
        """One layer of outside-air points beyond every open outer face (not the ground)."""
        if hasattr(self, '_ghost_X'):
            return self._ghost_X
        g = self.g; h = g.h; P = []
        for axis in range(3):
            for side in (0, 1):
                if axis == 2 and side == 0:
                    continue
                sl = self.X[:, axis] == (self.X[:, axis].min() if side == 0 else self.X[:, axis].max())
                Q = self.X[sl].copy(); Q[:, axis] += (-h if side == 0 else h); P.append(Q)
        self._ghost_X = np.concatenate(P)
        return self._ghost_X

    def _advect(self, dt):
        from m2kern import backward_gather, forward_scatter
        g = self.g; N = np.array(g.N, np.int64)
        if not hasattr(self, 'block_tab'):
            self._build_block_table()
        Uc = np.ascontiguousarray(self.U.reshape(-1, 3))
        vmax = np.abs(Uc).max() + 1e-9
        ns = min(60, max(1, int(np.ceil(vmax * dt / (0.5 * g.h)))))
        Ufar = (self.U_far[None, :] * self.prof(self.X[:, 2])[:, None]) if self.U_far.any() else np.zeros((self.n, 3))
        args = (dt, ns, g.lo, g.hi, g.h, N, self.block_tab, Uc, self.mu, self.mv, self.mw)
        if self.scheme == 'bwd':
            F = np.c_[self.T.ravel(), self.W.ravel()]
            far_vals = np.c_[np.full(self.n, self.T_out), np.full(self.n, self.W_out)]
            A = lambda Fin, sgn=-1.0: backward_gather(self.X, dt, ns, g.lo, g.hi, g.h, N, self.block_tab, Uc,
                                                       self.mu, self.mv, self.mw, self.dvu, self.dvv, self.dvw,
                                                       Fin, far_vals, self.dev_vals, sgn)[0]
            if self.bfecc:
                # BFECC (Dupont & Liu 2003; Kim et al. 2005): forward, back, correct, advect; then limit
                # to the local min/max of the old field so no new extremes appear
                F1 = A(F); F2 = A(F1, +1.0)
                Fc = F + 0.5 * (F - F2)
                out = A(Fc)
                for j, arr in enumerate((self.T, self.W)):
                    lo_ = minimum_filter(arr, size=3, mode='nearest').ravel()
                    hi_ = maximum_filter(arr, size=3, mode='nearest').ravel()
                    lo_ = np.minimum(lo_, far_vals[:, j].min()); hi_ = np.maximum(hi_, far_vals[:, j].max())
                    if len(self.supply_ids):
                        lo_ = np.minimum(lo_, self.dev_vals[:, j].min()); hi_ = np.maximum(hi_, self.dev_vals[:, j].max())
                    out[:, j] = np.clip(out[:, j], lo_, hi_)
            else:
                out = A(F)
            self.T = out[:, 0].reshape(self.T.shape); self.W = out[:, 1].reshape(self.W.shape)
        else:   # team's forward scheme (tent weights + ghost outside-air points) for heat & humidity
            Xg = self._ghosts()
            Xp = np.concatenate([self.X, Xg])
            Fp = np.concatenate([np.c_[self.T.ravel(), self.W.ravel()],
                                 np.tile([self.T_out, self.W_out], (len(Xg), 1))])
            F_old = np.c_[self.T.ravel(), self.W.ravel()]
            sc = forward_scatter(Xp, Fp, self.n, dt, ns, g.lo, g.hi, g.h, N, self.block_tab, Uc,
                                 self.mu, self.mv, self.mw, self.dvu, self.dvv, self.dvw, F_old, self.dev_vals)
            self.T = sc[:, 0].reshape(self.T.shape); self.W = sc[:, 1].reshape(self.W.shape)
        # velocity: advect each staggered component on its own face grid (no centre<->face averaging)
        from m2kern import face_gather
        if not hasattr(self, '_facepos'):
            self._facepos = []
            for axis, arr in enumerate((self.u, self.v, self.w)):
                org = g.lo + 0.5 * g.h; org = org.copy(); org[axis] = g.lo[axis]
                ax = [org[i] + np.arange(arr.shape[i]) * g.h for i in range(3)]
                XX = np.stack([q.ravel() for q in np.meshgrid(*ax, indexing='ij')], 1)
                self._facepos.append((XX, org))
        if not hasattr(self, '_nodev'):
            self._nodev = (np.zeros(self.mu.shape, np.int64), np.zeros(self.mv.shape, np.int64), np.zeros(self.mw.shape, np.int64))
        new = []
        for axis, arr in enumerate((self.u, self.v, self.w)):
            XX, org = self._facepos[axis]
            farv = self.U_far[axis] * self.prof(XX[:, 2]) if self.U_far[axis] != 0 else np.zeros(len(XX))
            Af = lambda Fin, sgn=-1.0: face_gather(XX, dt, ns, g.lo, g.hi, g.h, N, self.block_tab, Uc, self.mu, self.mv, self.mw,
                                                   self._nodev[0], self._nodev[1], self._nodev[2],
                                                   np.ascontiguousarray(Fin), org, farv, sgn).reshape(arr.shape)
            if self.bfecc_vel:
                f1 = Af(arr); f2 = Af(f1, +1.0)
                o = Af(arr + 0.5 * (arr - f2))
                lo_ = minimum_filter(arr, size=3, mode='nearest'); hi_ = maximum_filter(arr, size=3, mode='nearest')
                if self.U_far[axis] != 0:
                    lo_ = np.minimum(lo_, farv.min()); hi_ = np.maximum(hi_, farv.max())
                new.append(np.clip(o, lo_, hi_))
            else:
                new.append(Af(arr))
        self.u, self.v, self.w = new

    # ------------------------------------------------------------------ diffusion + walls
    def _lap(self, F, kind):
        """Explicit flux divergence (per unit volume) through OPEN faces only."""
        h = self.g.h; out = np.zeros_like(F)
        for axis, mm in enumerate((self.mu, self.mv, self.mw)):
            inner = [slice(None)] * 3; inner[axis] = slice(1, -1)
            open_ = (mm[tuple(inner)] != 1)
            lo = [slice(None)] * 3; lo[axis] = slice(0, -1); hi = [slice(None)] * 3; hi[axis] = slice(1, None)
            flux = (F[tuple(hi)] - F[tuple(lo)]) / h
            if F.ndim == 4:
                flux = flux * open_[..., None]
            else:
                flux = flux * open_
            out[tuple(lo)] += flux / h; out[tuple(hi)] -= flux / h
        return out

    def _lap_plain(self, F):
        h = self.g.h; P = np.pad(F, 1, mode='edge')
        return (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] + P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1]
                + P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2] - 6 * F) / (h * h)

    def _diffuse_and_walls(self, dt):
        h = self.g.h; V = h ** 3; A = h * h; rc = RHO * CP
        # explicit eddy diffusion, substepped to stay below the stability limit h^2/(6*alpha)
        nd = max(1, int(np.ceil(dt * 6 * max(self.alpha_t, self.nu_t) / (0.8 * h * h))))
        T0 = self.T.copy()
        for _ in range(nd):
            for nm in ('u', 'v', 'w'):
                setattr(self, nm, getattr(self, nm) + (dt / nd) * self.nu_t * self._lap_plain(getattr(self, nm)))
            self.T += (dt / nd) * self.alpha_t * self._lap(self.T, 'T')
            self.W += (dt / nd) * self.alpha_t * self._lap(self.W, 'W')
        Tf = T0.ravel(); dT = (self.T - T0).ravel()
        # panels: heat through U-value; roof is a thin sunlit panel between two air cells
        a, b = self.wl_a, self.wl_b
        Ta, Tb = Tf[a], Tf[b]
        I = self.cl['I_roof']
        # roof: outer (upper, b) side h_o = 15, inner side U; absorbed sun on top
        h_o = 15.0
        Tp = np.where(self.wl_roof, (self.wl_abs * I + h_o * Tb + self.wl_U * Ta) / (h_o + self.wl_U), 0.0)
        q_a = np.where(self.wl_roof, self.wl_U * (Tp - Ta), self.wl_U * (Tb - Ta)) * A   # W into cell a
        q_b = np.where(self.wl_roof, h_o * (Tp - Tb), self.wl_U * (Ta - Tb)) * A
        np.add.at(dT, a, dt * q_a / (rc * V)); np.add.at(dT, b, dt * q_b / (rc * V))
        # ground: convective exchange with the pavement surface
        h_g = 10.0
        T = (Tf + dT).reshape(self.T.shape)
        if getattr(self, 'T_floor', None) is not None:      # CAD venue: every air cell resting on a solid surface
            fl = self.floor
            T[fl] += dt * h_g * A * (self.T_floor[fl] - T[fl]) / (rc * V)
            if self.ceil is not None and self.ceil.any():   # hot roof: heat through the panel, U = roof_U W/m2K
                c_ = self.ceil
                T[c_] += dt * self.roof_U * A * (self.T_out + self.T_ceil_rise[c_] - T[c_]) / (rc * V)
        else:
            T[:, :, 0] += dt * h_g * A * (self.T_ground - T[:, :, 0]) / (rc * V)
        self.T = T

    # ------------------------------------------------------------------ devices
    def _apply_devices(self, dt):
        h = self.g.h; Vc = h ** 3; Tf = self.T.ravel(); Wf = self.W.ravel()
        self.dev_log = []
        for st in self.dev_state:
            dv = st['dev']; up, dn = st['up'], st['dn']
            if dv.kind in ('fan', 'curtain'):
                Tf[dn] += dt * dv.power_kW * 1000 / len(dn) / (RHO * CP * Vc)
                self.dev_log.append((dv.name, 0.0))
            elif dv.kind in ('evap', 'spotac'):
                # The outlet only MOVES air (intake air crosses the outlet face unchanged: dev_vals =
                # intake state). The device's effect is an exact source in its outlet cells:
                #   sensible  -rho*cp*Vdot*(T_in - T_sup)     latent  +rho*Vdot*(W_sup - W_in)
                # so the energy per step is exactly the spec output, independent of dt; in a steady
                # jet the air leaving the outlet cells is at (T_sup, W_sup).
                Ti, Wi = Tf[up].mean(), Wf[up].mean()
                if dv.kind == 'evap':
                    rh_i = float(rh_from_W(Ti, Wi)); Twb = float(wet_bulb_stull(Ti, rh_i))
                    Ts = Ti - dv.spec['eta'] * (Ti - Twb)
                    Ws = Wi + CP * (Ti - Ts) / H_FG
                else:
                    q = dv.spec['Q_kW'] * 1000 / (RHO * dv.flow)          # J/kg removed (total)
                    lo_, hi_ = Ti - 40, Ti
                    for _ in range(40):                                 # coil: leave at <= 95 % RH
                        Ts = 0.5 * (lo_ + hi_)
                        Ws = min(Wi, 0.95 * W_sat(Ts))
                        if CP * (Ti - Ts) + H_FG * (Wi - Ws) > q:
                            lo_ = Ts
                        else:
                            hi_ = Ts
                    self.heat_exhausted_J = getattr(self, 'heat_exhausted_J', 0.0) + dt * (dv.spec['Q_kW'] + dv.power_kW) * 1000
                Qs = RHO * CP * dv.flow * (Ti - Ts)                     # W sensible removed
                Mw = RHO * dv.flow * (Ws - Wi)                          # kg/s vapour added (neg = condensed)
                # this step's supply air fills L = V*dt/h cell layers downstream of the outlet: give
                # each (part-)layer the full intake->supply change, so the outlet air leaves at T_sup and
                # the total is exactly Qs*dt (sensible) and Mw*dt (moisture)
                L = abs(st['V']) * dt / h
                # place this step's cooling layer by layer along the jet; no cell is taken below the
                # supply temperature (cooling it cannot take is carried to the next layer, and any rest
                # is logged as unplaced instead of forcing cells colder than the device can make air)
                dTs = Ti - Ts
                carry = np.zeros(len(dn)); placed = 0.0
                for kk, cells in enumerate(st['layers']):
                    wk = min(max(L - kk, 0.0), 1.0)
                    need = wk * dTs + carry
                    if dTs > 0:
                        room = np.maximum(Tf[cells] - Ts, 0.0)
                        got = np.minimum(need, room)
                    else:
                        got = need
                    carry = need - got
                    Tf[cells] -= got
                    if dTs != 0:
                        Wf[cells] += got / dTs * (Ws - Wi)
                    placed += got.sum()
                    if wk <= 0 and not carry.any():
                        break
                frac = placed / (L * dTs * len(dn)) if dTs > 0 and L > 0 else 1.0
                self.placed_frac = getattr(self, 'placed_frac', []) + [frac]
                Qs = Qs * frac; Mw = Mw * frac                      # what was actually delivered
                self.dev_vals[self.supply_ids.index(self.dev_state.index(st))] = (Ti, Wi)
                self.dev_supply = getattr(self, 'dev_supply', {}); self.dev_supply[dv.name + str(self.dev_state.index(st))] = (Ts, Ws, Qs, Mw)
                if dv.kind == 'evap':
                    self.water_used += Mw * dt; self.water_evap += Mw * dt
                self.dev_log.append((dv.name, Ti - Ts))
            elif dv.kind == 'mist':
                cells = st['cells']
                m_in = dv.spec['water_kg_s'] / len(cells) * dt              # water sprayed per cell
                rh = rh_from_W(Tf[cells], Wf[cells]); Twb = wet_bulb_stull(Tf[cells], rh)
                cap = RHO * Vc * np.maximum(W_sat(Twb) - Wf[cells], 0)       # adiabatic saturation limit
                e = np.minimum(m_in, cap)
                Wf[cells] += e / (RHO * Vc); Tf[cells] -= e * H_FG / (RHO * CP * Vc)
                self.water_used += m_in * len(cells); self.water_evap += e.sum()
                self.dev_log.append((dv.name, e.sum() / (m_in * len(cells))))
        self.T = Tf.reshape(self.T.shape); self.W = Wf.reshape(self.W.shape)

    # ------------------------------------------------------------------ one step
    def _guard(self, where):
        """Catch and log any velocity spike (non-finite, or far above anything the inflow or the
        devices can produce) instead of letting it stall the tracer."""
        vcap = 3.0 * max([abs(s['V']) for s in self.dev_state] + [self.U10 * 1.5, 2.0])
        for nm in ('u', 'v', 'w'):
            a = getattr(self, nm)
            bad = ~np.isfinite(a) | (np.abs(a) > vcap)
            if bad.any():
                self.spike_log = getattr(self, 'spike_log', [])
                self.spike_log.append((round(self.time, 2), where, nm, int(bad.sum()), float(np.nanmax(np.abs(np.where(np.isfinite(a), a, 0))))))
                print('VELOCITY GUARD', self.spike_log[-1], flush=True)
                setattr(self, nm, np.clip(np.nan_to_num(a, nan=0.0, posinf=vcap, neginf=-vcap), -vcap, vcap))

    def step(self, dt):
        self._guard('start')
        self.U = self._centers()                       # cell-centred copy used for tracing
        self._advect(dt)
        self._diffuse_and_walls(dt)
        if self.extra_T.any():
            self.T += dt * self.extra_T / (RHO * CP * self.g.h ** 3)
            self.W += dt * self.extra_W / (RHO * self.g.h ** 3)
        if getattr(self, 'src_T', None) is not None:     # persistent sources (kiosk / grill heat) [W/cell]
            self.T += dt * self.src_T / (RHO * CP * self.g.h ** 3)
        if getattr(self, 'src_W', None) is not None:     # persistent moisture sources (seated crowd) [kg/s per cell]
            self.W += dt * self.src_W / (RHO * self.g.h ** 3)
        self._apply_devices(dt)
        if self.buoy:                                  # Boussinesq buoyancy on w-faces (virtual temp.)
            Tv = (self.T + 273.15) * (1 + 0.61 * self.W)
            Tv_ref = (self.T_out + 273.15) * (1 + 0.61 * self.W_out)
            bz = G_ACC * (Tv - Tv_ref) / Tv_ref
            self.w[:, :, 1:-1] += dt * 0.5 * (bz[:, :, 1:] + bz[:, :, :-1])
        self._enforce()
        self._guard('pre-projection')
        self._project()
        self.U = self._centers()
        self.time += dt

    # ------------------------------------------------------------------ helpers
    def interior(self):
        (x0, x1), (y0, y1), (z0, z1) = self.g.box
        X = self.X
        return ((X[:, 0] > x0) & (X[:, 0] < x1) & (X[:, 1] > y0) & (X[:, 1] < y1) & (X[:, 2] < z1)).reshape(tuple(self.g.N))

    def occupied(self):
        if getattr(self.g, 'occ_mask', None) is not None:
            return self.g.occ_mask
        z = self.X[:, 2].reshape(tuple(self.g.N))
        return self.interior() & (z > self.g.occ_z[0]) & (z < self.g.occ_z[1])

    def speed(self):
        return np.linalg.norm(self.U, axis=-1)

    def utci_field(self):
        return utci_full(self.T, self.Tmrt, rh_from_W(self.T, self.W), self.speed())

    def max_div(self):
        return np.abs(self._divergence()).max()
