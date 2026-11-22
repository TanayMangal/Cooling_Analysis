"""
Model 2: Model 1 + wind.
  * Wind field: Stable-Fluids solve around/inside the shelter (flow.py) with outside wind
    (log-law inflow from 3 directions) and the cooler/fan jets.
  * Heat and humidity: semi-Lagrangian advection on that wind (advect.py) + eddy diffusion,
    written as one linear operator so the transfer-matrix calibration still applies.
  * Comfort: FULL 4-variable UTCI (local speed converted to 10 m), weighted over the
    17 JOS-3 segments by the Zhang et al. (2009) cooling slopes (as in Model 1).
  * Bodies: JOS-3 driven with the LOCAL air speed of each segment's cell (this is the
    "wind chill" on body heat: convective + evaporative loss rise with speed), returning
    sensible and latent heat to those cells.
"""
import sys, os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shelter_model'))
from shelter_model import (W_from_rh, rh_from_W, wet_bulb_stull, seg_weights, SEGMENTS,
                           SEG_POS, RHO, CP, H_FG, SIGMA)
from pythermalcomfort.models import JOS3, utci
from flow import Domain, SH, UTCI_V10
from advect import advection_matrix

CLIMATES = {
    # T/RH/sun as Model 1 [verify vs ASHRAE]; wind = July mean, NOAA via currentresults.com
    'Phoenix (hot-dry)': dict(T_out=43.0, rh_out=13.0, I_dir=850.0, I_roof=950.0, U10=3.17),
    'Miami (hot-humid)': dict(T_out=33.0, rh_out=55.0, I_dir=700.0, I_roof=850.0, U10=3.53),
    'Reference (mild)':  dict(T_out=30.0, rh_out=50.0, I_dir=0.0,   I_roof=0.0,   U10=0.0),
}
SYSTEMS = {
    'None':         dict(kind='none', P_kW=0.0,  dT=0.0),
    'A Fan':        dict(kind='fan',  P_kW=0.25, dT=3.0, x=(1.2, 1.8), z=(1.75, 2.25)),
    'B Evap':       dict(kind='evap', P_kW=1.2,  dT=5.0, x=(1.2, 1.8), z=(1.75, 2.25)),
    'C Large evap': dict(kind='evap', P_kW=2.5,  dT=8.0, x=(0.9, 2.1), z=(1.75, 2.25)),
}
U_WALL = U_ROOF = 5.0; H_OUT = 15.0; ROOF_ABS = 0.6
ALPHA_T = 0.02                 # background eddy diffusivity [m2/s], same as Model 1


# ------------------------------------------------------------------ diffusion operators
def diffusion_operator(dom, kind='T', T_far=0.0, T_solair=0.0, alpha_t=ALPHA_T):
    """kind 'T': W/K energy operator with wall U-values and roof sol-air;
    'W': same eddy part, walls impermeable; 'momentum': plain Laplacian (1/m2)."""
    G = dom.G; Nx, Ny, Nz = dom.N; d = dom.d
    n = G.n; I = lambda i, j, k: (i * Ny + j) * Nz + k
    K = RHO * CP * alpha_t
    area = [d[1] * d[2], d[0] * d[2], d[0] * d[1]]
    rows, cols, vals = [], [], []; diag = np.zeros(n); b = np.zeros(n)
    for axis, mask in enumerate((dom.mu, dom.mv, dom.mw)):
        A = area[axis]
        for idx in np.argwhere(np.ones_like(mask, bool)):
            m = mask[tuple(idx)]
            a = idx.copy(); a[axis] -= 1; bb = idx.copy()
            lo_in, hi_in = a[axis] >= 0, bb[axis] < dom.N[axis]
            if kind == 'momentum':
                g = 1.0 / d[axis] ** 2 if (m != 1 and lo_in and hi_in) else 0.0
                gb = 0.0
            else:
                open_g = K * A / d[axis]
                if m == 1:                                   # solid wall face
                    if kind == 'W':
                        g = 0.0
                    elif axis == 2 and idx[2] == 0:
                        g = 0.0                              # ground: adiabatic
                    elif axis == 2:                          # roof: interior cell <-> sol-air
                        cell = I(*a)
                        Gr = 1.0 / ((d[2] / 2) / (K * A) + 1.0 / (U_ROOF * A))
                        diag[cell] += Gr; b[cell] += Gr * T_solair; g = 0.0
                    else:                                    # back/side wall, air both sides
                        g = 1.0 / (d[axis] / (K * A) + 1.0 / (U_WALL * A))
                else:
                    g = open_g
                gb = 0.0
                if not (lo_in and hi_in) and m != 1:          # outer face (not ground): far field
                    gb = K * A / (d[axis] / 2)
            if lo_in and hi_in and g > 0:
                p, q = I(*a), I(*bb)
                rows += [p, q]; cols += [q, p]; vals += [-g, -g]; diag[p] += g; diag[q] += g
            elif gb > 0:
                cell = I(*a) if lo_in else I(*bb)
                diag[cell] += gb; b[cell] += gb * T_far
    rows += list(range(n)); cols += list(range(n)); vals += list(diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n)), b


def utci_full(Ta, Tmrt, rh, speed):
    v10 = np.clip(speed * UTCI_V10, 0.5, 17.0)
    Ta = np.clip(Ta, -50, 50); dT = np.clip(Tmrt - Ta, -30, 70)
    return np.asarray(utci(tdb=Ta, tr=Ta + dT, v=v10, rh=rh, limit_inputs=False,
                           round_output=False).utci, float)


# ------------------------------------------------------------------ scenario
class Scenario2:
    def __init__(self, climate, system, direction='front', scheme='bwd_tri', dt_s=0.1,
                 n_people=6, wait_min=15, alpha_t=ALPHA_T, eta_pad=0.8, sfp=0.15,
                 c1=0.14, c2=0.002, fan_V=None, wind_scale=1.0, verbose=False):
        self.cl = dict(CLIMATES[climate]); self.sys = SYSTEMS[system]
        self.climate, self.system, self.direction = climate, system, direction
        self.scheme, self.dt_s, self.n_people, self.wait_min = scheme, dt_s, n_people, wait_min
        self.alpha_t, self.eta_pad, self.sfp, self.c1, self.c2 = alpha_t, eta_pad, sfp, c1, c2
        self.fan_V, self.verbose = fan_V, verbose
        self.U10 = self.cl['U10'] * wind_scale
        self.T_out, self.rh_out = self.cl['T_out'], self.cl['rh_out']
        self.W_out = W_from_rh(self.T_out, self.rh_out)
        Twb = wet_bulb_stull(self.T_out, self.rh_out)
        self.T_sup = self.T_out - eta_pad * (self.T_out - Twb)
        self.W_sup = self.W_out + CP * (self.T_out - self.T_sup) / H_FG
        self.T_solair = self.T_out + ROOF_ABS * self.cl['I_roof'] / H_OUT
        self.Vmax = self.sys['P_kW'] * 1000 / sfp / 3600 if self.sys['kind'] == 'evap' else 0.0

    # ---------------- wind field for a given jet speed
    def make_domain(self, Vjet):
        s = self.sys
        jets = []
        if s['kind'] == 'evap' and Vjet > 0:
            jets = [dict(kind='supply', x=s['x'], z=s['z'], V=Vjet)]
        if s['kind'] == 'fan' and Vjet > 0:
            jets = [dict(kind='fan', x=s['x'], z=s['z'], V=Vjet)]
        dom = Domain(self.direction if self.U10 > 0 else None, self.U10, jets=jets)
        dom.solve_steady(verbose=self.verbose)
        return dom

    # ---------------- radiation (same two-state model as Model 1, inside shelter only)
    def tmrt(self, dom):
        X = dom.G.X; inside = dom.interior_mask()
        z_exit = X[:, 2] + (SH['y1'] - X[:, 1]) * np.tan(np.radians(40))
        sunlit = ~inside | (z_exit < SH['H'])
        Tsh = self.T_out + 3.0 + 273.15
        Tsun = (Tsh ** 4 + 0.25 * 0.7 * self.cl['I_dir'] / (0.97 * SIGMA)) ** 0.25
        return np.where(sunlit, Tsun, Tsh) - 273.15

    # ---------------- linear scalar operators for this wind field
    def operators(self, dom, dt):
        S, B = advection_matrix(dom.G, dom.centers(), dt, self.scheme)
        A_T, b_T = diffusion_operator(dom, 'T', self.T_out, self.T_solair, self.alpha_t)
        A_W, b_W = diffusion_operator(dom, 'W', self.W_out, 0.0, self.alpha_t)
        M = RHO * CP * np.prod(dom.d) / dt
        bcT = np.array([self.T_out] + [self.T_sup] * (B.shape[1] - 1))
        bcW = np.array([self.W_out] + [self.W_sup] * (B.shape[1] - 1))
        return dict(S=S, B=B, A_T=A_T, b_T=b_T, A_W=A_W, b_W=b_W, M=M, bcT=bcT, bcW=bcW)

    def steady(self, ops, sT=None, sW=None):
        """Equilibrium of one SL+diffusion step:  ((M I + A) - M S) T = M B bc + b + s."""
        n = ops['S'].shape[0]; I = sp.identity(n)
        K_T = (ops['M'] * I + ops['A_T'] - ops['M'] * ops['S']).tocsc()
        K_W = (ops['M'] * I + ops['A_W'] - ops['M'] * ops['S']).tocsc()
        sT = np.zeros(n) if sT is None else sT; sW = np.zeros(n) if sW is None else sW
        T = spla.spsolve(K_T, ops['M'] * (ops['B'] @ ops['bcT']) + ops['b_T'] + sT)
        W = spla.spsolve(K_W, ops['M'] * (ops['B'] @ ops['bcW']) + ops['b_W'] + sW)
        return T, W

    def fan_heat(self, dom):
        s = self.sys; X = dom.G.X
        if s['kind'] != 'fan':
            return np.zeros(dom.G.n)
        m = ((X[:, 0] >= s['x'][0]) & (X[:, 0] <= s['x'][1]) & (X[:, 2] >= s['z'][0])
             & (X[:, 2] <= s['z'][1]) & (X[:, 1] > SH['y0']) & (X[:, 1] < SH['y0'] + 0.3))
        return s['P_kW'] * 1000 * m / m.sum()

    def comfort_field(self, dom, T, W):
        speed = np.linalg.norm(dom.centers(), axis=1)
        return utci_full(T, self.Tmrt, rh_from_W(T, W), speed)

    # ---------------- calibration -> equilibrium (relative t = 0)
    def calibrate(self, passive_T_mean=None):
        s = self.sys
        self.feasible = True; self.Vjet = 0.0; self.Vdot = 0.0
        if s['kind'] == 'evap':
            if passive_T_mean is None:
                d0 = self.make_domain(0.0); o0 = self.operators(d0, self.dt_s)
                passive_T_mean = self.steady(o0)[0][d0.interior_mask()].mean()
            self.T_passive = passive_T_mean
            area = (s['x'][1] - s['x'][0]) * (s['z'][1] - s['z'][0])
            cache = {}
            def drop(Vdot):
                dom = self.make_domain(Vdot / area); ops = self.operators(dom, self.dt_s)
                T, W = self.steady(ops); cache[Vdot] = (dom, ops, T, W)
                return passive_T_mean - T[dom.interior_mask()].mean() - s['dT']
            if drop(self.Vmax) < 0:
                self.Vdot, self.feasible = self.Vmax, False
            else:
                self.Vdot = brentq(drop, 0.02, self.Vmax, xtol=0.01, maxiter=30)
                if self.Vdot not in cache:
                    drop(self.Vdot)
            self.Vjet = self.Vdot / area
            self.dom, self.ops, self.T0, self.W0 = cache[self.Vdot]
            self.water_kg_s = RHO * self.Vdot * (self.W_sup - self.W_out)
            self.dT_achieved = passive_T_mean - self.T0[self.dom.interior_mask()].mean()
        else:
            V = (self.fan_V or 0.0) if s['kind'] == 'fan' else 0.0
            self.Vjet = V
            self.dom = self.make_domain(V); self.ops = self.operators(self.dom, self.dt_s)
            self.water_kg_s = 0.0
            self.T0, self.W0 = self.steady(self.ops, sT=self.fan_heat(self.dom))
            self.dT_achieved = np.nan
        self.Tmrt = self.tmrt(self.dom)
        return self

    # ---------------- people
    def people_positions(self):
        xs = np.linspace(0.4, SH['x1'] - 0.4, self.n_people)
        ys = np.where(np.arange(self.n_people) % 2 == 0, 0.45, 0.95)
        return list(zip(xs, ys))

    def cell_of(self, x, y, z):
        G = self.dom.G
        ijk = np.clip(((np.array([x, y, z]) - G.o) / G.d).astype(int), 0, G.N - 1)
        return G.idx(*ijk)

    def run(self, passive_T_mean=None):
        self.calibrate(passive_T_mean)
        dom, ops = self.dom, self.ops; n = dom.G.n
        I = sp.identity(n); M = ops['M']
        lu_T = spla.splu((M * I + ops['A_T']).tocsc()); lu_W = spla.splu((M * I + ops['A_W']).tocsc())
        S, B = ops['S'], ops['B']; bT = M * (B @ ops['bcT']) + ops['b_T']; bW = M * (B @ ops['bcW']) + ops['b_W']
        sfan = self.fan_heat(dom)
        T, W = self.T0.copy(), self.W0.copy()
        speed = np.linalg.norm(dom.centers(), axis=1)
        w = seg_weights()
        Tmrt_sun = self.tmrt(dom).max()
        people = []
        for (x, y) in self.people_positions():
            b = JOS3(height=1.72, weight=70)
            b.tdb, b.tr, b.rh, b.v, b.par = self.T_out, Tmrt_sun, self.rh_out, max(self.U10 * 0.55, 0.1), 1.6
            b.simulate(times=20, dtime=60); b.par = 1.2
            cells = np.array([self.cell_of(x + SEG_POS[s][0], y + SEG_POS[s][1], SEG_POS[s][2])
                              for s in SEGMENTS])
            people.append((b, cells))
        body_dt = 30.0; sub = int(body_dt / self.dt_s); steps = int(self.wait_min * 60 / body_dt)
        hist = []; srcT = np.zeros(n); srcW = np.zeros(n)
        for m in range(steps + 1):
            C = self.comfort_field(dom, T, W)
            Cp = [float(w @ C[c]) for _, c in people]
            hist.append(dict(t=m * body_dt / 60, C=np.mean(Cp), T_in=T[dom.interior_mask()].mean(),
                             Tsk=np.mean([b.t_skin_mean for b, _ in people])))
            if m == steps:
                break
            srcT[:] = sfan; srcW[:] = 0; rh = rh_from_W(T, W)
            for b, c in people:
                v_loc = np.maximum(speed[c], 0.1)
                b.tdb, b.tr, b.rh, b.v = T[c], self.Tmrt[c], rh[c], v_loc
                b.simulate(times=1, dtime=body_dt)
                d = b.dict_results()
                hc = 8.3 * v_loc ** 0.6                                   # forced convection (ISO 7730 form)
                hc = np.maximum(hc, 3.1)
                q_conv = hc * np.array(b.bsa).ravel() * (np.array(b.t_skin) - T[c])
                q_conv[0] += d['q_res_sensible'][-1]
                lat = np.array([d[f'q_skin2env_latent_{s}'][-1] for s in SEGMENTS]); lat[0] += d['q_res_latent'][-1]
                np.add.at(srcT, c, q_conv); np.add.at(srcW, c, lat / H_FG * CP)
            for _ in range(sub):
                T = lu_T.solve(M * (S @ T) + bT + srcT)
                W = lu_W.solve(M * (S @ W) + bW + srcW)
        self.hist = hist; self.C_mean = np.mean([h['C'] for h in hist])
        self.E_kWh = self.sys['P_kW']; self.W_L = self.water_kg_s * 3600
        self.cost = self.c1 * self.E_kWh + self.c2 * self.W_L
        inside = dom.interior_mask()
        self.speed_occ = speed[inside & (dom.G.X[:, 2] > 0.5) & (dom.G.X[:, 2] < 1.75)].mean()
        return self


def efficiency(base, sc):
    return np.nan if sc.cost == 0 else (base.C_mean - sc.C_mean) / sc.cost
