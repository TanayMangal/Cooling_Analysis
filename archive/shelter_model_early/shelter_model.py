"""
HiMCM 2026 Problem B, Part 2a: bus-shelter cooling model.

Pipeline
  1. Discretize the shelter into Nx*Ny*Nz finite-volume cells (default 10x10x10).
  2. Assemble the linear operator A for steady heat (T) and vapor (W) transport:
         A T = b_bc + s      ->   T = A^-1 b_bc + A^-1 s
     A^-1 is the transfer (influence) matrix: column j = response of every cell
     to a unit source in cell j.
  3. Calibrate each cooler: one solve with a unit-strength source gives the
     mean-temperature drop per watt, so Q = dT_target / (mean response per W).
     Evaporative coolers are capped by the wet-bulb limit.
  4. Steady cooled, empty shelter = equilibrium = relative t = 0.
  5. People arrive at t = 0: JOS-3 (17-segment) bodies put sensible and latent
     heat into the cell each segment sits in; the grid is stepped (implicit Euler)
     and JOS-3 is stepped with the local conditions of each segment's cell.
  6. Comfort = no-wind UTCI per cell (v fixed at 0.5 m/s), weighted over the
     17 segments by the Zhang et al. (2009) local-cooling slopes.
  7. Efficiency  eps = (C_baseline - C_i) / (c1*E + c2*W).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pythermalcomfort.models import JOS3, utci

# ---------------------------------------------------------------- constants
RHO, CP = 1.15, 1007.0          # air density [kg/m3], specific heat [J/kgK]
H_FG = 2.43e6                   # latent heat of vaporization [J/kg]
P_ATM = 101325.0
SIGMA = 5.67e-8
V_REF = 0.5                     # UTCI wind frozen at its minimum valid value [m/s]

# ---------------------------------------------------------------- psychrometrics
def p_sat(T):                   # Pa, Magnus-Tetens
    return 610.94 * np.exp(17.625 * T / (T + 243.04))

def W_from_rh(T, rh):           # humidity ratio [kg/kg]
    pv = rh / 100 * p_sat(T)
    return 0.622 * pv / (P_ATM - pv)

def rh_from_W(T, W):
    pv = W * P_ATM / (0.622 + W)
    return np.clip(100 * pv / p_sat(T), 1, 100)

def wet_bulb_stull(T, rh):      # Stull (2011) empirical wet bulb [C]
    return (T * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(T + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)

# ---------------------------------------------------------------- body weights
# Zhang, Arens, Huizenga, Han (2009) Building & Environment, Part III, Table 2:
# slope 'a' of overall sensation vs local sensation when that part is COOLED.
ZHANG_A_COOL = dict(pelvis=0.94, back=0.91, chest=0.91, thigh=0.81, leg=0.70,
                    face=0.70, neck=0.65, head=0.54, foot=0.50, upper_arm=0.43,
                    lower_arm=0.37, breath=0.27, hand=0.25)
SEGMENTS = ['head', 'neck', 'chest', 'back', 'pelvis',
            'left_shoulder', 'left_arm', 'left_hand',
            'right_shoulder', 'right_arm', 'right_hand',
            'left_thigh', 'left_leg', 'left_foot',
            'right_thigh', 'right_leg', 'right_foot']
# JOS-3 segment -> Zhang part (JOS-3 'head' includes the face: average head & face)
SEG2ZHANG = dict(head=(ZHANG_A_COOL['head'] + ZHANG_A_COOL['face']) / 2,
                 neck=ZHANG_A_COOL['neck'], chest=ZHANG_A_COOL['chest'],
                 back=ZHANG_A_COOL['back'], pelvis=ZHANG_A_COOL['pelvis'],
                 shoulder=ZHANG_A_COOL['upper_arm'], arm=ZHANG_A_COOL['lower_arm'],
                 hand=ZHANG_A_COOL['hand'], thigh=ZHANG_A_COOL['thigh'],
                 leg=ZHANG_A_COOL['leg'], foot=ZHANG_A_COOL['foot'])

def seg_weights(uniform=False):
    w = np.array([1.0 if uniform else SEG2ZHANG[s.replace('left_', '').replace('right_', '')]
                  for s in SEGMENTS])
    return w / w.sum()

# segment centroid offsets for a standing 1.72 m adult: (dx [m], dy [m], z [m])
SEG_POS = dict(head=(0, 0, 1.60), neck=(0, 0, 1.47), chest=(0, 0.08, 1.30),
               back=(0, -0.08, 1.30), pelvis=(0, 0, 0.95),
               left_shoulder=(-0.20, 0, 1.30), left_arm=(-0.22, 0, 1.00),
               left_hand=(-0.22, 0, 0.78), right_shoulder=(0.20, 0, 1.30),
               right_arm=(0.22, 0, 1.00), right_hand=(0.22, 0, 0.78),
               left_thigh=(-0.10, 0, 0.65), left_leg=(-0.10, 0, 0.30),
               left_foot=(-0.10, 0.05, 0.05), right_thigh=(0.10, 0, 0.65),
               right_leg=(0.10, 0, 0.30), right_foot=(0.10, 0.05, 0.05))

# ---------------------------------------------------------------- scenario data
CLIMATES = {
    # [verify] against ASHRAE 0.4% design values before final paper
    'Phoenix (hot-dry)':  dict(T_out=43.0, rh_out=13.0, I_dir=850.0, I_roof=950.0),
    'Miami (hot-humid)':  dict(T_out=33.0, rh_out=55.0, I_dir=700.0, I_roof=850.0),
}
SYSTEMS = {
    'None':  dict(kind='none',  P_kW=0.0,  dT=0.0),
    'A Fan': dict(kind='fan',   P_kW=0.25, dT=3.0),
    'B Evap': dict(kind='evap', P_kW=1.2,  dT=5.0),
    'C Large evap': dict(kind='evap', P_kW=2.5, dT=8.0),
}


class Shelter:
    """Open-front bus shelter. x: length, y: depth (y=0 back wall, y=Ly open front),
    z: height (z=0 ground, z=Lz roof)."""

    def __init__(self, Lx=3.0, Ly=1.5, Lz=2.5, N=(10, 10, 10), alpha_t=0.02,
                 U_wall=5.0, U_roof=5.0, h_out=15.0, roof_absorptance=0.6,
                 open_faces=('front',), sun_elev_deg=40.0, dTmrt_shade=3.0,
                 f_p=0.25, a_k=0.7, eps_p=0.97):
        self.L = np.array([Lx, Ly, Lz]); self.N = np.array(N)
        self.d = self.L / self.N; self.V = np.prod(self.d)
        self.alpha_t = alpha_t                     # effective (eddy) diffusivity [m2/s]
        self.K = RHO * CP * alpha_t                # effective conductivity [W/mK]
        self.U_wall, self.U_roof, self.h_out = U_wall, U_roof, h_out
        self.roof_abs = roof_absorptance
        self.open_faces = set(open_faces)
        self.sun_elev = np.radians(sun_elev_deg)
        self.dTmrt_shade, self.f_p, self.a_k, self.eps_p = dTmrt_shade, f_p, a_k, eps_p
        Nx, Ny, Nz = self.N
        self.n = Nx * Ny * Nz
        g = np.meshgrid(*[(np.arange(N[i]) + 0.5) * self.d[i] for i in range(3)], indexing='ij')
        self.xc, self.yc, self.zc = [a.ravel() for a in g]

    def idx(self, i, j, k):
        return (i * self.N[1] + j) * self.N[2] + k

    def cell_of(self, x, y, z):
        ijk = np.clip((np.array([x, y, z]) / self.d).astype(int), 0, self.N - 1)
        return self.idx(*ijk)

    # ------------------------------------------------------------ operators
    def assemble(self, T_out, I_roof, field='T'):
        """Return (A, b) with A x = b + s for x = T (field='T') or W (field='W').
        Units: rows are energy balances [W] for T; mass balances scaled by H_FG*... for W
        we use the same diffusivity so A_W = A_T with different boundary handling."""
        Nx, Ny, Nz = self.N; dx, dy, dz = self.d
        rows, cols, vals = [], [], []
        diag = np.zeros(self.n); b = np.zeros(self.n)
        area = {0: dy * dz, 1: dx * dz, 2: dx * dy}
        # interior faces
        for axis in range(3):
            G = self.K * area[axis] / self.d[axis]
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        nb = [i, j, k]; nb[axis] += 1
                        if nb[axis] >= self.N[axis]:
                            continue
                        p, q = self.idx(i, j, k), self.idx(*nb)
                        rows += [p, q]; cols += [q, p]; vals += [-G, -G]
                        diag[p] += G; diag[q] += G
        # boundary faces
        T_solair = T_out + self.roof_abs * I_roof / self.h_out   # sol-air temp of roof
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    p = self.idx(i, j, k)
                    faces = []
                    if i == 0: faces.append(('left', 0))
                    if i == Nx - 1: faces.append(('right', 0))
                    if j == 0: faces.append(('back', 1))
                    if j == Ny - 1: faces.append(('front', 1))
                    if k == 0: faces.append(('ground', 2))
                    if k == Nz - 1: faces.append(('roof', 2))
                    for name, axis in faces:
                        A_f = area[axis]; half = self.K * A_f / (self.d[axis] / 2)
                        if name in self.open_faces or (name in ('left', 'right') and 'sides' in self.open_faces):
                            # open face: Dirichlet, ambient air at the face (T_out or W_out)
                            G = half; diag[p] += G
                            b[p] += G * (T_out if field == 'T' else 1.0)  # W handled by scaling
                        elif name == 'ground':
                            pass                                    # adiabatic / impermeable
                        elif field == 'T':
                            U = self.U_roof if name == 'roof' else self.U_wall
                            G = 1.0 / (1.0 / half + 1.0 / (U * A_f))   # series: air -> panel -> outside
                            diag[p] += G
                            b[p] += G * (T_solair if name == 'roof' else T_out)
                        # field 'W': solid panels impermeable -> no term
        rows += list(range(self.n)); cols += list(range(self.n)); vals += list(diag)
        A = sp.csc_matrix((vals, (rows, cols)), shape=(self.n, self.n))
        return A, b

    # ------------------------------------------------------------ radiation field
    def tmrt_field(self, T_out, I_dir):
        """Mean radiant temperature per cell. Sun shines in through the open front
        (worst case: afternoon sun facing the opening). Cell is sunlit if the ray
        toward the sun leaves through the open front below the roof edge."""
        z_exit = self.zc + (self.L[1] - self.yc) * np.tan(self.sun_elev)
        sunlit = z_exit < self.L[2]
        Tsh = T_out + self.dTmrt_shade + 273.15
        Tsun = (Tsh ** 4 + self.f_p * self.a_k * I_dir / (self.eps_p * SIGMA)) ** 0.25
        return np.where(sunlit, Tsun, Tsh) - 273.15, sunlit


def throw_weights(sh, L_throw=1.0):
    """Where a back-wall device (cooler or fan) delivers its air: a 1.2 m wide band
    centered in x, 0.75-2.25 m high, decaying exp(-y/L_throw) toward the open front."""
    band = (np.abs(sh.xc - sh.L[0] / 2) <= 0.6) & (sh.zc >= 0.75) & (sh.zc <= 2.25)
    w = np.where(band, np.exp(-sh.yc / L_throw), 0.0)
    return w / w.sum()


def fan_climate_factor(T, rh, Tmrt, v_fan=2.0, ref=(30.0, 50.0, 33.0)):
    """OPTIONAL sensitivity check (off by default): scale the rated -3 C by how much
    extra air speed helps in THIS climate relative to a mild reference day, using the
    full 4-variable UTCI. Negative in very hot-dry air (fan heats you)."""
    def benefit(Ta, r, tr):
        u = lambda v: float(utci(tdb=Ta, tr=tr, v=v, rh=r, limit_inputs=False).utci)
        return u(V_REF) - u(v_fan)
    return benefit(T, rh, Tmrt) / benefit(*ref)


def fan_offset_field(sh, mean_offset=3.0, L_decay=1.0, interpretation='mean'):
    """Fan comfort offset (subtracted from UTCI). Exponential decay from the fan at
    the top-back-center. 'mean': scaled so the volume-average offset = 3 C;
    'peak': max offset = 3 C (literal 'up to')."""
    src = np.array([sh.L[0] / 2, 0.0, sh.L[2] - 0.3])
    d = np.sqrt((sh.xc - src[0]) ** 2 + (sh.yc - src[1]) ** 2 + (sh.zc - src[2]) ** 2)
    f = np.exp(-d / L_decay)
    f = f / (f.mean() if interpretation == 'mean' else f.max())
    return mean_offset * f


def utci_nowind(Ta, Tmrt, rh):
    Ta = np.clip(Ta, -50, 50)
    dT = np.clip(Tmrt - Ta, -30, 70)
    return np.asarray(utci(tdb=Ta, tr=Ta + dT, v=np.full_like(Ta, V_REF), rh=rh,
                           limit_inputs=False, round_output=False).utci, dtype=float)


# ---------------------------------------------------------------- the model
class Scenario:
    def __init__(self, climate, system, sh=None, n_people=6, wait_min=15,
                 dt=30.0, uniform_weights=False, fan_interp='mean', eta_pad=0.8,
                 c1=0.14, c2=0.002, sfp=0.15, fan_climate=False):
        self.sh = sh or Shelter()
        self.cl = CLIMATES[climate]; self.sys = SYSTEMS[system]
        self.climate, self.system = climate, system
        self.n_people, self.wait_min, self.dt = n_people, wait_min, dt
        self.w_seg = seg_weights(uniform_weights)
        self.fan_interp, self.eta_pad, self.c1, self.c2 = fan_interp, eta_pad, c1, c2
        self.sfp, self.fan_climate = sfp, fan_climate   # sfp: W per (m3/h) of cooler airflow
        sh = self.sh
        self.T_out, self.rh_out = self.cl['T_out'], self.cl['rh_out']
        self.W_out = W_from_rh(self.T_out, self.rh_out)
        self.A_T, self.b_T = sh.assemble(self.T_out, self.cl['I_roof'], 'T')
        self.A_W, bW = sh.assemble(self.T_out, 0, 'W')
        self.b_W = bW * self.W_out          # Dirichlet value scaled to W_out
        self.lu_T = spla.splu(self.A_T); self.lu_W = spla.splu(self.A_W)
        self.Tmrt, self.sunlit = sh.tmrt_field(self.T_out, self.cl['I_dir'])

    # -------- evaporative cooler as an outdoor-air supply (Robin sink)
    def _supply_state(self):
        Twb = wet_bulb_stull(self.T_out, self.rh_out)
        T_sup = self.T_out - self.eta_pad * (self.T_out - Twb)          # pad exit temperature
        W_sup = self.W_out + CP * (self.T_out - T_sup) / H_FG           # adiabatic: sensible -> latent
        return T_sup, W_sup, Twb

    def _with_supply(self, Vdot):
        """Operators with the cooler supplying Vdot [m3/s] of pad-cooled air into the
        throw cells: rho*cp*Vdot*phi_i*(T_sup - T_i) per cell (same form for W)."""
        G = RHO * CP * Vdot * self.phi
        A_T = (self.A_T + sp.diags(G)).tocsc(); b_T = self.b_T + G * self.T_sup
        A_W = (self.A_W + sp.diags(G)).tocsc(); b_W = self.b_W + G * self.W_sup
        return A_T, b_T, A_W, b_W

    # -------- step 3: calibration via the transfer matrix (equilibrium = relative t = 0)
    def calibrate(self):
        from scipy.optimize import brentq
        sh, s = self.sh, self.sys
        self.T_base = self.lu_T.solve(self.b_T)           # passive shelter (roof heating only)
        self.W_base = self.lu_W.solve(self.b_W)
        self.s_T = np.zeros(sh.n); self.fan = np.zeros(sh.n)
        self.Vdot = 0.0; self.Q = 0.0; self.water_kg_s = 0.0; self.feasible = True
        self.A_Tc, self.b_Tc, self.A_Wc, self.b_Wc = self.A_T, self.b_T, self.A_W, self.b_W
        if s['kind'] == 'evap':
            self.phi = throw_weights(sh)
            self.T_sup, self.W_sup, self.Twb = self._supply_state()
            Vmax = s['P_kW'] * 1000 / self.sfp / 3600      # m3/s the fan can move at rated power
            def drop(V):
                A_T, b_T, _, _ = self._with_supply(V)
                return self.T_base.mean() - spla.spsolve(A_T, b_T).mean() - s['dT']
            if drop(Vmax) < 0:                             # target unreachable: wet-bulb / airflow limit
                self.Vdot, self.feasible = Vmax, False
            else:
                self.Vdot = brentq(drop, 1e-6, Vmax, xtol=1e-6)
            self.A_Tc, self.b_Tc, self.A_Wc, self.b_Wc = self._with_supply(self.Vdot)
            self.water_kg_s = RHO * self.Vdot * (self.W_sup - self.W_out)
            self.Q = RHO * CP * self.Vdot * (self.T_out - self.T_sup)  # sensible cooling made at the pad
        elif s['kind'] == 'fan':
            u = throw_weights(sh)
            self.s_T = s['P_kW'] * 1000 * u                 # motor heat ends up in the air
            self.fan = fan_offset_field(sh, s['dT'], interpretation=self.fan_interp)
            if self.fan_climate:
                self.fan = self.fan * fan_climate_factor(self.T_out, self.rh_out,
                                                         self.T_out + sh.dTmrt_shade)
        self.lu_Tc = spla.splu(self.A_Tc); self.lu_Wc = spla.splu(self.A_Wc)
        self.T0 = self.lu_Tc.solve(self.b_Tc + self.s_T)
        self.W0 = self.lu_Wc.solve(self.b_Wc)
        self.dT_achieved = self.T_base.mean() - self.T0.mean() if s['kind'] == 'evap' else self.fan.mean()
        return self

    def transfer_matrix_columns(self, cells):
        """Columns of A^-1 (cooled operator): response of every cell to 1 W in each given cell."""
        E = np.zeros((self.sh.n, len(cells))); E[cells, range(len(cells))] = 1.0
        return self.lu_Tc.solve(E)

    # -------- comfort
    def comfort_field(self, T, W):
        rh = rh_from_W(T, W)
        return utci_nowind(T, self.Tmrt, rh) - self.fan

    # -------- step 5: transient with people
    def people_positions(self):
        sh = self.sh
        xs = np.linspace(0.4, sh.L[0] - 0.4, self.n_people)
        ys = np.where(np.arange(self.n_people) % 2 == 0, 0.45, 0.95)
        return list(zip(xs, ys))

    def run(self, verbose=False):
        sh = self.sh
        self.calibrate()
        M = RHO * CP * sh.V / self.dt                       # thermal mass per cell / dt
        Ad_T = (self.A_Tc + sp.identity(sh.n) * M).tocsc(); lu_dT = spla.splu(Ad_T)
        Ad_W = (self.A_Wc + sp.identity(sh.n) * M).tocsc(); lu_dW = spla.splu(Ad_W)
        T, W = self.T0.copy(), self.W0.copy()
        # people: pre-condition 20 min outdoors in the sun, then enter at t=0
        people = []
        Tmrt_sun = sh.tmrt_field(self.T_out, self.cl['I_dir'])[0].max()
        for (x, y) in self.people_positions():
            b = JOS3(height=1.72, weight=70)
            b.tdb, b.tr, b.rh, b.v, b.par = self.T_out, Tmrt_sun, self.rh_out, 0.5, 1.6
            b.simulate(times=20, dtime=60)
            b.par = 1.2
            cells = [sh.cell_of(x + SEG_POS[s][0], y + SEG_POS[s][1], SEG_POS[s][2]) for s in SEGMENTS]
            people.append((b, np.array(cells)))
        steps = int(self.wait_min * 60 / self.dt)
        hist = []
        hc = 3.1                                            # still-air convective coeff [W/m2K]
        for n in range(steps + 1):
            C_field = self.comfort_field(T, W)
            Cp = [float(self.w_seg @ C_field[c]) for _, c in people]
            Tsk = np.mean([b.t_skin_mean for b, _ in people])
            Tcore = np.mean([b.t_core[0] for b, _ in people])
            hist.append(dict(t=n * self.dt / 60, C=np.mean(Cp), C_people=Cp,
                             T_mean=T.mean(), Tsk=Tsk, Tcore_head=Tcore))
            if n == steps:
                break
            sT = self.s_T.copy(); sW = np.zeros(sh.n)
            rh = rh_from_W(T, W)
            for b, c in people:
                b.tdb, b.tr, b.rh, b.v = T[c], self.Tmrt[c], rh[c], 0.1
                b.simulate(times=1, dtime=self.dt)
                bsa = np.array(b.bsa).ravel()
                q_conv = hc * bsa * (np.array(b.t_skin) - T[c])          # W into air
                d = b.dict_results()
                lat = np.array([d[f'q_skin2env_latent_{s}'][-1] for s in SEGMENTS])
                lat[0] += d['q_res_latent'][-1]
                q_conv[0] += d['q_res_sensible'][-1]
                np.add.at(sT, c, q_conv)
                np.add.at(sW, c, lat / H_FG * CP)                         # kg/s -> A_W units
            T = lu_dT.solve(M * T + self.b_Tc + sT)
            W = lu_dW.solve(M * W + self.b_Wc + sW)
        self.hist = hist
        self.C_mean = np.mean([h['C'] for h in hist])
        self.T_end, self.W_end = T, W
        # resources per hour
        self.E_kWh = self.sys['P_kW']                       # rated kW for 1 h
        self.W_L = self.water_kg_s * 3600                    # L per h (1 kg = 1 L)
        self.cost = self.c1 * self.E_kWh + self.c2 * self.W_L
        return self


def efficiency(base, sc):
    if sc.cost == 0:
        return np.nan
    return (base.C_mean - sc.C_mean) / sc.cost
