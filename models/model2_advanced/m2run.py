"""
MODEL 2 (ADVANCED): devices from spec sheets, scenarios, people (JOS-3) and metrics.

Equilibrium (relative t = 0): the empty space with the device running is marched until the
occupied-zone air stops changing. People (pre-heated by 20 min outdoors in the sun) then
arrive and wait; every 30 s each JOS-3 segment reads its cell's T, Tmrt, RH and wind speed,
and returns its convective heat and its sweat/breath moisture to that cell. The wind keeps
being solved the whole time, so body plumes rise (buoyancy).
"""
import time, json, numpy as np
from pythermalcomfort.models import JOS3
from m2core import *
from shelter_model import seg_weights, SEGMENTS, SEG_POS

CFM = 0.000471947                          # m3/s per CFM
CLIMATES = {
    # Model 1 values [verify vs ASHRAE]; wind = July mean (NOAA via currentresults.com)
    'Phoenix (hot-dry)': dict(T_out=43.0, rh_out=13.0, I_dir=850.0, I_roof=950.0, U10=3.17),
    'Miami (hot-humid)': dict(T_out=33.0, rh_out=55.0, I_dir=700.0, I_roof=850.0, U10=3.53),
    # Round 8 cities (the paper's main pair); same values as model1_basic/shelter_model.py, plus wind.
    # Delhi wind: May mean 10.2 km/h = 2.83 m/s (Wikipedia 'Climate of Delhi'; height not stated, taken as 10 m).
    # Dubai wind: Aug mean 7.5 mph = 3.35 m/s at 10 m (Weatherspark, MERRA-2) [verify both].
    'Delhi (hot-dry)':   dict(T_out=40.0, rh_out=28.0, I_dir=750.0, I_roof=900.0, U10=2.83),
    'Dubai (hot-humid)': dict(T_out=42.0, rh_out=44.0, I_dir=750.0, I_roof=900.0, U10=3.35),
}

# --------------------------------------------------------------------------- spec sheets
SPEC = {
    # TPI MB 24-DF 24" drum fan: 2,900-3,300 CFM, 1/4 hp, 2.2 A @ 120 V (webstaurantstore)
    'fan_tpi24': dict(flow=3300 * CFM, kW=0.26, dia=0.61),
    # Port-A-Cool Jetstream 250: 8,500 CFM, 24 mph, 6.6 A @ 115 V (bestbuyautoequipment)
    'evap_js250': dict(flow=8500 * CFM, kW=0.76, V=24 * 0.44704, eta=0.75),
    # Port-A-Cool Jetstream 270: 22,500 CFM, 26 mph, 15.2 A @ 115 V
    'evap_js270': dict(flow=22500 * CFM, kW=1.75, V=26 * 0.44704, eta=0.75),
    # MovinCool Climate Pro K63: 60,000 BTU/h, 1,770 CFM, 460V/3ph/20A (axisair spec sheet);
    # electrical input not published -> assumed EER 10 BTU/Wh = 6.0 kW  [verify]
    'spotac_k63': dict(flow=1770 * CFM, Q_kW=60000 / 3412.14, kW=6.0),
    # Berner Architectural Contour 8 AC08-F-2120A: 118" nozzle, 3,368 CFM, avg outlet
    # 1,827 fpm, 0.67 kW (Berner DS-260E, Feb 2026). Slot = 3368/1827 ft2 / 9.83 ft = 5.7 cm
    'curtain_berner118': dict(flow=3368 * CFM, kW=0.67, V_slot=1827 * 0.00508, length=118 * 0.0254,
                              slot=(3368 / 1827) / (118 / 12) * 0.3048),
    # High-pressure misting: 1.0 GPM @ 1000 PSI pump, 0.75 hp (hotsylv listing, '75' read as
    # 0.75 hp); .008" nozzles, 50 per pump (=1.2 GPH each), 24" spacing (cool-off.com)
    'mist_1000psi': dict(pump_kW=0.75 * 0.7457, nozzle_gph=1.2, spacing=0.6096),
}
GPH = 3.785411784 / 3600                   # kg/s per gallon/hour of water


def devices_for(site, system):
    S = SPEC; D = []
    if site == 'bus':
        if system == 'A Fan':
            D.append(Device('TPI 24in fan', 'fan', (1.5, 0.25, 1.75), 1, +1, (0.5, 0.5), S['fan_tpi24']['flow'], 0.25))
        if system == 'B Evap':
            D.append(Device('Jetstream 250', 'evap', (1.5, 0.75, 0.95), 1, +1, (0.625, 0.625), S['evap_js250']['flow'], 1.2,
                            eta=S['evap_js250']['eta']))
        if system == 'C Large evap':
            D.append(Device('Jetstream 270', 'evap', (1.5, 0.875, 1.125), 1, +1, (1.0, 1.0), S['evap_js270']['flow'], 2.5,
                            eta=S['evap_js270']['eta']))
        return D
    # rail platform canopy (2b): numbers of units chosen for a 20 m x 4 m platform
    if system == 'A Fans':
        for x in (2, 6, 10, 14, 18):
            D.append(Device('TPI 24in fan', 'fan', (x, 0.25, 2.5), 1, +1, (0.5, 0.5), S['fan_tpi24']['flow'], S['fan_tpi24']['kW']))
    if system == 'B Evap x3':
        for x in (3.5, 10, 16.5):
            D.append(Device('Jetstream 250', 'evap', (x, 0.75, 1.0), 1, +1, (0.75, 0.5), S['evap_js250']['flow'], S['evap_js250']['kW'],
                            eta=S['evap_js250']['eta']))
    if system == 'C Large evap x2':
        for x in (5, 15):
            D.append(Device('Jetstream 270', 'evap', (x, 1.0, 1.25), 1, +1, (1.0, 1.0), S['evap_js270']['flow'], S['evap_js270']['kW'],
                            eta=S['evap_js270']['eta']))
    if system == 'D Misting':
        n_noz = int(20 / S['mist_1000psi']['spacing']) + 1
        D.append(Device('Misting line (%d nozzles)' % n_noz, 'mist', (10, 3.625, 3.125), 1, +1, (20, 0.25), 0.0,
                        S['mist_1000psi']['pump_kW'], water_kg_s=n_noz * S['mist_1000psi']['nozzle_gph'] * GPH))
    if system == 'E Spot AC + air curtain':
        for x in (5, 15):
            D.append(Device('MovinCool K63', 'spotac', (x, 1.0, 1.375), 1, +1, (0.5, 0.25), S['spotac_k63']['flow'], S['spotac_k63']['kW'],
                            Q_kW=S['spotac_k63']['Q_kW'], exhaust_at=(x, 2.0, 4.0)))
        n = int(np.ceil(20 / S['curtain_berner118']['length']))
        c = S['curtain_berner118']
        D.append(Device('Berner 118in air curtain x%d' % n, 'curtain', (10, 3.875, 3.0), 2, -1, (20, 0.25), 0.0, n * c['kW'],
                        V_slot=c['V_slot'], slot=c['slot']))
    return D


SYSTEMS = {'bus': ['None', 'A Fan', 'B Evap', 'C Large evap'],
           'rail': ['None', 'A Fans', 'B Evap x3', 'C Large evap x2', 'D Misting', 'E Spot AC + air curtain']}


# --------------------------------------------------------------------------- scenario
class Run:
    def __init__(self, site, climate, system, wind_dir='front', dt=0.1, wait_min=10.0, scheme='bwd',
                 n_people=None, c1=0.14, c2=0.002, h=None, verbose=True, eq_tmax=600.0, eta=None, **kw):
        self.site, self.climate, self.system, self.wind_dir = site, climate, system, wind_dir
        self.dt, self.wait_min, self.c1, self.c2, self.verbose = dt, wait_min, c1, c2, verbose
        self.eq_tmax = eq_tmax                     # warm-up cap [s]; longer for runs that did not settle
        geom = (bus_shelter(h) if h else bus_shelter()) if site == 'bus' else (rail_canopy(h) if h else rail_canopy())
        self.devs = devices_for(site, system)
        if eta is not None:                        # round 7: pad-efficiency sensitivity (evaporative coolers only)
            for d in self.devs:
                if 'eta' in d.spec:
                    d.spec['eta'] = float(eta)
        self.m = Model2(geom, CLIMATES[climate], wind_dir, devices=self.devs, scheme=scheme, **kw)
        self.n_people = n_people or (6 if site == 'bus' else 20)
        self.w_seg = seg_weights()

    def log(self, *a):
        if self.verbose:
            print(f'[{self.site}|{self.climate[:5]}|{self.system}|{self.wind_dir}]', *a, flush=True)

    def equilibrate(self, t_min=60.0, t_max=600.0, window=20.0, tolT=0.05, tolT_max=0.5, tolW=2e-5, tolV=0.03):
        """March the empty space with devices on. Every `window` seconds compare the window-averaged
        occupied-zone FIELDS (not just their mean) with the previous window:
          mean |dT| < tolT, max |dT| < tolT_max, mean |dW| < tolW, mean |d speed| < tolV.
        Stops when two consecutive windows pass (after t_min), or at t_max with converged=False."""
        m = self.m; occ = m.occupied(); t0 = time.time()
        nwin = int(round(window / self.dt)); acc = None; k = 0; prev = None; passes = 0
        self.converged = False; self.eq_change = None
        while m.time < t_max - 1e-9:
            m.step(self.dt)
            cur = (m.T[occ].copy(), m.W[occ].copy(), m.speed()[occ].copy())
            acc = cur if acc is None else tuple(a_ + c_ for a_, c_ in zip(acc, cur)); k += 1
            if k == nwin:
                avg = tuple(a_ / k for a_ in acc); acc = None; k = 0
                if prev is not None:
                    dT = np.abs(avg[0] - prev[0]); dW = np.abs(avg[1] - prev[1]); dV = np.abs(avg[2] - prev[2])
                    self.eq_change = dict(dT_mean=float(dT.mean()), dT_max=float(dT.max()), dW_mean=float(dW.mean()), dV_mean=float(dV.mean()))
                    ok = (dT.mean() < tolT and dT.max() < tolT_max and dW.mean() < tolW and dV.mean() < tolV)
                    passes = passes + 1 if ok else 0
                    if passes >= 2 and m.time >= t_min:
                        self.converged = True
                        break
                prev = avg
        self.t_eq = m.time
        self.log(f'equilibrium t={m.time:.0f}s converged={self.converged} change={self.eq_change} occ T={m.T[occ].mean():.2f} ({time.time()-t0:.0f}s wall)')

    def positions(self):
        if self.site == 'bus':
            xs = np.linspace(0.4, 2.6, self.n_people); ys = np.where(np.arange(self.n_people) % 2 == 0, 0.45, 1.05)
            return list(zip(xs, ys))
        rng = np.random.default_rng(7)
        xs = np.linspace(1.0, 19.0, self.n_people) + rng.uniform(-0.3, 0.3, self.n_people)
        ys = rng.uniform(0.8, 3.4, self.n_people)
        return list(zip(xs, ys))

    def cell(self, p):
        g = self.m.g
        ijk = np.clip(((np.array(p) - g.lo) / g.h).astype(int), 0, g.N - 1)
        return np.ravel_multi_index(ijk, tuple(g.N))

    def run(self):
        t_wall = time.time()
        m = self.m
        self.equilibrate(t_max=self.eq_tmax)
        occ = m.occupied()
        # outdoor sun for the walk to the stop
        Tmrt_sun = m.Tmrt_open_sun   # weather-based open sun, identical in every case (not the grill-heated field max)
        people = []
        for (x, y) in self.positions():
            b = JOS3(height=1.72, weight=70)
            b.tdb, b.tr, b.rh, b.v, b.par = m.T_out, Tmrt_sun, m.rh_out, max(m.U10 * 0.55, 0.1), 1.6
            b.simulate(times=20, dtime=60); b.par = 1.2
            cells = np.array([self.cell((x + SEG_POS[s][0], y + SEG_POS[s][1], SEG_POS[s][2])) for s in SEGMENTS])
            people.append((b, cells))
        body_dt = 30.0; nsub = int(round(body_dt / self.dt)); steps = int(self.wait_min * 60 / body_dt)
        m.water_used = 0.0; m.water_evap = 0.0; m.heat_exhausted_J = 0.0   # wait-period counters start at arrival
        hist = []; sweat = 0.0
        for k in range(steps + 1):
            Tf = m.T.ravel(); Wf = m.W.ravel(); rh = rh_from_W(Tf, Wf); sp_ = m.speed().ravel()
            Tm = m.Tmrt.ravel()
            U = utci_full(Tf, Tm, rh, sp_)                                   # UTCI of every cell
            Cp = [float(self.w_seg @ U[c]) for _, c in people]
            hist.append(dict(t=k * body_dt / 60, C=float(np.mean(Cp)), C_people=Cp,
                             utci_occ=float(U[occ.ravel()].mean()), T_occ=float(Tf[occ.ravel()].mean()),
                             RH_occ=float(rh[occ.ravel()].mean()), v_occ=float(sp_[occ.ravel()].mean()),
                             Tsk=float(np.mean([b.t_skin_mean for b, _ in people])),
                             Tcore=float(np.mean([b.t_core[0] for b, _ in people])),
                             Tcore_max=float(np.max([b.t_core[0] for b, _ in people]))))
            if k == steps:
                break
            m.extra_T[:] = 0; m.extra_W[:] = 0
            eT = m.extra_T.ravel(); eW = m.extra_W.ravel()
            for b, c in people:
                v = np.maximum(sp_[c], 0.1)
                b.tdb, b.tr, b.rh, b.v = Tf[c], Tm[c], rh[c], v
                b.simulate(times=1, dtime=body_dt)
                d = b.dict_results()
                hc = np.maximum(8.3 * v ** 0.6, 3.1)
                q = hc * np.array(b.bsa).ravel() * (np.array(b.t_skin) - Tf[c]); q[0] += d['q_res_sensible'][-1]
                lat = np.array([d[f'q_skin2env_latent_{s}'][-1] for s in SEGMENTS]); lat[0] += d['q_res_latent'][-1]
                np.add.at(eT, c, q); np.add.at(eW, c, lat / H_FG)
                sweat += lat.sum() / H_FG * body_dt
            for _ in range(nsub):
                m.step(self.dt)
            if self.verbose:
                self.log(f'wait t={(k+1)*body_dt:.0f}s C={hist[-1]["C"]:.2f} maxspeed={m.speed().max():.2f} occT={m.T[occ].mean():.2f} wall={time.time()-t_wall:.0f}s')
        self.hist = hist
        self.people = people
        wait_s = self.wait_min * 60
        self.result = dict(site=self.site, climate=self.climate, system=self.system, wind=self.wind_dir,
                           dt=self.dt, scheme=m.scheme, wait_min=self.wait_min, t_eq=self.t_eq,
                           eq_converged=bool(self.converged), eq_change=self.eq_change, eq_tmax=self.eq_tmax,
                           exhaust_kW_removed=float(getattr(m, 'heat_exhausted_J', 0.0) / wait_s / 1000), exhaust_wait_only=True,
                           C=float(np.mean([h['C'] for h in hist])),
                           utci_occ=float(np.mean([h['utci_occ'] for h in hist])),
                           T_occ=float(np.mean([h['T_occ'] for h in hist])),
                           RH_occ=float(np.mean([h['RH_occ'] for h in hist])),
                           v_occ=float(np.mean([h['v_occ'] for h in hist])),
                           Tsk_end=hist[-1]['Tsk'], Tcore_end=hist[-1]['Tcore'], Tcore_max=hist[-1]['Tcore_max'],
                           sweat_g_per_person=float(sweat * 1000 / len(people)),
                           kW=float(sum(d.power_kW for d in self.devs)),
                           water_L_h=float(m.water_used / wait_s * 3600),
                           water_evap_frac=float(m.water_evap / m.water_used) if m.water_used > 0 else np.nan,
                           wall_min=(time.time() - t_wall) / 60)
        self.result['cost'] = self.c1 * self.result['kW'] + self.c2 * self.result['water_L_h']
        self.result.update(nu_t=float(m.nu_t), alpha_t=float(m.alpha_t), U10=float(m.U10),
                           eta=next((float(d.spec['eta']) for d in self.devs if 'eta' in d.spec), None))
        self.log('done', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.result.items()})
        return self


def efficiency(base, r):
    return np.nan if r['cost'] == 0 else (base['C'] - r['C']) / r['cost']
