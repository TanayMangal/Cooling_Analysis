"""
PROBLEM 3: Model 2 applied to two real venues (geometry, devices, crowds).

Venue 1  STADIUM CONCOURSE  modelled on a concourse section under the stands at
         M. A. Chidambaram Stadium (Chepauk), Chennai, India: coastal, hot-humid, afternoon sea
         breeze. Semi-open: gates and high vents in the outer facade, vomitories to the bowl,
         a concession kiosk with kitchen heat, concourse ends open (the concourse continues).
Venue 2  FARMERS MARKET SHED  modelled on the covered market shed in Flowery Branch, Georgia
         (city project: covered shed with fans and space for about 32 vendors; summer market
         Thursdays 3:30-7 pm). Open on all sides, two rows of 16 vendor tables, a hot-food stall
         (grill) with a queue in the aisle, ceiling fans.
All dimensions below are DESIGN ASSUMPTIONS (marked [assume]); replace with measured values when
known. Climates are marked [verify] where they are not from a cited source.

People are agents with a group and a path: queueing and browsing people stand; walkers move along
the aisle / concourse at a set speed (wrapping at the ends = a steady stream). Every 10 s each JOS-3
body reads its current cells and returns its heat and moisture there.
"""
import numpy as np
from m2core import Wall, Device, Geometry, Model2, utci_full, RHO, CP, H_FG, SIGMA, rh_from_W, W_from_rh
from m2run import Run, SPEC, CFM, GPH, JOS3, SEGMENTS, SEG_POS, seg_weights, time

# ------------------------------------------------------------------ climates
VENUE_CLIMATES = {
    # Chennai, May afternoon. T: mean daily max May 2022/2023 at Nungambakkam 36.3/36.7 C and
    # Meenambakkam 37.2/37.8 C; RH at 17:30 IST 63-77 % (TN Statistical Handbook 2022-23, Table 3.1,
    # source RMC Chennai). U10 sea breeze 4.0 m/s [verify]; solar [assume, afternoon].
    'stadium': dict(name='Chennai (coastal hot-humid)', T_out=37.0, rh_out=65.0, I_dir=750.0, I_roof=850.0, U10=4.0),
    # Flowery Branch / Gainesville GA, July market hours (late afternoon). [verify all: T, RH, wind]
    'market': dict(name='Flowery Branch GA (warm-humid)', T_out=32.0, rh_out=50.0, I_dir=650.0, I_roof=800.0, U10=3.0),
}
MAIN_WIND = {'stadium': 'behind', 'market': 'front'}
WIND_LABEL = {
    'stadium': {'behind': 'sea breeze through the gates (facade side)', 'front': 'from the bowl side',
                'along': 'along the concourse'},
    'market': {'front': 'across the shed (from the north row)', 'behind': 'across the shed (from the south row)',
               'along': 'along the shed axis'},
}

SPEC = dict(SPEC)
# 60-inch commercial ceiling fan on high: 7,500 CFM at 70 W [verify against the installed model]
SPEC['ceiling_fan60'] = dict(flow=7500 * CFM, kW=0.07)


def panels_with_openings(axis, c, along, zr, openings, kind='wall', U=2.0, absorb=0.6):
    """Vertical wall on plane axis=c spanning `along` (the horizontal axis) x zr, with rectangular
    openings [((a0, a1), (z0, z1)), ...]. Returns solid Wall panels around the openings."""
    W = []; a = along[0]
    for (o0, o1), (z0, z1) in sorted(openings):
        if o0 > a:
            W.append(Wall(axis, c, (a, o0), zr, kind, U, absorb))
        if z0 > zr[0]:
            W.append(Wall(axis, c, (o0, o1), (zr[0], z0), kind, U, absorb))
        if z1 < zr[1]:
            W.append(Wall(axis, c, (o0, o1), (z1, zr[1]), kind, U, absorb))
        a = o1
    if a < along[1]:
        W.append(Wall(axis, c, (a, along[1]), zr, kind, U, absorb))
    return W


# ------------------------------------------------------------------ venue 1: stadium concourse
STAD = dict(L=40.0, D=8.0, H=4.5,                         # concourse section length, depth, clear height [assume]
            gates=[(4.0, 8.0), (18.0, 22.0), (32.0, 36.0)], gate_h=3.5,
            vents=[(11.0, 15.0), (25.0, 29.0)], vent_z=(3.0, 4.0),
            voms=[(10.0, 13.0), (27.0, 30.0)], vom_h=3.0,
            kiosk=(18.0, 22.0, 6.5, 8.0, 3.0), kiosk_kW=5.0)  # x0,x1,y0,y1,height; sensible kitchen heat [assume]


def stadium_concourse(h=0.5, gates='open'):
    """x along the concourse (ends open: it continues around the stadium), y = 0 outer facade
    (gates, high vents), y = 8 inner wall to the bowl (vomitories), roof = underside of the stand."""
    S = STAD; L, D, H = S['L'], S['D'], S['H']
    g_open = S['gates'] if gates == 'open' else [S['gates'][1]]
    ops = [((a, b), (0.0, S['gate_h'])) for a, b in g_open] + [((a, b), S['vent_z']) for a, b in S['vents']]
    W = panels_with_openings(1, 0.0, (0, L), (0, H), ops)                                   # facade
    W += panels_with_openings(1, D, (0, L), (0, H), [((a, b), (0.0, S['vom_h'])) for a, b in S['voms']])
    W.append(Wall(2, H, (0, L), (0, D), 'roof', U=2.0, absorb=0.6))                          # stand above
    x0, x1, y0, y1, kh = S['kiosk']
    W += [Wall(1, y0, (x0, x1), (0.0, 1.0)), Wall(1, y0, (x0, x1), (2.0, kh)),               # counter + header
          Wall(0, x0, (y0, y1), (0.0, kh)), Wall(0, x1, (y0, y1), (0.0, kh)), Wall(2, kh, (x0, x1), (y0, y1))]
    return Geometry(f'Stadium concourse (Chepauk-type), gates {gates}', h, (-4.0, -6.0, 0.0), (44.0, 14.0, 9.0), W,
                    ((0, L), (0, D), (0, H)), sun_dir=(0.0, 1.0, 0.0), sun_elev_deg=40.0)


# ------------------------------------------------------------------ venue 2: farmers market shed
MKT = dict(L=48.0, D=12.0, H=5.0, booth=3.0, n_booth=16, aisle=(3.0, 9.0), table_z=1.0,
           grill_booth=7, grill_conv_kW=10.5, grill_rad_kW=5.3)
# grill: one 60,000 BTU/h (17.6 kW) propane grill [assume]; ~60 % leaves as convected heat, ~30 % as radiation [assume]


def market_shed(h=0.5):
    """Open-sided shed, roof at 5 m, 16 booths per side (3 m each), 6 m central aisle."""
    M = MKT; L, D, H, b = M['L'], M['D'], M['H'], M['booth']
    W = [Wall(2, H, (0, L), (0, D), 'roof', U=5.0, absorb=0.6)]
    for i in range(M['n_booth']):
        xs = (i * b + 0.5, i * b + b - 0.5)
        W.append(Wall(2, M['table_z'], xs, (2.0, 3.0)))          # south-row table
        W.append(Wall(2, M['table_z'], xs, (9.0, 10.0)))         # north-row table
    return Geometry('Farmers market shed (Flowery Branch-type)', h, (-6.0, -6.0, 0.0), (54.0, 18.0, 9.0), W,
                    ((0, L), (0, D), (0, H)), sun_dir=(1.0, 0.4, 0.0), sun_elev_deg=35.0)


# ------------------------------------------------------------------ systems
VENUE_SYSTEMS = {
    'stadium': ['None', 'A Fans x6', 'B Evap at gates x3', 'C Evap at queue x3', 'D Spot AC at queue x2',
                'E Spot AC x2 + gate curtains'],
    'market': ['None', 'A Ceiling fans x8', 'B Drum fans at queue x4', 'C Drum fans along aisle x4',
               'D Evap at queue', 'E Misting at queue'],
}


def venue_devices(site, system, gates='open', h=0.5):
    S = SPEC; D = []
    fan = S['fan_tpi24']; js = S['evap_js250']; ac = S['spotac_k63']; cur = S['curtain_berner118']
    if site == 'stadium':
        if system == 'A Fans x6':                                   # wall fans on the facade, blowing in
            for x in (2.75, 10.25, 15.25, 24.75, 29.75, 37.25):
                D.append(Device('TPI 24in fan', 'fan', (x, 0.5, 2.75), 1, +1, (0.5, 0.5), fan['flow'], fan['kW']))
        if system == 'B Evap at gates x3':                          # one Jetstream 250 inside each gate
            for x in (6.0, 20.0, 34.0):
                D.append(Device('Jetstream 250', 'evap', (x, 1.0, 1.25), 1, +1, (1.0, 0.5), js['flow'], js['kW'], eta=js['eta']))
        if system == 'C Evap at queue x3':                          # same three units, aimed at the kiosk queue
            D.append(Device('Jetstream 250', 'evap', (16.5, 4.0, 1.25), 0, +1, (1.0, 0.5), js['flow'], js['kW'], eta=js['eta']))
            D.append(Device('Jetstream 250', 'evap', (23.5, 4.0, 1.25), 0, -1, (1.0, 0.5), js['flow'], js['kW'], eta=js['eta']))
            D.append(Device('Jetstream 250', 'evap', (20.0, 1.0, 1.25), 1, +1, (1.0, 0.5), js['flow'], js['kW'], eta=js['eta']))
        if system in ('D Spot AC at queue x2', 'E Spot AC x2 + gate curtains'):
            for x, sg in ((16.5, +1), (23.5, -1)):
                D.append(Device('MovinCool K63', 'spotac', (x, 4.0, 1.25), 0, sg, (0.5, 0.5), ac['flow'], ac['kW'],
                                Q_kW=ac['Q_kW'], exhaust_at=(x, 4.0, 6.0)))
        if system == 'E Spot AC x2 + gate curtains':                # air curtains blowing down inside each open gate
            g_open = STAD['gates'] if gates == 'open' else [STAD['gates'][1]]
            for a, b in g_open:
                n = int(np.ceil((b - a) / cur['length']))
                D.append(Device('Berner 118in air curtain x%d' % n, 'curtain', ((a + b) / 2, h / 2, STAD['gate_h']), 2, -1,
                                (b - a, h), 0.0, n * cur['kW'], V_slot=cur['V_slot'], slot=cur['slot']))
        return D
    # market
    if system == 'A Ceiling fans x8':
        cf = S['ceiling_fan60']
        for i in range(8):
            D.append(Device('60in ceiling fan', 'fan', (3.25 + 6 * i, 6.25, 4.0), 2, -1, (1.5, 1.5), cf['flow'], cf['kW']))
    if system == 'B Drum fans at queue x4':                         # aisle side of the queue, blowing across it
        for x in (15.75, 17.75, 19.75, 21.75):
            D.append(Device('TPI 24in fan', 'fan', (x, 7.0, 1.25), 1, +1, (0.5, 0.5), fan['flow'], fan['kW']))
    if system == 'C Drum fans along aisle x4':                      # same four fans, blowing along the aisle
        for x in (4.0, 16.0, 28.0, 40.0):
            D.append(Device('TPI 24in fan', 'fan', (x, 6.25, 1.25), 0, +1, (0.5, 0.5), fan['flow'], fan['kW']))
    if system == 'D Evap at queue':                                 # one Jetstream 250 at the tail of the queue
        D.append(Device('Jetstream 250', 'evap', (14.5, 8.0, 1.25), 0, +1, (1.0, 0.5), js['flow'], js['kW'], eta=js['eta']))
    if system == 'E Misting at queue':                              # nozzle line under the stall eave, over the queue
        ms = S['mist_1000psi']; n_noz = int(7.0 / ms['spacing']) + 1
        D.append(Device('Misting line (%d nozzles)' % n_noz, 'mist', (19.0, 8.0, 3.0), 1, +1, (7.0, 0.5), 0.0,
                        ms['pump_kW'], water_kg_s=n_noz * ms['nozzle_gph'] * GPH))
    return D


# ------------------------------------------------------------------ crowds
def stadium_people(n_queue=12):
    P = []
    ys = [6.0, 5.2, 4.4, 3.6, 2.8, 2.0, 1.6, 1.2]
    for i in range(n_queue):                                        # two lines at the kiosk window
        P.append(dict(group='queue', x=19.25 if i % 2 == 0 else 20.75, y=ys[i // 2], vx=0.0, par=1.2))
    for i in range(6):                                              # two walking streams, opposite directions
        P.append(dict(group='walk', x=1.0 + i * 38.0 / 6, y=1.0, vx=+1.0, par=2.0))
        P.append(dict(group='walk', x=1.0 + (i + 0.5) * 38.0 / 6, y=1.75, vx=-1.0, par=2.0))
    for x, y in ((11.0, 7.0), (12.0, 6.5), (11.5, 6.0), (28.0, 7.0), (29.0, 6.5), (28.5, 6.0)):
        P.append(dict(group='stand', x=x, y=y, vx=0.0, par=1.2))   # waiting by the vomitories
    return P


def market_people(n_queue=10):
    P = []
    for i in range(n_queue):                                        # queue for the grill stall, along the aisle
        P.append(dict(group='queue', x=22.5 - 0.75 * i, y=8.25, vx=0.0, par=1.2))
    for i in range(6):                                              # shoppers strolling both ways
        P.append(dict(group='walk', x=1.0 + i * 46.0 / 6, y=5.25, vx=+0.8, par=2.0))
        P.append(dict(group='walk', x=1.0 + (i + 0.5) * 46.0 / 6, y=6.75, vx=-0.8, par=2.0))
    for x, y in ((4.5, 3.75), (10.5, 3.75), (16.5, 3.75), (28.5, 3.75), (34.5, 3.75), (40.5, 3.75), (31.5, 8.25), (40.5, 8.25)):
        P.append(dict(group='browse', x=x, y=y, vx=0.0, par=1.4))
    for x, y in ((4.5, 1.25), (16.5, 1.25), (34.5, 1.25), (22.5, 10.75), (31.5, 10.75), (40.5, 10.75)):
        P.append(dict(group='vendor', x=x, y=y, vx=0.0, par=1.4))  # (22.5, 10.75) is the cook at the grill
    return P


# ------------------------------------------------------------------ run
class VenueRun(Run):
    """Same physics and equilibrium as Run (Model 2); adds venue geometry, persistent heat sources,
    a radiant grill, moving agents in groups, and saves time-averaged 3-D fields for figures."""

    def __init__(self, site, system, wind_dir=None, dt=0.1, wait_min=5.0, scheme='bwd', gates='open',
                 grill=True, n_queue=None, eq_tmax=600.0, h=0.5, c1=0.14, c2=0.002, verbose=True, body_dt=10.0):
        cl = VENUE_CLIMATES[site]
        self.site, self.climate, self.system = site, cl['name'], system
        self.wind_dir = wind_dir or MAIN_WIND[site]
        self.dt, self.wait_min, self.c1, self.c2, self.verbose, self.eq_tmax = dt, wait_min, c1, c2, verbose, eq_tmax
        self.gates, self.grill, self.body_dt = gates, bool(grill), body_dt
        geom = stadium_concourse(h, gates) if site == 'stadium' else market_shed(h)
        self.devs = venue_devices(site, system, gates, h)
        self.m = Model2(geom, cl, self.wind_dir, devices=self.devs, scheme=scheme)
        m = self.m; X = m.X
        src = np.zeros(m.n)
        if site == 'stadium':
            x0, x1, y0, y1, _ = STAD['kiosk']
            cells = (X[:, 0] > x0 + 0.4) & (X[:, 0] < x1 - 0.4) & (X[:, 1] > y0 + 0.4) & (X[:, 1] < y1) & (X[:, 2] > 0.5) & (X[:, 2] < 2.0)
            src[cells] = STAD['kiosk_kW'] * 1000 / cells.sum()
            self.n_queue = n_queue or 12; self.spec_people = stadium_people(self.n_queue)
        else:
            if self.grill:
                gx = MKT['grill_booth'] * MKT['booth']
                cells = (X[:, 0] > gx + 0.5) & (X[:, 0] < gx + 2.5) & (X[:, 1] > 10.0) & (X[:, 1] < 11.0) & (np.abs(X[:, 2] - 1.25) < 0.26)
                src[cells] = MKT['grill_conv_kW'] * 1000 / cells.sum()
                # radiant grill: point source added to the mean radiant temperature, same form as the sun term
                P = np.array([gx + 1.5, 10.5, 1.25]); r2 = np.maximum(((X - P) ** 2).sum(1), 0.25)
                E = MKT['grill_rad_kW'] * 1000 / (4 * np.pi * r2)
                Tm = m.Tmrt.ravel() + 273.15
                m.Tmrt = (((Tm ** 4 + 0.25 * 0.7 * E / (0.97 * SIGMA)) ** 0.25) - 273.15).reshape(m.Tmrt.shape)
            self.n_queue = n_queue or 10; self.spec_people = market_people(self.n_queue)
        m.src_T = src.reshape(tuple(m.g.N)) if src.any() else None
        self.n_people = len(self.spec_people)
        self.w_seg = seg_weights()

    def log(self, *a):
        if self.verbose:
            print(f'[{self.site}|{self.system}|{self.wind_dir}|gates={self.gates}|grill={self.grill}|q={self.n_queue}]', *a, flush=True)

    def _pos(self, p, t):
        if p['vx'] == 0:
            return p['x'], p['y']
        lo, hi = 1.0, (STAD['L'] if self.site == 'stadium' else MKT['L']) - 1.0
        return lo + (p['x'] - lo + p['vx'] * t) % (hi - lo), p['y']

    def _cells(self, x, y, p=None):
        return np.array([self.cell((x + SEG_POS[s][0], y + SEG_POS[s][1], SEG_POS[s][2])) for s in SEGMENTS])

    def run(self):
        t_wall = time.time(); m = self.m
        self.equilibrate(t_max=self.eq_tmax, **getattr(self, 'eq_kw', {}))
        occ = m.occupied()
        eq = dict(T=m.T.astype(np.float32), W=m.W.astype(np.float32), U=m.U.astype(np.float32),
                  UTCI=m.utci_field().astype(np.float32))
        Tmrt_sun = m.Tmrt_open_sun   # weather-based open sun, identical in every case (not the grill-heated field max)
        people = []
        for p in self.spec_people:                         # 20 min outdoors in the sun before arriving
            b = JOS3(height=1.72, weight=70)
            b.tdb, b.tr, b.rh, b.v, b.par = m.T_out, Tmrt_sun, m.rh_out, max(m.U10 * 0.55, 0.1), 1.6
            b.simulate(times=20, dtime=60); b.par = p['par']
            people.append(b)
        bdt = self.body_dt; nsub = int(round(bdt / self.dt)); steps = int(round(self.wait_min * 60 / bdt))
        m.water_used = 0.0; m.water_evap = 0.0; m.heat_exhausted_J = 0.0; m.placed_frac = []
        groups = [p['group'] for p in self.spec_people]
        acc = dict(T=0.0, W=0.0, U=0.0, UTCI=0.0); nacc = 0
        hist = []; sweat = 0.0
        for k in range(steps + 1):
            t = k * bdt
            Tf = m.T.ravel(); Wf = m.W.ravel(); rh = rh_from_W(Tf, Wf); sp_ = m.speed().ravel(); Tm = m.Tmrt.ravel()
            U = utci_full(Tf, Tm, rh, sp_)
            pos = [self._pos(p, t) for p in self.spec_people]
            cells = [self._cells(x, y, p) for (x, y), p in zip(pos, self.spec_people)]
            Cp = [float(self.w_seg @ U[c]) for c in cells]
            hist.append(dict(t=t / 60, C=float(np.mean(Cp)), C_people=Cp, xy=[(round(x, 2), round(y, 2)) for x, y in pos],
                             utci_occ=float(U[occ.ravel()].mean()), T_occ=float(Tf[occ.ravel()].mean()),
                             RH_occ=float(rh[occ.ravel()].mean()), v_occ=float(sp_[occ.ravel()].mean()),
                             Tsk=float(np.mean([b.t_skin_mean for b in people])),
                             Tcore=float(np.mean([b.t_core[0] for b in people])),
                             Tcore_max=float(np.max([b.t_core[0] for b in people]))))
            acc['T'] = acc['T'] + m.T; acc['W'] = acc['W'] + m.W; acc['U'] = acc['U'] + m.U
            acc['UTCI'] = acc['UTCI'] + U.reshape(m.T.shape); nacc += 1
            if k == steps:
                break
            m.extra_T[:] = 0; m.extra_W[:] = 0
            eT = m.extra_T.ravel(); eW = m.extra_W.ravel()
            for b, c in zip(people, cells):
                v = np.maximum(sp_[c], 0.1)
                b.tdb, b.tr, b.rh, b.v = Tf[c], Tm[c], rh[c], v
                b.simulate(times=1, dtime=bdt)
                d = b.dict_results()
                hc = np.maximum(8.3 * v ** 0.6, 3.1)
                q = hc * np.array(b.bsa).ravel() * (np.array(b.t_skin) - Tf[c]); q[0] += d['q_res_sensible'][-1]
                lat = np.array([d[f'q_skin2env_latent_{s}'][-1] for s in SEGMENTS]); lat[0] += d['q_res_latent'][-1]
                np.add.at(eT, c, q); np.add.at(eW, c, lat / H_FG)
                sweat += lat.sum() / H_FG * bdt
            for _ in range(nsub):
                m.step(self.dt)
            if self.verbose and k % 3 == 0:
                self.log(f'wait t={(k+1)*bdt:.0f}s C={hist[-1]["C"]:.2f} maxspeed={m.speed().max():.2f} occT={m.T[occ].mean():.2f} wall={time.time()-t_wall:.0f}s')
        self.hist = hist; self.groups = groups
        self.fields = dict(eq=eq, avg={k_: (v_ / nacc).astype(np.float32) for k_, v_ in acc.items()},
                           Tmrt=m.Tmrt.astype(np.float32), xc=m.xc, yc=m.yc, zc=m.zc)
        wait_s = self.wait_min * 60
        Cp_t = np.array([h_['C_people'] for h_ in hist]).mean(0)            # time-averaged per person
        res = dict(site=self.site, climate=self.climate, system=self.system, wind=self.wind_dir, dt=self.dt,
                   scheme=m.scheme, wait_min=self.wait_min, eq_tmax=self.eq_tmax, gates=self.gates, grill=self.grill,
                   n_queue=self.n_queue, n_people=self.n_people, t_eq=self.t_eq, eq_converged=bool(self.converged),
                   eq_change=self.eq_change, C=float(Cp_t.mean()))
        for g in sorted(set(groups)):
            sel = np.array(groups) == g
            res[f'C_{g}'] = float(Cp_t[sel].mean()); res[f'n_{g}'] = int(sel.sum())
        res.update(utci_occ=float(np.mean([h_['utci_occ'] for h_ in hist])), T_occ=float(np.mean([h_['T_occ'] for h_ in hist])),
                   RH_occ=float(np.mean([h_['RH_occ'] for h_ in hist])), v_occ=float(np.mean([h_['v_occ'] for h_ in hist])),
                   Tsk_end=hist[-1]['Tsk'], Tcore_end=hist[-1]['Tcore'], Tcore_max=hist[-1]['Tcore_max'],
                   sweat_g_per_person=float(sweat * 1000 / self.n_people),
                   kW=float(sum(d_.power_kW for d_ in self.devs)),
                   water_L_h=float(m.water_used / wait_s * 3600),
                   water_evap_frac=float(m.water_evap / m.water_used) if m.water_used > 0 else float('nan'),
                   exhaust_kW_removed=float(m.heat_exhausted_J / wait_s / 1000),
                   placed_frac=float(np.mean(m.placed_frac)) if m.placed_frac else float('nan'),
                   wall_min=(time.time() - t_wall) / 60)
        res['cost'] = self.c1 * res['kW'] + self.c2 * res['water_L_h']
        res.update(getattr(self, 'extra_result', {}))
        self.result = res
        self.log('done', {k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in res.items()})
        return self
