"""
PROBLEM 3 (CAD version): the two venues built from the team's CAD models instead of hand-placed panels.

  'stands'   cricket_stadium_1.stl, scaled x1500 (scale not known; chosen to be Chepauk-sized). The CAD model
             stands on a 7-8 m solid base under the field; that base is removed so the field is at
             ground level and the bowl rim is about 12 m high. Climate: Chennai (M. A. Chidambaram
             Stadium). Question: is air conditioning IN THE STANDS worth it?
             Sun from the west at 40 deg (late afternoon), so the EAST stand (x > 0) faces the sun:
             that is the studied sector (spectators seated at x 59-85 m, y -6 to 36 m).
  'market2'  Assembly 1.stl (Flowery Branch farmers market), scaled x10 [confirm scale]: four gabled
             sheds (16.5 x 6.1 m, eave 3.25 m, ridge 5 m) around a square, 24 vendor tables.
             Two grills (north and east sheds) with queues and a kettle-corn kettle (west shed).

The CAD is voxelised once (cad_voxelize.py -> cad/<site>_h<h>.npz); the solver treats every solid
cell as blocked (no air through it), as a shade-caster for the sun, and as a floor that the sun
heats. Roof undersides of sunlit roofs are hot (same panel balance as Model 2 roofs).
"""
import numpy as np
from m2core import Geometry, Device, Model2, SIGMA, H_FG
from m2run import SPEC, GPH, SEGMENTS, SEG_POS, seg_weights, time
from m3venues import VenueRun, VENUE_CLIMATES

CAD_CLIMATES = {'stands': VENUE_CLIMATES['stadium'], 'market2': VENUE_CLIMATES['market']}
CAD_MAIN_WIND = {'stands': 'against', 'market2': 'front'}
CAD_WIND_LABEL = {   # model axes: +x = east (toward the Bay of Bengal at Chepauk), +y = north  [confirm]
    'stands': {'against': 'from the east: afternoon sea breeze off the Bay of Bengal (main)',
               'behind': 'from the south (the sea breeze often turns SE)', 'along': 'from the west (hot land wind before the sea breeze)'},
    'market2': {'front': 'from the north', 'behind': 'from the south', 'along': 'from the west'},
}
# The air that arrives depends on where it comes from. Sea breeze (east) = cooler and more humid;
# land wind (west) = hotter and drier. Base values (37 C / 65 %) are the cited May afternoon figures.
# Sea breeze: Nungambakkam (coastal) May max 36.3-36.7 C, 17:30 IST RH 63-77 % (TN Statistical Handbook 2022-23).
# Land wind: hotter inland max (Meenambakkam 37.2-37.8 C) and dry westerlies before onset -> 39 C / 45 % [VERIFY].
WIND_CLIMATE = {'stands': {'against': dict(T_out=36.0, rh_out=72.0, name='Chennai, sea breeze (E)'),
                           'behind': dict(T_out=37.0, rh_out=65.0, name='Chennai, southerly'),
                           'along': dict(T_out=39.0, rh_out=45.0, name='Chennai, land wind (W)')}}
CAD_WINDS = {'stands': ['against', 'behind', 'along'], 'market2': ['front', 'along', 'behind']}
CAD_SYSTEMS = {
    'stands': ['None', 'A Evap x8', 'B Misting x8 rows', 'C Seat AC x8', 'D Seat AC x16', 'E Seat AC x32', 'F Seat AC x64', 'G Seat AC whole bowl'],
    # the market already has ceiling fans, so they are the BASELINE and are on in every case except 'No fans'
    'market2': ['Base ceiling fans x12', 'No fans', 'B +Drum fans at queues x4', 'C +Evap at queues x2',
                'D +Misting at queues x2', 'E +Evap in every shed x4', 'F +Spot AC at queues x2'],
}
SPEC = dict(SPEC)
SPEC['ceiling_fan60'] = dict(flow=7500 * 0.000471947, kW=0.07)    # 60" commercial ceiling fan [verify]

# crowd in the stands [assume / VERIFY]: 0.4 m2 per seat, 50 % of seats filled -> 1.25 people/m2;
# a seated adult gives off about 70 W sensible + 45 W latent (ASHRAE Fundamentals ch.18, seated, light work)
REV = 2                                   # model revision written into every result row
CAD_BASE = {'stands': 'None', 'market2': 'Base ceiling fans x12'}
CROWD = dict(per_m2=1.25, sens_W=70.0, lat_W=45.0)
# stadium scale: unknown in the CAD; x1500 gives ~18 m high stands and a ~230 m bowl (Chepauk-sized) [estimate]
SC = 1.5                                 # all stadium positions below are written at x1000 and multiplied by SC
STAND_C = (0.0, 10.7 * SC)               # centre of the bowl [m]
SECTOR = dict(x=(38.0 * SC, 58.0 * SC), y=(-5.0 * SC, 25.0 * SC))


# ------------------------------------------------------------------ geometry from the voxels
def load_solid(site, h):
    d = np.load(f'cad/{site}_h{h:g}.npz')
    S = d['solid'].astype(bool); lo = d['lo'].astype(float); hi = d['hi'].astype(float); h = float(d['h'])
    if site == 'stands':
        # remove the solid base under the field: shift everything down by the field's thickness
        ic = int((STAND_C[0] - lo[0]) / h); jc = int((STAND_C[1] - lo[1]) / h)
        shift = int(S[ic, jc].sum())
        nz = int(round(36.0 / h))                                    # keep 36 m above ground (rim ~18 m)
        S2 = np.zeros(S.shape[:2] + (nz,), bool)
        part = S[:, :, shift:shift + nz]; S2[:, :, :part.shape[2]] = part
        S = S2; hi = np.array([hi[0], hi[1], lo[2] + nz * h])
    return S, lo, hi, h


def column_top(S, h):
    """height of the top solid surface in every (i, j) column [m] (0 where there is no solid)"""
    nz = S.shape[2]; has = S.any(2)
    ktop = nz - 1 - np.argmax(S[:, :, ::-1], 2)
    return np.where(has, (ktop + 1) * h, 0.0)


def cad_geometry(site, h):
    S, lo, hi, h = load_solid(site, h)
    if site == 'stands':
        g = Geometry(f'Cricket stadium stands (CAD, h={h:g} m)', h, lo, hi, [],
                     ((lo[0], hi[0]), (lo[1], hi[1]), (0.0, hi[2])), sun_dir=(-1.0, 0.0, 0.0), sun_elev_deg=40.0)
    else:
        g = Geometry(f'Flowery Branch market (CAD, h={h:g} m)', h, lo, hi, [],
                     ((-16.5, 16.5), (-16.5, 16.5), (0.0, 5.5)), sun_dir=(1.0, 0.4, 0.0), sun_elev_deg=35.0)
    assert tuple(g.N) == S.shape, (g.N, S.shape)
    # roofs: market pavilion = wooden roof, U ~ 2 W/m2K [assume]; stadium overhangs = concrete, U ~ 5 [assume]
    g.roof_U = 2.0 if site == 'market2' else 5.0
    g.longwave = True
    g.solid = S
    xc = lo[0] + (np.arange(g.N[0]) + 0.5) * h; yc = lo[1] + (np.arange(g.N[1]) + 0.5) * h
    zc = lo[2] + (np.arange(g.N[2]) + 0.5) * h
    X, Y, Z = np.meshgrid(xc, yc, zc, indexing='ij')
    top = column_top(S, h)
    if site == 'stands':
        r = np.hypot(xc[:, None] - STAND_C[0], yc[None, :] - STAND_C[1])
        seat_col = (top > 0.5 * h) & (r > 30 * SC) & (r < 62 * SC)     # stepped seating around the bowl
        first_air = (Z > top[:, :, None]) & (Z < top[:, :, None] + h) & ~S
        g.seats = seat_col[:, :, None] & first_air                     # air cells people sit in
        sec = ((X > SECTOR['x'][0]) & (X < SECTOR['x'][1]) & (Y > SECTOR['y'][0]) & (Y < SECTOR['y'][1]))
        g.occ_mask = g.seats & sec
    else:
        occ = ((np.abs(X) < 16.5) & (np.abs(Y) < 16.5) & (Z > 0.5) & (Z < 1.75) & ~S)
        g.occ_mask = occ
    g.top = top; g.axes = (xc, yc, zc)
    return g


def snap(axis_vals, v):
    return float(axis_vals[np.argmin(np.abs(axis_vals - v))])


# ------------------------------------------------------------------ devices
def cad_devices(site, system, g):
    D = []; h = g.h; xc, yc, zc = g.axes; top = g.top
    js250 = SPEC['evap_js250']; js270 = SPEC['evap_js270']; ac = SPEC['spotac_k63']; fan = SPEC['fan_tpi24']
    ms = SPEC['mist_1000psi']
    if site == 'stands':
        rows_all = xc[(xc > 39.0 * SC) & (xc < 57.5 * SC)]             # one seat row per column of cells
        def zseat(x, y, yface=None):
            # seat-level cell; for a unit blowing along y through the face at yface, both the intake cell and
            # the outlet cell must be air, so take the higher of the two steps either side of the face
            i = np.argmin(np.abs(xc - x)); j = np.argmin(np.abs(yc - y)); t = top[i, j]
            if yface is not None:
                jf = int(round((yface - g.lo[1]) / h)); t = max(top[i, max(jf - 1, 0)], top[i, min(jf, len(yc) - 1)])
            return t + 0.5 * h
        def pick(n):                                                   # n rows spread over the stand
            return rows_all[np.unique(np.round(np.linspace(0, len(rows_all) - 1, n)).astype(int))]
        ends = [(SECTOR['y'][0], +1, SECTOR['y'][0] + 1.0), (SECTOR['y'][1], -1, SECTOR['y'][1] - 1.0)]
        if system == 'A Evap x8':                                      # Jetstream 270 at the aisle end of 4 rows
            for x in pick(4):
                for yf, sg, yin in ends:
                    D.append(Device('Jetstream 270', 'evap', (x, yf, zseat(x, yin, yf)), 1, sg, (1.0, 1.0), js270['flow'],
                                    js270['kW'], eta=js270['eta']))
        if system == 'B Misting x8 rows':                              # one 30 m nozzle line along each of 8 rows
            n_noz = int((SECTOR['y'][1] - SECTOR['y'][0]) / ms['spacing']) + 1
            for x in pick(8):
                i = np.argmin(np.abs(xc - x)); xf = xc[i] - 0.5 * h
                yc_mid = 0.5 * (SECTOR['y'][0] + SECTOR['y'][1])
                D.append(Device('Misting line (%d nozzles)' % n_noz, 'mist', (xf, yc_mid, top[i, np.argmin(np.abs(yc - yc_mid))] + 0.5 * h),
                                0, +1, (SECTOR['y'][1] - SECTOR['y'][0], 0.5 * h), 0.0, ms['pump_kW'],
                                water_kg_s=n_noz * ms['nozzle_gph'] * GPH, follow_surface=True))
        n_ac = {'C Seat AC x8': 8, 'D Seat AC x16': 16, 'E Seat AC x32': 32, 'F Seat AC x64': 64}.get(system, 0)
        if n_ac:
            # seat-level air conditioning: K63 units under the seats at row ends (and mid-row for x32),
            # discharging along the row at seat height; condenser heat ducted out above the rim
            spots = list(ends) if n_ac <= 16 else ends + [(5.0 * SC, +1, 6.0 * SC), (15.0 * SC, -1, 14.0 * SC)]
            if n_ac == 64:          # 8 units per row: both ends + 6 more spread along the row, alternating direction
                y0, y1 = SECTOR['y']; L = y1 - y0
                spots = list(ends) + [(y0 + L * k / 7, s_, y0 + L * k / 7 + s_ * 1.0) for k, s_ in zip(range(1, 7), (+1, -1) * 3)]
            rows = pick(max(1, n_ac // len(spots)))
            for x in rows:
                for yf, sg, yin in spots:
                    D.append(Device('MovinCool K63 (seat level)', 'spotac', (x, yf, zseat(x, yin, yf)), 1, sg, (h, h), ac['flow'],
                                    ac['kW'], Q_kW=ac['Q_kW'], exhaust_at=(66.0 * SC, yin, 24.0)))
        if system == 'G Seat AC whole bowl':
            # the same density of units as E (32 units in the east sector) over EVERY stand of the bowl.
            # Each unit sits at seat level and blows along its row: along x or y, whichever neighbour cell has
            # the same step height (so intake and outlet are both air); condenser heat ducted out at 30 m.
            seat_cols = np.argwhere(g.seats.any(2)); n_sector = int(g.occ_mask.sum())
            n_units = int(round(32 * len(seat_cols) / max(n_sector, 1)))
            rng = np.random.default_rng(11); order = rng.permutation(len(seat_cols)); used = set()
            for idx in order:
                if len([d for d in D]) >= n_units:
                    break
                i, j = seat_cols[idx]; t = top[i, j]
                if (i, j) in used:
                    continue
                for ax_, di, dj in ((0, 1, 0), (1, 0, 1), (0, -1, 0), (1, 0, -1)):
                    i2, j2 = i + di, j + dj
                    if 0 <= i2 < len(xc) and 0 <= j2 < len(yc) and abs(top[i2, j2] - t) < 1e-6 and (i2, j2) not in used:
                        sg = 1 if (di + dj) > 0 else -1
                        cen = [xc[i], yc[j], t + 0.5 * h]
                        cen[ax_] = (xc[i] if ax_ == 0 else yc[j]) + sg * 0.5 * h        # the face between the two cells
                        D.append(Device('MovinCool K63 (seat level)', 'spotac', tuple(cen), ax_, sg, (h, h), ac['flow'],
                                        ac['kW'], Q_kW=ac['Q_kW'], exhaust_at=(xc[i], yc[j], 30.0)))
                        used.update({(i, j), (i2, j2)})
                        break
        return D
    # ---- market2
    if system != 'No fans':                                           # existing ceiling fans: 3 per shed along the aisle, blades at 2.5-3 m (air above and below)
        cf = SPEC['ceiling_fan60']
        for s in (-5.5, 0.0, 5.5):
            for x, y in ((s, 12.75), (s, -12.75), (12.75, s), (-12.75, s)):
                D.append(Device('60in ceiling fan', 'fan', (snap(xc, x), snap(yc, y), 2.5), 2, -1, (1.5, 1.5), cf['flow'], cf['kW']))
    if system == 'B +Drum fans at queues x4':                         # plaza side of each queue, blowing across it
        for x in (0.75, 3.75):
            D.append(Device('TPI 24in fan', 'fan', (snap(xc, x), 10.0, 1.25), 1, +1, (0.5, 0.5), fan['flow'], fan['kW']))
        for y in (-0.75, -3.75):
            D.append(Device('TPI 24in fan', 'fan', (10.0, snap(yc, y), 1.25), 0, +1, (0.5, 0.5), fan['flow'], fan['kW']))
    if system == 'C +Evap at queues x2':
        D.append(Device('Jetstream 250', 'evap', (snap(xc, 2.25), 10.0, 1.25), 1, +1, (1.0, 0.5), js250['flow'], js250['kW'], eta=js250['eta']))
        D.append(Device('Jetstream 250', 'evap', (10.0, snap(yc, -2.25), 1.25), 0, +1, (1.0, 0.5), js250['flow'], js250['kW'], eta=js250['eta']))
    if system == 'D +Misting at queues x2':                           # nozzle line under the eave, over each queue
        L = 5.5; n_noz = int(L / ms['spacing']) + 1
        D.append(Device('Misting line (%d nozzles)' % n_noz, 'mist', (2.125, 13.0, 2.75), 1, +1, (L, 0.5), 0.0, ms['pump_kW'],
                        water_kg_s=n_noz * ms['nozzle_gph'] * GPH))
        D.append(Device('Misting line (%d nozzles)' % n_noz, 'mist', (13.0, -2.125, 2.75), 0, +1, (L, 0.5), 0.0, ms['pump_kW'],
                        water_kg_s=n_noz * ms['nozzle_gph'] * GPH))
    if system == 'E +Evap in every shed x4':                          # one Jetstream 250 per shed, down the aisle
        for c, ax, sg in (((-8.75, 12.75), 0, +1), ((-12.75, -8.75), 1, +1), ((8.75, -12.75), 0, -1), ((12.75, 8.75), 1, -1)):
            ctr = [snap(xc, c[0]), snap(yc, c[1]), 1.25]; ctr[ax] = c[ax]
            D.append(Device('Jetstream 250', 'evap', tuple(ctr), ax, sg, (1.0, 0.5), js250['flow'], js250['kW'], eta=js250['eta']))
    if system == 'F +Spot AC at queues x2':
        D.append(Device('MovinCool K63', 'spotac', (snap(xc, 2.25), 10.0, 1.25), 1, +1, (0.5, 0.5), ac['flow'], ac['kW'],
                        Q_kW=ac['Q_kW'], exhaust_at=(2.25, 8.0, 6.5)))
        D.append(Device('MovinCool K63', 'spotac', (10.0, snap(yc, -2.25), 1.25), 0, +1, (0.5, 0.5), ac['flow'], ac['kW'],
                        Q_kW=ac['Q_kW'], exhaust_at=(8.0, -2.25, 6.5)))
    return D


# ------------------------------------------------------------------ people
def stands_people(g):
    """80 spectators in the sunny east sector + 20 in the west and 20 in the north stand (controls)."""
    P = []
    for x in np.linspace(39.5 * SC, 56.5 * SC, 8):
        for y in np.linspace(-4.0 * SC, 24.0 * SC, 10):
            P.append(dict(group='sector', x=x, y=y, vx=0.0, par=1.1, seated=True))
    for x in np.linspace(-56.5 * SC, -39.5 * SC, 4):
        for y in np.linspace(-2.0 * SC, 22.0 * SC, 5):
            P.append(dict(group='west', x=x, y=y, vx=0.0, par=1.1, seated=True))
    for x in np.linspace(-12.0 * SC, 12.0 * SC, 5):
        for y in np.linspace(STAND_C[1] + 39.5 * SC, STAND_C[1] + 56.5 * SC, 4):
            P.append(dict(group='north', x=x, y=y, vx=0.0, par=1.1, seated=True))
    return P


MK = dict(grills=[(6.0, 14.25), (14.25, -6.0)], grill_conv_kW=10.5, grill_rad_kW=5.3,
          kettle=(-14.25, 6.0), kettle_kW=5.0, loop=12.25)
# grill: one 60,000 BTU/h (17.6 kW) propane grill [assume]; ~60 % convected, ~30 % radiated [assume]
# kettle corn: 5 kW of convected heat [assume]


def market2_people(n_queue=8):
    P = []
    for i in range(n_queue):                                         # north grill queue (aisle, toward -x)
        P.append(dict(group='queue', x=4.75 - 0.75 * i, y=13.25, vx=0.0, par=1.2))
    for i in range(n_queue):                                         # east grill queue (aisle, toward +y)
        P.append(dict(group='queue', x=13.25, y=-4.75 + 0.75 * i, vx=0.0, par=1.2))
    per = 8 * MK['loop']                                             # walkers loop the square through the sheds
    for i in range(6):
        P.append(dict(group='walk', loop=True, s0=i * per / 6, vx=+0.8, par=2.0, x=0, y=0))
        P.append(dict(group='walk', loop=True, s0=(i + 0.5) * per / 6, vx=-0.8, par=2.0, x=0, y=0))
    for x, y in ((-13.25, -6.25), (-13.25, -2.0), (-13.25, 6.0), (-6.0, -13.25), (2.0, -13.25), (6.5, -13.25)):
        P.append(dict(group='browse', x=x, y=y, vx=0.0, par=1.4))
    for x, y in ((6.0, 15.4), (15.4, -6.0), (-2.0, 15.4), (-6.5, 10.1), (10.1, 2.0), (-15.4, -2.0), (-10.1, 6.0), (2.0, -15.4)):
        P.append(dict(group='vendor', x=x, y=y, vx=0.0, par=1.4))   # first two are the cooks at the grills
    return P


def loop_xy(s, a):
    """position on the square of half-side a, arc length s from (-a, -a), counter-clockwise"""
    L = 2 * a; s = s % (4 * L)
    if s < L:
        return -a + s, -a
    if s < 2 * L:
        return a, -a + (s - L)
    if s < 3 * L:
        return a - (s - 2 * L), a
    return -a, a - (s - 3 * L)


# ------------------------------------------------------------------ run
class CadVenueRun(VenueRun):
    def __init__(self, site, system, wind_dir=None, dt=None, wait_min=None, scheme='bwd', grill=True, n_queue=8,
                 eq_tmax=600.0, h=None, c1=0.14, c2=0.002, verbose=True, body_dt=None, crowd=True, u10=None):
        cl = dict(CAD_CLIMATES[site], **WIND_CLIMATE.get(site, {}).get(wind_dir or CAD_MAIN_WIND[site], {}))
        if u10 is not None:                       # e.g. a calm day
            cl['U10'] = float(u10)
        h = h or (2.5 if site == 'stands' else 0.5)
        dt = dt or (0.2 if site == 'stands' else 0.1)
        wait_min = wait_min or (15.0 if site == 'stands' else 5.0)
        body_dt = body_dt or (10.0 if site == 'stands' else 2.0)
        self.site, self.climate, self.system = site, cl['name'], system
        self.wind_dir = wind_dir or CAD_MAIN_WIND[site]
        self.dt, self.wait_min, self.c1, self.c2, self.verbose, self.eq_tmax = dt, wait_min, c1, c2, verbose, eq_tmax
        self.gates, self.grill, self.body_dt, self.crowd = 'cad', bool(grill), body_dt, bool(crowd)
        g = cad_geometry(site, h); self.h = h
        self.devs = cad_devices(site, system, g)
        self.m = Model2(g, cl, self.wind_dir, devices=self.devs, scheme=scheme)
        m = self.m; X = m.X; S = g.solid.ravel()
        src = np.zeros(m.n); srcW = np.zeros(m.n)
        if site == 'stands':
            if self.crowd:                                           # the rest of the crowd: heat and moisture per m2
                seats = g.seats.ravel(); A = h * h
                src[seats] = CROWD['per_m2'] * CROWD['sens_W'] * A
                srcW[seats] = CROWD['per_m2'] * CROWD['lat_W'] * A / H_FG
            self.spec_people = stands_people(g); self.n_queue = 0
        else:
            heat = [(c, MK['grill_conv_kW']) for c in MK['grills']] if self.grill else []
            heat.append((MK['kettle'], MK['kettle_kW']))
            for (cx, cy), kW in heat:
                cells = (np.abs(X[:, 0] - cx) < 0.8) & (np.abs(X[:, 1] - cy) < 0.6) & (X[:, 2] > 0.8) & (X[:, 2] < 1.6) & ~S
                src[cells] += kW * 1000 / cells.sum()
            if self.grill:                                           # radiant grills added to Tmrt (as in m3venues)
                Tm = m.Tmrt.ravel() + 273.15; E = np.zeros(m.n)
                for cx, cy in MK['grills']:
                    r2 = np.maximum(((X - np.array([cx, cy, 1.2])) ** 2).sum(1), 0.25)
                    E += MK['grill_rad_kW'] * 1000 / (4 * np.pi * r2)
                m.Tmrt = (((Tm ** 4 + 0.25 * 0.7 * E / (0.97 * SIGMA)) ** 0.25) - 273.15).reshape(m.Tmrt.shape)
            self.n_queue = n_queue; self.spec_people = market2_people(n_queue)
        m.src_T = src.reshape(tuple(g.N)) if src.any() else None
        m.src_W = srcW.reshape(tuple(g.N)) if srcW.any() else None
        self.n_people = len(self.spec_people)
        self.w_seg = seg_weights()
        # equilibrium judged on 60-s averages and on the mean change only: cooler jets puff and swirl, so a
        # single cell never stops changing (rail runs showed max changes of 2-17 C per window) even when the
        # space as a whole has settled. The largest cell change is still recorded in eq_change.
        self.eq_kw = dict(window=60.0, tolT_max=np.inf, t_min=120.0)
        # rev 2 (Nov 11): ceiling fans moved to 2.5 m and every device checked to draw from / blow into air
        # cells (rev 1 had some fan and seat-AC intakes inside the roof / a step); misting follows the seats
        self.extra_result = dict(u10=float(cl['U10']), u10_set=u10, rev=REV, eq_window_s=60.0, h=h, body_dt=body_dt, crowd=self.crowd, n_devices=len(self.devs),
                                 cooling_kW=float(sum(d.spec.get('Q_kW', 0.0) for d in self.devs)))
        if site == 'market2':       # queue LENGTH per grill (the group count n_queue would overwrite it)
            self.extra_result.update(n_queue=n_queue, n_queue_people=2 * n_queue)
        if site == 'stands':        # how many spectators the studied sector holds at the assumed occupancy
            self.extra_result['sector_people'] = float(g.occ_mask.sum() * h * h * CROWD['per_m2'])
            # spectators whose seats the system covers (whole bowl for G, the east sector otherwise)
            self.extra_result['covered_people'] = float((g.seats.sum() if system.startswith('G ') else g.occ_mask.sum())
                                                        * h * h * CROWD['per_m2'])

    def log(self, *a):
        if self.verbose:
            print(f'[{self.site}|{self.system}|{self.wind_dir}|h={self.h:g}|grill={self.grill}|bdt={self.body_dt:g}]', *a, flush=True)

    def _pos(self, p, t):
        if p.get('loop'):
            return loop_xy(p['s0'] + p['vx'] * t, MK['loop'])
        return p['x'], p['y']

    def _cells(self, x, y, p=None):
        g = self.m.g; h = g.h; S = g.solid
        i = int(np.clip((x - g.lo[0]) / h, 0, g.N[0] - 1)); j = int(np.clip((y - g.lo[1]) / h, 0, g.N[1] - 1))
        z0 = g.top[i, j] if (p is not None and p.get('seated')) else 0.0
        zs = 0.7 if (p is not None and p.get('seated')) else 1.0     # seated: body folded to ~70 % height
        out = []
        for s in SEGMENTS:
            dx, dy, dz = SEG_POS[s]
            ijk = np.clip(((np.array([x + dx, y + dy, z0 + 0.05 + dz * zs]) - g.lo) / h).astype(int), 0, g.N - 1)
            while S[tuple(ijk)] and ijk[2] < g.N[2] - 1:              # never read a solid cell: step up
                ijk[2] += 1
            out.append(np.ravel_multi_index(ijk, tuple(g.N)))
        return np.array(out)
