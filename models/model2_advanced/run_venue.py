"""Problem 3 runner (one venue case per call; resumable).
  python run_venue.py stands  --system "D Seat AC x16" --wind behind            (CAD cricket stadium)
  python run_venue.py market2 --system "C Evap at queues x2" --wind front        (CAD farmers market)
  python run_venue.py stadium --system "C Evap at queue x3" --wind behind       (old hand-built concourse)
Results append to out/<site>_results.jsonl. For every run it also saves
  out/<site>_hist/<tag>.json         per-person comfort every body step, positions, groups, zone means
  out/<site>_hist/<tag>_fields.npz   3-D fields at equilibrium (before people) and time-averaged over the
                                     wait: T, W, U (wind vector), UTCI, plus Tmrt and the grid axes"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('site', choices=['stadium', 'market', 'stands', 'market2'])
ap.add_argument('--system', required=True)
ap.add_argument('--wind', default=None)
ap.add_argument('--dt', type=float, default=None, help='default 0.1 (stands 0.2)')
ap.add_argument('--wait', type=float, default=None, help='minutes; default 5 (stands 15)')
ap.add_argument('--eq-tmax', type=float, default=600.0)
ap.add_argument('--gates', default='open', choices=['open', 'one'])
ap.add_argument('--grill', type=int, default=1)
ap.add_argument('--nqueue', type=int, default=0)
ap.add_argument('--h', type=float, default=None, help='cell size; default 0.5 (stands 2.5)')
ap.add_argument('--body-dt', type=float, default=None, help='seconds between body updates; default 10 (market2 2)')
ap.add_argument('--crowd', type=int, default=1, help='stands: background crowd heat on/off')
ap.add_argument('--u10', type=float, default=None, help='override the wind speed at 10 m (e.g. 0.5 for a calm day)')
a = ap.parse_args()
CAD = a.site in ('stands', 'market2')
if CAD:
    from m3cad import CadVenueRun, CAD_SYSTEMS as SYS, CAD_MAIN_WIND as MW
else:
    from m3venues import VenueRun, VENUE_SYSTEMS as SYS, MAIN_WIND as MW
assert a.system in SYS[a.site], f'unknown system {a.system!r}; choose from {SYS[a.site]}'
wind = a.wind or MW[a.site]
h = a.h or (2.5 if a.site == 'stands' else 0.5)
dt = a.dt or (0.2 if a.site == 'stands' else 0.1)
wait = a.wait or (15.0 if a.site == 'stands' else 5.0)
bdt = a.body_dt or (2.0 if a.site == 'market2' else 10.0)
nq = a.nqueue or {'stadium': 12, 'market': 10, 'market2': 8, 'stands': 0}[a.site]
os.makedirs(f'out/{a.site}_hist', exist_ok=True)
fn = f'out/{a.site}_results.jsonl'
key = lambda d: (d['system'], d['wind'], d['dt'], d['wait_min'], d['eq_tmax'], d['gates'], bool(d['grill']), d['n_queue'],
                 d.get('h', 0.5), d.get('body_dt', 10.0), bool(d.get('crowd', True)), d.get('rev', 1), d.get('u10_set'))
from m3cad import REV
me = (a.system, wind, dt, wait, a.eq_tmax, 'cad' if CAD else a.gates, bool(a.grill), nq, h, bdt, bool(a.crowd), REV if CAD else 1, a.u10)
if os.path.exists(fn) and any(key(json.loads(l)) == me for l in open(fn) if l.strip()):
    print('already done:', me); raise SystemExit
if CAD:
    r = CadVenueRun(a.site, a.system, wind, dt=dt, wait_min=wait, grill=a.grill, n_queue=nq, eq_tmax=a.eq_tmax, h=h,
                    body_dt=bdt, crowd=a.crowd, u10=a.u10).run()
else:
    r = VenueRun(a.site, a.system, wind, dt=dt, wait_min=wait, gates=a.gates, grill=a.grill, n_queue=nq,
                 eq_tmax=a.eq_tmax, h=h, body_dt=bdt).run()
    r.result.update(h=h, body_dt=bdt, crowd=True)
tag = (f"{a.system.split()[0]}_{wind}_dt{dt}_g{'cad' if CAD else a.gates}_grill{a.grill}_q{nq}_eq{a.eq_tmax:g}_h{h:g}"
       f"_b{bdt:g}_c{a.crowd}" + (f"_w{wait:g}" if wait != (15.0 if a.site == 'stands' else 5.0) else '') + (f"_u{a.u10:g}" if a.u10 is not None else ''))
json.dump(dict(groups=r.groups, hist=r.hist, result=r.result), open(f'out/{a.site}_hist/{tag}.json', 'w'))
F = r.fields
np.savez_compressed(f'out/{a.site}_hist/{tag}_fields.npz', xc=F['xc'], yc=F['yc'], zc=F['zc'], Tmrt=F['Tmrt'],
                    **({'solid': r.m.g.solid, 'top': r.m.g.top} if CAD else {}),
                    **{f'eq_{k}': v for k, v in F['eq'].items()}, **{f'avg_{k}': v for k, v in F['avg'].items()})
with open(fn, 'a') as f:
    f.write(json.dumps(r.result) + '\n')
