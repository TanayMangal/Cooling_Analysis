"""Batch runner for Model 2.
Examples:
  python run_m2.py rail                      # quick set used in the paper draft (wind into front, 5-min wait)
  python run_m2.py rail --winds front along behind --wait 15     # full sweep (big machine)
  python run_m2.py bus --winds front
Results append to out/<site>_results.jsonl ; time series to out/<site>_hist/<run>.json"""
import argparse, json, os, time
from m2run import *
ap = argparse.ArgumentParser()
ap.add_argument('site', choices=['bus', 'rail'])
ap.add_argument('--climates', nargs='+', default=list(CLIMATES))
ap.add_argument('--winds', nargs='+', default=['front'])
ap.add_argument('--systems', nargs='+', default=None)
ap.add_argument('--dt', type=float, default=0.1)
ap.add_argument('--wait', type=float, default=5.0)
ap.add_argument('--scheme', default='bwd')
ap.add_argument('--eq-tmax', type=float, default=600.0, help='warm-up cap in seconds (longer reruns for unsettled cases)')
a = ap.parse_args()
os.makedirs(f'out/{a.site}_hist', exist_ok=True)
done = set()
fn = f'out/{a.site}_results.jsonl'
if os.path.exists(fn):
    for line in open(fn):
        d = json.loads(line); done.add((d['climate'], d['wind'], d['system'], d['dt'], d['scheme'], d.get('wait_min'), d.get('eq_tmax', 600.0)))
for cl in a.climates:
    for wd in a.winds:
        for sysn in (a.systems or SYSTEMS[a.site]):
            if (cl, wd, sysn, a.dt, a.scheme, a.wait, a.eq_tmax) in done:
                continue
            r = Run(a.site, cl, sysn, wd, dt=a.dt, wait_min=a.wait, scheme=a.scheme, eq_tmax=a.eq_tmax).run()
            with open(fn, 'a') as f:
                f.write(json.dumps(r.result) + '\n')
            tag = f"{cl.split()[0]}_{wd}_{sysn.replace(' ', '_').replace('+','')}_dt{a.dt}_wait{a.wait:g}_{a.scheme}" + ('' if a.eq_tmax == 600 else f'_eq{a.eq_tmax:g}')
            json.dump(r.hist, open(f'out/{a.site}_hist/{tag}.json', 'w'))
            np.savez_compressed(f'out/{a.site}_hist/{tag}_fields.npz', T=r.m.T.astype(np.float32), W=r.m.W.astype(np.float32),
                                U=r.m.U.astype(np.float32), Tmrt=r.m.Tmrt.astype(np.float32))
