"""Round 7: one rail-platform sensitivity run (Part 2b). Writes to out/rail_sens_results.jsonl, NOT to
rail_results.jsonl, so the production table is never touched.
  python run_sens.py --climate "Phoenix (hot-dry)" --system "B Evap x3" --param nu --value 0.01
--param: base (nothing changed) | nu (nu_t = alpha_t) | eta (cooler pad efficiency) | u10 (wind speed at 10 m)
Settings otherwise match the Part 2b production runs: wind from the track, dt 0.1 s, 5-min wait, 600 s warm-up."""
import argparse, json, os
from m2run import *
ap = argparse.ArgumentParser()
ap.add_argument('--climate', required=True)
ap.add_argument('--system', required=True)
ap.add_argument('--param', choices=['base', 'nu', 'eta', 'u10'], default='base')
ap.add_argument('--value', type=float, default=None)
ap.add_argument('--wind', default='front')
ap.add_argument('--dt', type=float, default=0.1)
ap.add_argument('--wait', type=float, default=5.0)
ap.add_argument('--eq-tmax', type=float, default=600.0)
ap.add_argument('--h', type=float, default=None, help='grid size override (testing only)')
a = ap.parse_args()
os.makedirs('out/rail_sens_hist', exist_ok=True)
fn = 'out/rail_sens_results.jsonl'
key = (a.climate, a.system, a.param, a.value)
if os.path.exists(fn):
    for line in open(fn):
        d = json.loads(line)
        if (d['climate'], d['system'], d['sens_param'], d['sens_value']) == key:
            print('already done', key); raise SystemExit
kw = {}
if a.param == 'nu':
    kw = dict(nu_t=a.value, alpha_t=a.value)
elif a.param == 'eta':
    kw = dict(eta=a.value)
elif a.param == 'u10':
    kw = dict(U10=a.value)
r = Run('rail', a.climate, a.system, a.wind, dt=a.dt, wait_min=a.wait, eq_tmax=a.eq_tmax, h=a.h, **kw).run()
r.result.update(sens_param=a.param, sens_value=a.value)
with open(fn, 'a') as f:
    f.write(json.dumps(r.result) + '\n')
tag = f"{a.climate.split()[0]}_{a.system.split()[0]}_{a.param}{'' if a.value is None else f'{a.value:g}'}"
json.dump(r.hist, open(f'out/rail_sens_hist/{tag}.json', 'w'))
