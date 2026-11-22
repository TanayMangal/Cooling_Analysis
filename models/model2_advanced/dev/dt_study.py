"""Time-step study on the 2a bus shelter (Phoenix, wind into front, Jetstream 250 running):
equilibrium of the empty shelter after 75 s, averaged over the last 30 s, for dt down to 0.01 s,
backward semi-Lagrangian vs the team's forward scheme (tent weights + ghost outside-air points)."""
import sys, os, time, json, numpy as np
from m2run import *
fn = 'out/dt_study.jsonl'
done = set()
if os.path.exists(fn):
    for l in open(fn):
        d = json.loads(l); done.add((d['scheme'], d['dt'], d['system']))
CASES = [('bwd', dt) for dt in [0.1, 0.05, 0.02, 0.01]] + [('fwd', dt) for dt in [0.1, 0.01]]
for scheme, dt in CASES:
        for system in ['None', 'B Evap']:
            if (scheme, dt, system) in done:
                continue
            t0 = time.time()
            r = Run('bus', 'Phoenix (hot-dry)', system, 'front', dt=dt, scheme=scheme, verbose=False)
            m = r.m; occ = m.occupied(); acc = []
            while m.time < 75 - 1e-9:
                m.step(dt)
                if m.time >= 45 - 1e-9 and int(round(m.time / dt)) % max(1, int(round(1.0 / dt))) == 0:
                    U = utci_full(m.T, m.Tmrt, rh_from_W(m.T, m.W), m.speed())
                    acc.append((m.T[occ].mean(), U[occ].mean(), m.speed()[occ].mean()))
            a = np.mean(acc, 0)
            r = dict(scheme=scheme, dt=dt, system=system, T_occ=float(a[0]), utci_occ=float(a[1]), v_occ=float(a[2]), wall_s=time.time() - t0)
            open(fn, 'a').write(json.dumps(r) + '\n'); print(r, flush=True)
