"""Bus-shelter check (frozen wind, Phoenix, wind into front, cooler B at full airflow):
equilibrium cooling vs time step (down to 0.01 s) for backward and both forward schemes."""
import sys, time, numpy as np, flow
fine = len(sys.argv) > 1 and sys.argv[1] == 'fine'
if fine:
    flow.DOMAIN.update(d=(0.15, 0.075, 0.125), N=(40, 48, 28))
from model2 import *
t = time.time()
s0 = Scenario2('Phoenix (hot-dry)', 'None', 'front'); d0 = s0.make_domain(0)
s = Scenario2('Phoenix (hot-dry)', 'B Evap', 'front'); d = s.make_domain(s.Vmax / 0.3)
print('flows done %.0fs' % (time.time() - t), flush=True)
for sch in ['bwd_tri', 'fwd_cic', 'fwd_idw_ghost']:
    for dt in [0.1, 0.05, 0.025, 0.01]:
        s0.scheme = s.scheme = sch
        Tp = s0.steady(s0.operators(d0, dt))[0][d0.interior_mask()].mean()
        T = s.steady(s.operators(d, dt))[0][d.interior_mask()].mean()
        print(('fine  ' if fine else 'coarse'), f'{sch:14s} dt={dt:5.3f}  passive {Tp:.2f}  cooled {T:.2f}  drop {Tp-T:.2f}  ({time.time()-t:.0f}s)', flush=True)
