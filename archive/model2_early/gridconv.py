import sys, time, numpy as np, flow
fine = len(sys.argv)>1 and sys.argv[1]=='fine'
if fine:
    flow.DOMAIN.update(d=(0.15,0.075,0.125), N=(40,48,28))
from model2 import *
t=time.time()
s0=Scenario2('Phoenix (hot-dry)','None','front'); d0=s0.make_domain(0)
s=Scenario2('Phoenix (hot-dry)','B Evap','front'); d=s.make_domain(s.Vmax/0.3)
print('flows done %.0fs'%(time.time()-t),flush=True)
for dt in ([0.25,0.1,0.05,0.025,0.0125] if fine else [0.25,0.1,0.05,0.025,0.0125]):
    Tp=s0.steady(s0.operators(d0,dt))[0][d0.interior_mask()].mean()
    T=s.steady(s.operators(d,dt))[0][d.interior_mask()].mean()
    print(('fine ' if fine else 'coarse'),f'dt={dt:6.4f} passive {Tp:.2f} cooled {T:.2f} drop {Tp-T:.2f}  ({time.time()-t:.0f}s)',flush=True)
