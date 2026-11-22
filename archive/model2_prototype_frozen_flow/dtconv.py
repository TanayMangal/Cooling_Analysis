import numpy as np
from model2 import *
for cl in ['Phoenix (hot-dry)','Miami (hot-humid)']:
    s0=Scenario2(cl,'None','front'); d0=s0.make_domain(0)
    s=Scenario2(cl,'B Evap','front'); d=s.make_domain(s.Vmax/0.3)
    for sch in ['bwd_tri','fwd_idw']:
        for dt in [0.5,0.25,0.1,0.05,0.025]:
            s0.scheme=s.scheme=sch
            Tp=s0.steady(s0.operators(d0,dt))[0][d0.interior_mask()].mean()
            T=s.steady(s.operators(d,dt))[0][d.interior_mask()].mean()
            print(cl[:7],sch,f'dt={dt:5.3f} passive {Tp:.2f}  cooled {T:.2f}  drop {Tp-T:.2f}',flush=True)
