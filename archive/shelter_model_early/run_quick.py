import time, numpy as np
from shelter_model import *
t=time.time()
for cl in CLIMATES:
    base=Scenario(cl,'None').run()
    print(cl, 'T_out',base.T_out,'base Tmean %.2f'%base.T0.mean(),'C_base %.2f'%base.C_mean, 'sunlit frac %.2f'%base.sunlit.mean())
    for sysn in ['A Fan','B Evap','C Large evap']:
        sc=Scenario(cl,sysn).run()
        print(f"  {sysn:13s} V={sc.Vdot:5.2f}m3/s Q={sc.Q/1000:6.2f}kW feas={sc.feasible} dT={sc.dT_achieved:.2f} T0mean={sc.T0.mean():.2f} water={sc.W_L:.1f}L/h "
              f"RHmean={rh_from_W(sc.T0,sc.W0).mean():.0f}% C={sc.C_mean:.2f} dC={base.C_mean-sc.C_mean:.2f} cost=${sc.cost:.3f}/h eps={efficiency(base,sc):.1f}")
print(time.time()-t,'s')
