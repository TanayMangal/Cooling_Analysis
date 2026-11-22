import time, numpy as np, sys
from m2run import *
sch=sys.argv[1]; 
for dt in [float(x) for x in sys.argv[2:]]:
    out=[]
    for system in ['None','B Evap']:
        r=Run('bus','Phoenix (hot-dry)',system,'front',dt=dt,scheme=sch,verbose=False); m=r.m; occ=m.occupied(); t=time.time()
        while m.time<30-1e-9: m.step(dt)
        out.append(m.T[occ].mean())
    print(sch,'dt',dt,'None %.2f  B %.2f  drop %.2f  (%.0fs)'%(out[0],out[1],out[0]-out[1],time.time()-t),flush=True)
