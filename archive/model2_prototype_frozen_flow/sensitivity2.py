import numpy as np, pandas as pd, time
from multiprocessing import Pool
from model2 import *
fanV=3.59
CASES=[('baseline',{}),('forward IDW (team scheme)',dict(scheme='fwd_idw')),('dt_s=0.25 s',dict(dt_s=0.25)),
       ('alpha_t=0.01',dict(alpha_t=0.01)),('alpha_t=0.05',dict(alpha_t=0.05)),
       ('wind x0.5',dict(wind_scale=0.5)),('wind x1.5',dict(wind_scale=1.5)),('calm (no outside wind)',dict(wind_scale=0.0))]
def job(a):
    cl,(name,kw)=a; rows=[]
    b=Scenario2(cl,'None','front',**kw).run(); pm=b.T0[b.dom.interior_mask()].mean()
    for s in ['A Fan','B Evap']:
        x=Scenario2(cl,s,'front',fan_V=fanV,**kw).run(passive_T_mean=pm)
        rows.append(dict(climate=cl,case=name,system=s,C_base=b.C_mean,dC=b.C_mean-x.C_mean,eps=efficiency(b,x),dT_air=x.dT_achieved,feasible=x.feasible))
        print(rows[-1],flush=True)
    return rows
if __name__=='__main__':
    jobs=[(cl,c) for cl in ['Phoenix (hot-dry)','Miami (hot-humid)'] for c in CASES]
    with Pool(2) as p: res=p.map(job,jobs)
    pd.DataFrame([r for rr in res for r in rr]).to_csv('out/sensitivity_model2.csv',index=False); print('done')
