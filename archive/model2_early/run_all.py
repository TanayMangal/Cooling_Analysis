import json, time, pickle, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import brentq
from model2 import *

def fan_drop(V):
    """Reference mild day, calm, no sun: peak UTCI drop in the occupied zone from the fan."""
    out=[]
    for sysn,v in (('None',0.0),('A Fan',V)):
        s=Scenario2('Reference (mild)',sysn,fan_V=v).calibrate()
        out.append(s.comfort_field(s.dom,s.T0,s.W0))
    X=s.dom.G.X; occ=s.dom.interior_mask()&(X[:,2]>0.5)&(X[:,2]<1.75)
    return (out[0]-out[1])[occ].max(), (out[0]-out[1])[occ].mean()

def group(args):
    cl,dirn,fanV=args; rows=[]; hist={}
    b=Scenario2(cl,'None',dirn).run(); pm=b.T0[b.dom.interior_mask()].mean()
    hist['None']=b.hist
    for sysn in ['A Fan','B Evap','C Large evap']:
        s=Scenario2(cl,sysn,dirn,fan_V=fanV).run(passive_T_mean=pm); hist[sysn]=s.hist
        rows.append(dict(climate=cl,direction=dirn,system=sysn,Vjet=s.Vjet,Vdot=s.Vdot,feasible=s.feasible,
            dT_air=s.dT_achieved,water_L_h=s.W_L,cost=s.cost,C_base=b.C_mean,C=s.C_mean,dC=b.C_mean-s.C_mean,
            eps=efficiency(b,s),occ_speed=s.speed_occ,occ_speed_base=b.speed_occ,T_in_base=pm))
        print(rows[-1],flush=True)
    return rows,(cl,dirn,hist)

if __name__=='__main__':
    t=time.time()
    f=lambda V: fan_drop(V)[0]-3.0
    fanV=brentq(f,0.5,12,xtol=0.05)
    pk,mn=fan_drop(fanV)
    print(f'fan calibrated: outlet {fanV:.2f} m/s -> peak drop {pk:.2f}, occupied mean drop {mn:.2f}  ({time.time()-t:.0f}s)',flush=True)
    json.dump(dict(fanV=fanV,peak=pk,mean=mn),open('out/fan_cal.json','w'))
    jobs=[(cl,d,fanV) for cl in ['Phoenix (hot-dry)','Miami (hot-humid)'] for d in ['front','along','behind']]
    with Pool(2) as p: res=p.map(group,jobs)
    rows=[r for rr,_ in res for r in rr]; pd.DataFrame(rows).to_csv('out/results_model2.csv',index=False)
    pickle.dump({k[:2]:k[2] for _,k in res},open('out/hist_model2.pkl','wb'))
    print('done',time.time()-t)
