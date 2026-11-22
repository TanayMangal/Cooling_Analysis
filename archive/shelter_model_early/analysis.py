import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from shelter_model import *
OUT='/home/claude/shelter_model/out'; import os; os.makedirs(OUT,exist_ok=True)
rows=[]; runs={}
for cl in CLIMATES:
    base=Scenario(cl,'None').run(); runs[(cl,'None')]=base
    for s in ['A Fan','B Evap','C Large evap']:
        sc=Scenario(cl,s).run(); runs[(cl,s)]=sc
        rows.append(dict(climate=cl,system=s,Vdot=sc.Vdot,Q_kW=sc.Q/1000,feasible=sc.feasible,
            dT_mean_air=sc.dT_achieved,water_L_h=sc.W_L,cost=sc.cost,C_base=base.C_mean,C=sc.C_mean,
            dC=base.C_mean-sc.C_mean,eps=efficiency(base,sc),eps_people=efficiency(base,sc)*sc.n_people))
import pandas as pd
df=pd.DataFrame(rows); df.to_csv(f'{OUT}/results.csv',index=False); print(df.round(2).to_string())

# people effect: equilibrium vs end-of-wait mean air T
for k,sc in runs.items(): print(k, 'T0 %.2f -> T_end %.2f'%(sc.T0.mean(), sc.T_end.mean()), 'Tsk %.2f->%.2f'%(sc.hist[0]['Tsk'],sc.hist[-1]['Tsk']))

# sensitivity (Phoenix + Miami, system B and A)
sens=[]
def one(cl,s,**kw):
    shkw={k:kw.pop(k) for k in list(kw) if k in ('alpha_t','sun_elev_deg','dTmrt_shade')}
    b=Scenario(cl,'None',sh=Shelter(**shkw),**kw).run(); x=Scenario(cl,s,sh=Shelter(**shkw),**kw).run()
    return efficiency(b,x), b.C_mean-x.C_mean
for cl in CLIMATES:
  for s in ['A Fan','B Evap','C Large evap']:
    for name,kw in [('baseline',{}),('alpha_t=0.01',dict(alpha_t=0.01)),('alpha_t=0.05',dict(alpha_t=0.05)),
                    ('sun 70deg (midday)',dict(sun_elev_deg=70)),('uniform body weights',dict(uniform_weights=True)),
                    ('c2 x10 (water scarcity)',dict(c2=0.02)),('fan: peak not mean',dict(fan_interp='peak')),
                    ('fan: climate-corrected',dict(fan_climate=True)),('eta_pad 0.7',dict(eta_pad=0.7))]:
        if s!='A Fan' and name.startswith('fan'): continue
        e,d=one(cl,s,**kw); sens.append(dict(climate=cl,system=s,case=name,eps=e,dC=d))
sdf=pd.DataFrame(sens); sdf.to_csv(f'{OUT}/sensitivity.csv',index=False); print(sdf.round(2).to_string())

# figures: mid-x slices of T and comfort for Phoenix B and Miami B
fig,ax=plt.subplots(2,3,figsize=(13,8)); fig.subplots_adjust(hspace=0.55)
for r,cl in enumerate(CLIMATES):
    for c,s in enumerate(['None','B Evap','C Large evap']):
        sc=runs[(cl,s)]; sh=sc.sh; i=sh.N[0]//2
        Cf=sc.comfort_field(sc.T0,sc.W0).reshape(sh.N)[i]
        im=ax[r,c].imshow(Cf.T,origin='lower',extent=[0,sh.L[1],0,sh.L[2]],cmap='inferno',vmin=36,vmax=52,aspect='auto')
        ax[r,c].set_title(f'{cl.split()[0]} | {s}\nmean air {sc.T0.mean():.1f}°C',fontsize=9)
        ax[r,c].set_xlabel('depth y (m)  back → open front'); ax[r,c].set_ylabel('z (m)')
fig.colorbar(im,ax=ax,label='no-wind UTCI (°C)'); fig.savefig(f'{OUT}/slices_utci.png',dpi=150,bbox_inches='tight')

fig,ax=plt.subplots(1,2,figsize=(11,4),sharey=False)
for a,cl in zip(ax,CLIMATES):
    for s in ['None','A Fan','B Evap','C Large evap']:
        h=runs[(cl,s)].hist; a.plot([x['t'] for x in h],[x['C'] for x in h],label=s)
    a.set_title(cl); a.set_xlabel('minutes after arrival (t=0 = cooled equilibrium)'); a.set_ylabel('body-weighted no-wind UTCI (°C)'); a.legend(fontsize=8)
fig.savefig(f'{OUT}/transient.png',dpi=150,bbox_inches='tight')

w=seg_weights(); order=np.argsort(-w)
fig,a=plt.subplots(figsize=(7,4)); a.barh([SEGMENTS[i] for i in order][::-1],w[order][::-1],color='#3b7dd8')
a.set_xlabel('normalized comfort weight w_k (Zhang et al. 2009 cooling slope)'); fig.savefig(f'{OUT}/body_weights.png',dpi=150,bbox_inches='tight')
print(dict(zip(SEGMENTS,np.round(w,4))))
