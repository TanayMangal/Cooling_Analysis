import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from model2 import *
fig,ax=plt.subplots(2,3,figsize=(15,8.5)); fig.subplots_adjust(hspace=0.35)
for c,dirn in enumerate(['front','along','behind']):
    for r,sysn in enumerate(['None','B Evap']):
        s=Scenario2('Phoenix (hot-dry)',sysn,dirn)
        V=s.Vmax if sysn!='None' else 0
        dom=s.make_domain(V/0.3 if V else 0); ops=s.operators(dom,0.1); T,W=s.steady(ops)
        G=dom.G; U=dom.centers().reshape(*G.N,3); Tg=T.reshape(G.N)
        if dirn=='along':   # horizontal slice at z=1.1 m (x-y plane)
            k=7; a=ax[r,c]; im=a.imshow(Tg[:,:,k].T,origin='lower',extent=[G.lo[0],G.hi[0],G.lo[1],G.hi[1]],cmap='coolwarm',vmin=34,vmax=46,aspect='auto')
            X,Y=np.meshgrid(dom.xc,dom.yc,indexing='ij'); a.quiver(X,Y,U[:,:,k,0],U[:,:,k,1],scale=40,width=0.003)
            a.plot([0,3],[0,0],'k',lw=3); a.plot([0,0],[0,1.5],'k',lw=3); a.plot([3,3],[0,1.5],'k',lw=3)
            a.set_xlabel('x (m)'); a.set_ylabel('y (m)'); a.set_title(f'{dirn} wind, {sysn}: plan view at z=1.9 m (jet height)',fontsize=9)
        else:               # vertical slice at mid-x (y-z plane)
            i=10; a=ax[r,c]; im=a.imshow(Tg[i].T,origin='lower',extent=[G.lo[1],G.hi[1],G.lo[2],G.hi[2]],cmap='coolwarm',vmin=34,vmax=46,aspect='auto')
            Y,Z=np.meshgrid(dom.yc,dom.zc,indexing='ij'); a.quiver(Y,Z,U[i,:,:,1],U[i,:,:,2],scale=40,width=0.003)
            a.plot([0,0],[0,2.5],'k',lw=3); a.plot([0,1.5],[2.5,2.5],'k',lw=3)
            a.set_xlabel('y (m)  back wall at 0, open front at 1.5'); a.set_ylabel('z (m)'); a.set_title(f'{dirn} wind, {sysn}: side view at mid-x',fontsize=9)
fig.colorbar(im,ax=ax,label='air temperature (°C)'); fig.suptitle('Phoenix: wind field and air temperature at equilibrium (t = 0)')
fig.savefig('out/flow_slices.png',dpi=140,bbox_inches='tight')
H=pickle.load(open('out/hist_model2.pkl','rb'))
fig,ax=plt.subplots(1,2,figsize=(12,4))
for a,cl in zip(ax,['Phoenix (hot-dry)','Miami (hot-humid)']):
    for sysn,col in zip(['None','A Fan','B Evap','C Large evap'],['C0','C1','C2','C3']):
        h=H[(cl,'front')][sysn]; a.plot([x['t'] for x in h],[x['C'] for x in h],col,label=sysn)
        h=H[(cl,'along')][sysn]; a.plot([x['t'] for x in h],[x['C'] for x in h],col,ls='--')
    a.set_title(cl+' (solid: wind into front, dashed: along front)'); a.set_xlabel('minutes after arrival'); a.set_ylabel('body-weighted full UTCI (°C)'); a.legend(fontsize=8)
fig.savefig('out/transient_model2.png',dpi=140,bbox_inches='tight')
