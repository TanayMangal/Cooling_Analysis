"""Rotation + translation tests incl. the team's forward scheme with tent (CIC) weights."""
import numpy as np
from advect import Grid, advection_matrix
N=32; L=1.0
G=Grid((0,0,0),(L/N,)*3,(N,N,N)); X=G.X
U=np.zeros_like(X); U[:,0]=-2*np.pi*(X[:,1]-.5); U[:,1]=2*np.pi*(X[:,0]-.5)
b=lambda cx,cy: np.exp(-((X[:,0]-cx)**2+(X[:,1]-cy)**2+(X[:,2]-.5)**2)/(2*.1**2))
T0=b(.75,.5)
S_=['fwd_idw','fwd_cic','bwd_tri']
print('rotation, one full turn')
for steps in [40,120,400]:
    for sch in S_:
        S,_=advection_matrix(G,U,1/steps,sch); T=T0.copy()
        for _ in range(steps): T=S@T
        print(f'  {sch:8s} CFL~{2*np.pi*.25/steps/(L/N):.2f}  relL2 {np.sqrt(np.mean((T-T0)**2))/np.sqrt(np.mean(T0**2)):.3f}  peak {T.max()/T0.max():.3f}  heat drift {100*(T.sum()/T0.sum()-1):+.1f}%')
print('translation, x by 0.3')
U2=np.zeros_like(X); U2[:,0]=1
T0b=np.exp(-((X[:,0]-.3)**2+(X[:,1]-.5)**2+(X[:,2]-.5)**2)/(2*.1**2)); Tex=np.exp(-((X[:,0]-.6)**2+(X[:,1]-.5)**2+(X[:,2]-.5)**2)/(2*.1**2))
for cfl in [2.5,1.0,0.3,0.05]:
    dt=cfl*L/N; steps=int(round(0.3/dt)); dt=0.3/steps
    for sch in S_:
        S,_=advection_matrix(G,U2,dt,sch); T=T0b.copy()
        for _ in range(steps): T=S@T
        print(f'  {sch:8s} CFL {cfl:4.2f}  relL2 {np.sqrt(np.mean((T-Tex)**2))/np.sqrt(np.mean(Tex**2)):.3f}  peak {T.max():.3f}  centroid x {np.sum(T*X[:,0])/T.sum():.3f} (exact 0.600)')
