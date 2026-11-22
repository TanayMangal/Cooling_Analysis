"""Compare the three semi-Lagrangian variants on a solid-body-rotation test:
a Gaussian temperature blob carried one full turn should come back unchanged."""
import numpy as np, time
from advect import Grid, advection_matrix
N=32; L=1.0
G=Grid((0,0,0),(L/N,)*3,(N,N,N))
c=np.array([0.5,0.5]); X=G.X
omega=2*np.pi                      # one revolution per second
U=np.zeros_like(X); U[:,0]=-omega*(X[:,1]-c[1]); U[:,1]=omega*(X[:,0]-c[0])
def blob(cx,cy): return np.exp(-((X[:,0]-cx)**2+(X[:,1]-cy)**2+(X[:,2]-0.5)**2)/(2*0.1**2))
T0=blob(0.75,0.5)
res={}
for dt_steps in [40,120]:
    dt=1.0/dt_steps
    for sch in ['fwd_idw','bwd_idw','bwd_tri']:
        t=time.time(); S,_=advection_matrix(G,U,dt,sch); build=time.time()-t
        T=T0.copy()
        for _ in range(dt_steps): T=S@T
        err=np.sqrt(np.mean((T-T0)**2))/np.sqrt(np.mean(T0**2))
        res[(sch,dt_steps)]=(err,T.max()/T0.max(),T.sum()/T0.sum()-1,build)
        print(f"{sch:8s} steps={dt_steps:4d} (CFL~{omega*0.25*dt/(L/N):.2f}) relL2err={err:.3f} peak kept={T.max()/T0.max():.3f} total-heat drift={100*(T.sum()/T0.sum()-1):+.1f}%  build {build:.2f}s")
# translation test, CFL 2.5 (large time step)
U2=np.zeros_like(X); U2[:,0]=1.0
dt=2.5*(L/N); steps=4
T0b=np.exp(-((X[:,0]-0.3)**2+(X[:,1]-0.5)**2+(X[:,2]-0.5)**2)/(2*0.1**2))
Tex=np.exp(-((X[:,0]-0.3-steps*dt)**2+(X[:,1]-0.5)**2+(X[:,2]-0.5)**2)/(2*0.1**2))
for sch in ['fwd_idw','bwd_idw','bwd_tri']:
    S,_=advection_matrix(G,U2,dt,sch); T=T0b.copy()
    for _ in range(steps): T=S@T
    print(f"translate CFL 2.5 {sch:8s} relL2err={np.sqrt(np.mean((T-Tex)**2))/np.sqrt(np.mean(Tex**2)):.3f} peak={T.max():.3f}")
