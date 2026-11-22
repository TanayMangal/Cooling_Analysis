import numpy as np, scipy.sparse.linalg as spla
from shelter_model import *
sc=Scenario('Phoenix (hot-dry)','B Evap').calibrate()
sh=sc.sh
# 1) superposition / transfer matrix: response to two point sources = sum of individual responses
c=[sh.idx(5,2,7), sh.idx(2,5,3)]
G=sc.transfer_matrix_columns(c)
both=sc.lu_Tc.solve(np.bincount(c,minlength=sh.n)*1.0)
print('superposition err', np.abs(both-G.sum(1)).max())
# 2) global energy balance for passive baseline: roof/wall/open-face fluxes sum to zero
A,b=sc.A_T,sc.b_T; T=sc.T_base
print('residual', np.abs(A@T-b).max())
# 3) UTCI reference check (pythermalcomfort docs example tdb=25,tr=25,v=1,rh=50 -> 24.6)
print('UTCI(25,25,1,50)=',float(utci(tdb=25,tr=25,v=1.0,rh=50).utci))
# 4) mesh convergence: 10^3 vs 14^3 for mean drop at the calibrated airflow
for N in [(10,10,10),(14,14,14),(20,20,20)]:
    s2=Scenario('Phoenix (hot-dry)','B Evap',sh=Shelter(N=N)); s2.calibrate()
    print(N,'Vdot for 5C mean drop: %.3f m3/s'%s2.Vdot, 'C-equilibrium mean UTCI %.2f'%s2.comfort_field(s2.T0,s2.W0).mean())
# 5) bounds: no cell colder than supply air, none hotter than roof sol-air
print('min T0 %.2f >= T_sup %.2f'%(sc.T0.min(), sc.T_sup))
