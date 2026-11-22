"""Finish the v2 sweep and rerun the unsettled cases with a longer warm-up (review item #5).
  1 rail job  : Phoenix, wind from behind, spot AC, dt 0.1 s              (last production run)
  5 rail jobs : wind into track side, dt 0.05 s: Phoenix C/D/E, Miami D/E (time-step check)
  8 rail jobs : the 8 production runs that hit the 600 s warm-up cap, rerun with a 1500 s cap
Each job: 2 solver threads, math libraries 1 thread. Resumable. Progress: python sweep_status.py"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
os.makedirs('out/logs', exist_ok=True)
PHX, MIA = 'Phoenix (hot-dry)', 'Miami (hot-humid)'
B, C, D, E = 'B Evap x3', 'C Large evap x2', 'D Misting', 'E Spot AC + air curtain'
def job(name, cl, wd, sysn, dt, eq=600):
    return (name, ['rail', '--climates', cl, '--winds', wd, '--systems', sysn, '--wait', '5', '--dt', str(dt), '--eq-tmax', str(eq)])
jobs = [job('v3_phx_behind_E', PHX, 'behind', E, 0.1)]
for cl, ck, systems in [(PHX, 'phx', (C, D, E)), (MIA, 'mia', (D, E))]:
    for s in systems:
        jobs.append(job(f'v3_{ck}_front_dt005_{s[0]}', cl, 'front', s, 0.05))
for cl, ck, wd, s in [(PHX, 'phx', 'front', B), (MIA, 'mia', 'front', B), (MIA, 'mia', 'behind', B), (PHX, 'phx', 'behind', B),
                      (PHX, 'phx', 'front', C), (MIA, 'mia', 'behind', C), (PHX, 'phx', 'behind', C), (PHX, 'phx', 'behind', D)]:
    jobs.append(job(f'v3_{ck}_{wd}_{s[0]}_eq1500', cl, wd, s, 0.1, 1500))
only = set(sys.argv[1:])                 # optional: launch only these job names
jobs = [j for j in jobs if not only or j[0] in only]
flags = 0x00000008 | 0x00000200          # detached, own process group: survives closing this window
for name, args in jobs:
    env = dict(os.environ, NUMBA_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    log = open(f'out/logs/{name}.log', 'a')
    p = subprocess.Popen([sys.executable, 'run_m2.py', *args], stdout=log, stderr=subprocess.STDOUT,
                         env=env, creationflags=flags, cwd=HERE)
    print(name, 'pid', p.pid, flush=True)
    time.sleep(15)
