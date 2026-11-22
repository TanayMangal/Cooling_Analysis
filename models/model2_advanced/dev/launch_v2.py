"""Model 2 sweep after the review fixes (exact device energy, exhaust ducted out, field-based
equilibrium with a converged flag, BFECC transport, wait-aware run tracking).
  6 rail jobs  : 2 climates x 3 winds x 6 systems, dt 0.1 s, 5-min wait  (production)
  2 rail jobs  : 2 climates, wind into track side, 6 systems, dt 0.05 s   (time-step robustness)
  2 bus jobs   : 2 climates, wind into front, 4 systems, dt 0.05 s        (Model 1 comparison)
Each job: 3 solver threads, math libraries 1 thread. Resumable. Progress: python sweep_status.py"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
os.makedirs('out/logs', exist_ok=True)
CL = {'phx': 'Phoenix (hot-dry)', 'mia': 'Miami (hot-humid)'}
jobs = []
for ck, cl in CL.items():
    for wd in ('front', 'along', 'behind'):
        jobs.append((f'rail_{ck}_{wd}', ['rail', '--climates', cl, '--winds', wd, '--wait', '5', '--dt', '0.1']))
for ck, cl in CL.items():
    jobs.append((f'rail_{ck}_front_dt005', ['rail', '--climates', cl, '--winds', 'front', '--wait', '5', '--dt', '0.05']))
for ck, cl in CL.items():
    jobs.append((f'bus_{ck}', ['bus', '--climates', cl, '--winds', 'front', '--wait', '5', '--dt', '0.05']))
flags = 0x00000008 | 0x00000200
for name, args in jobs:
    env = dict(os.environ, NUMBA_NUM_THREADS='3', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    log = open(f'out/logs/{name}.log', 'a')
    p = subprocess.Popen([sys.executable, 'run_m2.py', *args], stdout=log, stderr=subprocess.STDOUT,
                         env=env, creationflags=flags, cwd=HERE)
    print(name, 'pid', p.pid, flush=True)
    time.sleep(20)
