"""Rail sweep at the chosen settings: dt 0.1 s, 5-minute wait, 3 winds x 2 climates x 6 systems.
6 background jobs (one per climate x wind), 4 solver threads each, started 20 s apart.
Resumable: rerun to continue. Progress:  python sweep_status.py"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
os.makedirs('out/logs', exist_ok=True)
CL = {'phx': 'Phoenix (hot-dry)', 'mia': 'Miami (hot-humid)'}
flags = 0x00000008 | 0x00000200
for ck, cl in CL.items():
    for wd in ('front', 'along', 'behind'):
        name = f'rail_{ck}_{wd}'
        env = dict(os.environ, NUMBA_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        log = open(f'out/logs/{name}.log', 'a')
        p = subprocess.Popen([sys.executable, 'run_m2.py', 'rail', '--climates', cl, '--winds', wd,
                              '--wait', '5', '--dt', '0.1'], stdout=log, stderr=subprocess.STDOUT,
                             env=env, creationflags=flags, cwd=HERE)
        print(name, 'pid', p.pid, flush=True)
        time.sleep(20)
