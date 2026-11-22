"""Start the full Model 2 sweep as independent background jobs (Windows).
6 rail jobs (one per climate x wind, each runs all 6 systems) + 2 bus-shelter check jobs.
Math libraries are kept to 1 thread (OpenBLAS otherwise reserves memory for every core);
the solver's own kernels use 4 threads per rail job. Jobs start 20 s apart.
Resumable: rerun this script and finished runs are skipped. Progress:  python sweep_status.py"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PY = sys.executable
os.makedirs('out/logs', exist_ok=True)
CL = {'phx': 'Phoenix (hot-dry)', 'mia': 'Miami (hot-humid)'}
jobs = []
for ck, cl in CL.items():
    for wd in ('front', 'along', 'behind'):
        jobs.append((f'rail_{ck}_{wd}', ['rail', '--climates', cl, '--winds', wd, '--wait', '15', '--dt', '0.05'], '4'))
for ck, cl in CL.items():
    jobs.append((f'bus_{ck}', ['bus', '--climates', cl, '--winds', 'front', '--wait', '5', '--dt', '0.05'], '2'))
flags = 0x00000008 | 0x00000200          # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
for name, args, nthreads in jobs:
    env = dict(os.environ, NUMBA_NUM_THREADS=nthreads, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    log = open(f'out/logs/{name}.log', 'a')
    p = subprocess.Popen([PY, 'run_m2.py', *args], stdout=log, stderr=subprocess.STDOUT, env=env,
                         creationflags=flags, cwd=HERE)
    print(name, 'pid', p.pid, flush=True)
    time.sleep(20)
