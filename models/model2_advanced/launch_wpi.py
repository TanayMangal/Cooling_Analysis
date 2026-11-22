"""ONE-SHOT SWEEP, ROUND 8: Delhi (hot-dry) and Dubai (hot-humid), the paper's two cities. Standalone.
      python launch_wpi.py            (50% of the logical CPUs)        python launch_wpi.py --dry   (list jobs)
  1. Model 1 (Part 2a, bus shelter) for all four cities                          1 job, a few minutes
  2. Part 2b rail platform, Model 2: 6 systems x 3 winds x 2 cities, 5-min wait  36 runs
  3. Part 2a bus shelter, Model 2: 4 systems x 3 winds x 2 cities, 15-min wait   24 runs
  4. Part 2b sensitivity, wind from the track: mixing 0.01/0.04, wind 1/6 m/s,
     pad efficiency 0.65/0.85 (the base is the production run of step 2)         56 runs
  5. Automatic: every rail run from step 2 whose warm-up did not settle is rerun
     with a 1500 s warm-up, as in the paper (added as soon as step 2 is done)    up to 36 runs
Progress any time: python status_wpi.py"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=0, help='limit: workers x --threads solver threads (default: --cpu-share of the CPUs)')
ap.add_argument('--cpu-share', type=float, default=0.5, help='fraction of the machine to use (default 0.5 = 50%%)')
ap.add_argument('--threads', type=int, default=2, help='solver threads for a normal run (long runs use more)')
ap.add_argument('--hours', type=float, default=40.0)
ap.add_argument('--last-start', type=float, default=30.0, help='hours after which no new run is started')
ap.add_argument('--dry', action='store_true')
ap.add_argument('--ncpu', type=int, default=0, help='pretend this many logical CPUs (for --dry planning)')
a = ap.parse_args()
os.makedirs('out/logs', exist_ok=True)

DEL, DXB = 'Delhi (hot-dry)', 'Dubai (hot-humid)'
CITIES = ((DEL, 'del'), (DXB, 'dxb'))
WINDS = ('front', 'behind', 'along')
RAIL = ['None', 'A Fans', 'B Evap x3', 'C Large evap x2', 'D Misting', 'E Spot AC + air curtain']
BUS = ['None', 'A Fan', 'B Evap', 'C Large evap']
jobs = []          # (name, args, threads)
def m2(name, site, cl, wd, s, dt, wait, eq=600, threads=2):
    jobs.append((name, ['run_m2.py', site, '--climates', cl, '--winds', wd, '--systems', s, '--wait', str(wait),
                        '--dt', str(dt), '--scheme', 'bwd', '--eq-tmax', str(eq)], threads))
def sens(cl, ck, s, param, value, threads=2):
    jobs.append((f"8_sens_{ck}_{s.split()[0]}_{param}{value:g}",
                 ['run_sens.py', '--climate', cl, '--system', s, '--param', param, '--value', str(value)], threads))
jobs.append(('8_model1', ['run_model1.py'], 1))
for cl, ck in CITIES:
    for wd in WINDS:
        for s in RAIL:
            m2(f'8_rail_{ck}_{wd}_{s.split()[0]}', 'rail', cl, wd, s, 0.1, 5)
        for s in BUS:
            m2(f'8_bus_{ck}_{wd}_{s.split()[0]}', 'bus', cl, wd, s, 0.05, 15)
    for s in RAIL:
        for v in (0.01, 0.04):
            sens(cl, ck, s, 'nu', v)
        for v in (1.0, 6.0):
            sens(cl, ck, s, 'u10', v)
    for s in ('B Evap x3', 'C Large evap x2'):
        for v in (0.65, 0.85):
            sens(cl, ck, s, 'eta', v)
PRIO = ('model1', '8_rail_', '8_bus_', '_eta', '_nu', '_u10')          # most important first
jobs.sort(key=lambda j: next(i for i, k in enumerate(PRIO) if k in j[0]))
RAIL_PROD = {n for n, _, _ in jobs if n.startswith('8_rail_')}

def unsettled_reruns():
    """Rail production runs (600 s warm-up) that did not meet the equilibrium test -> rerun at 1500 s."""
    fn = 'out/rail_results.jsonl'
    if not os.path.exists(fn):
        return []
    rows = [json.loads(l) for l in open(fn) if l.strip()]
    have = {(r['climate'], r['wind'], r['system'], r.get('eq_tmax', 600.0)) for r in rows}
    out = []
    for r in rows:
        if (r['climate'] in (DEL, DXB) and r['dt'] == 0.1 and r.get('wait_min') == 5.0 and r.get('eq_tmax', 600.0) == 600.0
                and r['system'] != 'None' and not r.get('eq_converged', True)
                and (r['climate'], r['wind'], r['system'], 1500.0) not in have):
            ck = 'del' if r['climate'] == DEL else 'dxb'
            name = f"8_rail_{ck}_{r['wind']}_{r['system'].split()[0]}_eq1500"
            out.append((name, ['run_m2.py', 'rail', '--climates', r['climate'], '--winds', r['wind'], '--systems', r['system'],
                               '--wait', '5', '--dt', '0.1', '--scheme', 'bwd', '--eq-tmax', '1500'], 2))
    return sorted({j[0]: j for j in out}.values())

ncpu = a.ncpu or os.cpu_count() or 4
budget = a.workers * a.threads if a.workers else max(2, int(ncpu * a.cpu_share))   # solver threads in use at once
jobs = [(n, ar, min(t, budget)) for n, ar, t in jobs]
print(f'{len(jobs)} jobs, up to {budget} solver threads at once on {ncpu} logical CPUs ({100 * budget / ncpu:.0f} %), '
      f'budget {a.hours} h, no new starts after {a.last_start} h', flush=True)
if a.dry:
    for n, args, t in jobs:
        print(f'[{t} thr]', n, ' '.join(f'"{x}"' if ' ' in x else x for x in args))
    raise SystemExit

env = dict(os.environ, NUMBA_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
env_t = lambda t: dict(env, NUMBA_NUM_THREADS=str(t))
flags = (0x00000008 | 0x00000200) if os.name == 'nt' else 0      # detached + own process group on Windows

print('compiling the numerical kernels once (about 1-2 min) ...', flush=True)
subprocess.run([sys.executable, 'warmup.py'], env=env, check=True)

t0 = time.time(); queue = list(jobs); running = {}
stat = open('out/logs/wpi_status_r8.jsonl', 'a')
def log(**kw):
    kw['elapsed_h'] = round((time.time() - t0) / 3600, 3); stat.write(json.dumps(kw) + '\n'); stat.flush()
    print(kw, flush=True)
finished = set(); reruns_built = False
while queue or running:
    for n in list(running):
        p, ts, lf, t = running[n]
        if p.poll() is not None:
            lf.close(); del running[n]; finished.add(n)
            log(event='finished', job=n, code=p.returncode, minutes=round((time.time() - ts) / 60, 1))
    if not reruns_built and RAIL_PROD <= finished:
        reruns_built = True; rr = unsettled_reruns()
        queue = rr + queue
        log(event='warmup_reruns_added', n=len(rr), jobs=[r[0] for r in rr])
    el = (time.time() - t0) / 3600
    used = sum(r[3] for r in running.values())
    if el < a.last_start:
        # highest-priority job that fits in the free threads (smaller jobs fill the gaps)
        for q in list(queue):
            n, args, t = q
            if used + t <= budget:
                queue.remove(q)
                lf = open(f'out/logs/{n}.log', 'a')
                p = subprocess.Popen([sys.executable] + args, stdout=lf, stderr=subprocess.STDOUT, env=env_t(t),
                                     creationflags=flags, cwd=HERE)
                running[n] = (p, time.time(), lf, t); used += t
                log(event='started', job=n, pid=p.pid, threads=t, threads_in_use=used, waiting=len(queue))
                time.sleep(2)
    elif queue:
        log(event='stopped', reason='past last-start time', not_run=[q[0] for q in queue]); queue = []
    time.sleep(30)

log(event='all done')
failed = []
for script, cwd in (('analyze_m2.py', HERE), ('analyze_sens.py', HERE)):
    rc = subprocess.run([sys.executable, script], env=env, cwd=cwd).returncode
    if rc != 0:
        failed.append(script); log(event='analysis_failed', script=script, code=rc)
if failed:
    print('SIMULATIONS FINISHED, BUT THE ANALYSIS FAILED:', ', '.join(failed),
          '- the runs are saved; zip model2_advanced\\out and bring it back anyway (the analysis can be rerun alone).', flush=True)
    sys.exit(1)
print('Finished. Zip the model2_advanced\\out folder AND the model1_basic\\out folder and bring them back.', flush=True)
