"""Progress of the WPI sweep (round 8): python status_wpi.py"""
import json, os, glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ev = [json.loads(l) for l in open('out/logs/wpi_status_r8.jsonl')] if os.path.exists('out/logs/wpi_status_r8.jsonl') else []
started = {e['job']: e for e in ev if e['event'] == 'started'}
done = {e['job']: e for e in ev if e['event'] == 'finished'}
print(f"elapsed {ev[-1]['elapsed_h'] if ev else 0:.2f} h | started {len(started)} | finished {len(done)} | running {len(set(started) - set(done))}")
bad = [j for j, e in done.items() if e['code'] != 0]
if bad:
    print('FAILED (see out/logs/<job>.log):', bad)
for j in sorted(set(started) - set(done)):
    try:
        last = open(f'out/logs/{j}.log').read().strip().splitlines()[-1][-120:]
    except Exception:
        last = ''
    print(f'  {j:40s} {last}')
for f in sorted(glob.glob('out/*_results.jsonl')):
    print(f, sum(1 for _ in open(f)), 'rows')
