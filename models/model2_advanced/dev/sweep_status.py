"""Progress of the Model 2 sweep: finished runs and the last line of every job log."""
import glob, json, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for site, total in (('rail', 48), ('bus', 8)):
    fn = f'out/{site}_results.jsonl'
    n = sum(1 for _ in open(fn)) if os.path.exists(fn) else 0
    print(f'{site}: {n}/{total} runs finished')
for f in sorted(glob.glob('out/logs/*.log')):
    lines = [l.strip() for l in open(f, errors='ignore') if l.strip() and 'Warning' not in l and 'np.clip' not in l]
    print(os.path.basename(f)[:-4].ljust(24), (lines[-1][:150] if lines else '(starting)'))
