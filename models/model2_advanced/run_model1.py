"""Round 7: rerun Model 1 (Part 2a) with the output-path fix and the electricity-price cases, then the figure data."""
import os, subprocess, sys
M1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model1_basic')
for script in ('analysis.py', 'fig2a_data.py'):
    print('running', script, flush=True)
    subprocess.run([sys.executable, script], cwd=M1, check=True)
print('model 1 done', flush=True)
