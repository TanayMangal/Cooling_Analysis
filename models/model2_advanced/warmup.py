"""Compile the numba kernels once (cached in __pycache__) before many runs start at the same time."""
import time
from m2core import Model2, bus_shelter
from m2run import CLIMATES
t = time.time()
m = Model2(bus_shelter(), CLIMATES['Phoenix (hot-dry)'], 'front')
for _ in range(3):
    m.step(0.1)
print(f'kernels compiled ({time.time() - t:.0f} s)', flush=True)
