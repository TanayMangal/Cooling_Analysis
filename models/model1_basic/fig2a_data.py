"""Data for the three Part 2a figures (Model 1, bus shelter). Cheap: runs in about a minute.
  Fig 2  fan felt-cooling vs distance, rating read as a shelter AVERAGE vs as a PEAK ('up to 3 C')
  Fig 3  top-down chest-height map of air-temperature drop (cooling off minus on), every system/climate
  Fig 4  comfort improvement per dollar for A/B/C in both climates, both fan readings
Outputs go to out/fig2a/ (CSV + NPZ). Run analysis.py first (Fig 4 reads its CSVs)."""
import os, numpy as np, pandas as pd
from shelter_model import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'fig2a'); os.makedirs(OUT, exist_ok=True)
sh = Shelter()

# ---------------- Fig 2: distance decay, two readings of the rating
src = np.array([sh.L[0] / 2, 0.0, sh.L[2] - 0.3])
dist = np.sqrt((sh.xc - src[0]) ** 2 + (sh.yc - src[1]) ** 2 + (sh.zc - src[2]) ** 2)
f = np.exp(-dist / 1.0)
d = np.linspace(dist.min(), dist.max(), 200)          # nearest cell centre to the fan .. farthest
pd.DataFrame(dict(distance_m=d,
                  felt_cooling_avg_reading_C=3.0 * np.exp(-d) / f.mean(),
                  felt_cooling_peak_reading_C=3.0 * np.exp(-d) / f.max())).to_csv(f'{OUT}/fig2_fan_decay_curves.csv', index=False)
sc = Scenario('Phoenix (hot-dry)', 'A Fan')
pos = sc.people_positions()
rows = []
for interp in ('mean', 'peak'):
    fo = fan_offset_field(sh, 3.0, 1.0, interp)
    for i, (x, y) in enumerate(pos):
        c = sh.cell_of(x, y, 1.3)
        rows.append(dict(reading=interp, person=i, x=x, y=y, distance_m=float(dist[c]), felt_cooling_C=float(fo[c])))
pd.DataFrame(rows).to_csv(f'{OUT}/fig2_fan_cooling_at_waiting_positions.csv', index=False)

# ---------------- Fig 3: chest-height air-temperature drop maps (equilibrium, before people arrive)
k = int(np.argmin(np.abs((np.arange(sh.N[2]) + 0.5) * sh.d[2] - 1.3)))
maps = {}
for cl in CLIMATES:
    for s in SYSTEMS:
        if s == 'None':
            continue
        sc = Scenario(cl, s).calibrate()
        dT = (sc.T_base - sc.T0).reshape(tuple(sh.N))[:, :, k]
        dRH = (rh_from_W(sc.T0, sc.W0) - rh_from_W(sc.T_base, sc.W_base)).reshape(tuple(sh.N))[:, :, k]
        key = f"{cl.split()[0]}_{s.split()[0]}"
        maps[key + '_dT'] = dT; maps[key + '_dRH'] = dRH
maps['x_centers'] = (np.arange(sh.N[0]) + 0.5) * sh.d[0]
maps['y_centers'] = (np.arange(sh.N[1]) + 0.5) * sh.d[1]
maps['z_layer'] = (k + 0.5) * sh.d[2]
maps['waiting_xy'] = np.array(pos)
np.savez(f'{OUT}/fig3_airdrop_maps.npz', **maps)
pd.DataFrame(maps['Phoenix_B_dT'], index=maps['x_centers'], columns=maps['y_centers']).to_csv(
    f'{OUT}/fig3_Phoenix_B_airdrop_C_rows_x_cols_y.csv')

# ---------------- Fig 4: comfort per dollar, both fan readings
res = pd.read_csv(os.path.join(HERE, 'out', 'results.csv'))
sens = pd.read_csv(os.path.join(HERE, 'out', 'sensitivity.csv'))
rows = []
for _, r in res.iterrows():
    rows.append(dict(climate=r.climate, system=r.system, fan_reading='average (main)', dC=r.dC, cost_per_h=r.cost, eps=r.eps))
for _, r in sens[sens.case == 'fan: peak not mean'].iterrows():
    rows.append(dict(climate=r.climate, system=r.system, fan_reading='peak', dC=r.dC,
                     cost_per_h=res[(res.climate == r.climate) & (res.system == r.system)].cost.iloc[0], eps=r.eps))
pd.DataFrame(rows).to_csv(f'{OUT}/fig4_efficiency.csv', index=False)
print('Part 2a figure data written to', OUT)
