# HiMCM 2026 – Problem B: How Much Cooling Reaches People?

A UTCI-based model of cooling devices (misters, evaporative coolers, fans) in open and semi-open spaces: bus shelters, rail platforms, stadiums and farmers markets. Team control number 18782.

## Layout

```
models/
  part1_freezer/      Part 1: open freezer case energy / food-life model
  model1_basic/       Model 1 (Part 2a): bus shelter, no wind, diffusion + transfer matrix, 17-segment JOS-3 UTCI
  model2_advanced/    Model 2 (Part 2b/3): buoyant wind (Stable Fluids), semi-Lagrangian transport,
                      per-cell UTCI, venues (rail, bus, stadium, market). Code from the final round-8 run.
    cad/              voxelized venue geometry
    dev/              time-step studies, energy checks, older launchers (not used for final results)
  README_models_original.txt   original run instructions for both models
scripts/wpi/          Windows batch files used to run the big sweep on the WPI machine
archive/              superseded first attempts (frozen-flow Model 2, early shelter model)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Or on Windows just double-click `scripts/wpi/1_SETUP.bat`.

## Running

```
cd models/model1_basic && python analysis.py          # Model 1 results + figures -> out/
cd models/model2_advanced && python run_m2.py rail     # quick Model 2 rail run
python launch_wpi.py                                   # full round-8 sweep (Delhi + Dubai), resumable
python analyze_m2.py                                   # tables + figures from out/
```

Results are written to each model's `out/` folder, which is git-ignored. The raw outputs from the final runs are zipped in `../05_Run_Results/`.
