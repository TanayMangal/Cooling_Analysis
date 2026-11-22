HiMCM 2026 Problem B - cooling models

Setup (Python 3.10+):  pip install pythermalcomfort numpy scipy pandas matplotlib numba pyamg

model1_basic/                 MODEL 1 (basic, Part 2a): bus shelter, no wind, diffusion + transfer matrix,
                              no-wind UTCI, 17-segment JOS-3 weighting.
    python model1_basic/analysis.py      results, sensitivity, figures -> model1_basic/out
    (cd model1_basic; python verify.py)  checks

model2_advanced/              MODEL 2 (advanced, Part 2b): Model 1 + resolved buoyant wind (Stable Fluids),
                              semi-Lagrangian heat/humidity transport (backward + team's forward scheme),
                              full UTCI per cell, two-way JOS-3 bodies, real devices from spec sheets.
    Run everything from inside model2_advanced/ :
    python run_m2.py rail                                   quick set: wind into track side, 3-min wait, dt 0.1 s
    python run_m2.py rail --winds front along behind --wait 15 --dt 0.05     FULL SWEEP (big machine)
    python run_m2.py bus --dt 0.05                          2a bus-shelter check (compare with Model 1)
    python dt_study.py                                      time-step study down to 0.01 s
    python dt_quick.py bwd 0.1 0.05 0.025 0.01              quick time-step check (backward); 'fwd' = team scheme
    python analyze_m2.py                                    tables + figures -> model2_advanced/out
    Runs are resumable: finished cases are skipped (results in out/<site>_results.jsonl).
    Each rail run is ~120k cells; on a 2-core machine ~15-25 min per run at dt 0.1 s.
    Tip: run different climates in separate processes, e.g.
        python run_m2.py rail --climates "Phoenix (hot-dry)" &  python run_m2.py rail --climates "Miami (hot-humid)"

model2_prototype_frozen_flow/ first Model 2 attempt (frozen wind, no buoyancy). Superseded; kept for the
                              advection-scheme tests and the record of what was fixed.

Methods + results writeup: the Claude doc "Problem 2a Shelter Model: Methods & Equations"
(tabs: Model 1 (basic, 2a) and Model 2 (advanced, 2b)); download as Word or PDF from the doc.
