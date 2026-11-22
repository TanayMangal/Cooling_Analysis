HiMCM 2026 Problem B - ROUND 8: Delhi (hot-dry) and Dubai (hot-humid). STANDALONE - works on a fresh computer.
==============================================================================================================

SETUP (once)
1. Unzip anywhere (right-click > Extract All). You get a HiMCM_WPI folder.
   Use a DIFFERENT folder from round 7 if both run on the same computer.
2. Double-click 1_SETUP.bat. It makes a Python environment and installs the packages (5-10 minutes,
   needs internet). It should end with "SETUP OK". If it cannot make a Python environment, install
   Python 3.11 from python.org (tick "Add python.exe to PATH") and run 1_SETUP.bat again.

RUN
3. Double-click 2_RUN.bat and leave the window open. 3_STATUS.bat shows progress, 4_STOP.bat stops.
   Finished runs are saved; running 2_RUN.bat again skips them and continues.
4. When it says "Finished", zip BOTH folders and bring them back:
     model2_advanced\out
     model1_basic\out

HOW LONG
  Half of the logical CPUs, 2 per run; each run takes about 2-2.5 hours (rail and bus), longer for warm-up reruns.
  128 logical CPUs (WPI machine): about 10-12 hours.   32 logical CPUs: about 2 days.

WHAT RUNS (117 jobs + automatic warm-up reruns), most important first
  1. Model 1 (Part 2a) for Phoenix, Miami, Delhi and Dubai (a few minutes)
  2. Rail platform, Model 2: 6 systems x 3 winds x 2 cities                      36 runs
  3. Bus shelter, Model 2: 4 systems x 3 winds x 2 cities, 15-minute wait        24 runs
  4. Rail sensitivity, wind from the track, both cities: mixing 0.01 and 0.04,
     wind 1 and 6 m/s, pad efficiency 0.65 and 0.85 (coolers only)               56 runs
  5. Rail runs whose warm-up did not settle are rerun with 1500 s, as in the paper (added automatically)

CITY INPUTS (hottest-month normals, 1991-2020)
  Delhi, May: 40 C, 28 % RH (17:30 IST), wind 2.8 m/s           [IMD normals]
  Dubai, Aug: 42 C, 44 % RH (from the 27.2 C mean dew point), wind 3.4 m/s
  Sun: the same hazy-summer beam for both (750 W/m2 direct)     [verify]

OUTPUTS (model2_advanced\out)
  rail_results.jsonl, bus_results.jsonl, rail_table.csv, bus_table.csv, *_bars.png, *_utci_maps.png
  rail_sens_results.jsonl, rail_sensitivity.csv, rail_sensitivity_rank.csv, rail_cost_sensitivity.csv, rail_sensitivity.png
