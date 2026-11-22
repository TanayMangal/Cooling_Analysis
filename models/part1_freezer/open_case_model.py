"""Part 1: an OPEN supermarket freezer case (vertical, low temperature, air curtain, no doors).
Lumped product temperature:  C dT/dt = Q_inf + Q_rad + Q_cond + Q_int - COP * P_el * s(t)
Calibration at rating conditions (store 23.9 C / 55 % RH [verify AHRI 1200], product at -18 C):
  * daily electricity = DOE limit for VOP.RC.L, 2.2*TDA + 6.85 kWh/day (10 CFR 431.66(e))
  * load split: infiltration 73 %, radiation 12 %, lights 8 %, conduction 4 %, fans 3 %
    (Faramarzi 1999 via Sarhadian et al. 2004, measured on an open multi-deck case)
Cases (same case hardware):
  1  calm aisle, rating conditions, thermostat
  2  next to the store entrance in summer: cross-draft (+20 % air-curtain infiltration [assume]) and
     a warmer, more humid store (25 C / 60 % [assume]); SAME compressor schedule as case 1
  3  as case 2, with its own thermostat (uses more electricity)"""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

TDA = 25.0                                  # ft2 total display area of an 8-ft case [assume]
E_rate = 2.2 * TDA + 6.85                   # kWh/day (DOE VOP.RC.L)
COP = 1.6                                   # [assume] low-temperature remote system
T_safe = -18.0
T_set, band = -19.5, 1.0                    # thermostat keeps product at -20.5..-18.5 C [assume]
C = 300.0 * 2000.0                          # 300 kg frozen product x 2.0 kJ/kgK [assume]
split = dict(inf=0.73, rad=0.12, lights=0.08, cond=0.04, fans=0.03)

def psat(T):
    return np.where(T >= 0, 610.94 * np.exp(17.625 * T / (T + 243.04)), 611.15 * np.exp(22.452 * T / (T + 272.55)))
def h_air(T, RH):
    W = 0.622 * RH / 100 * psat(T) / (101325 - RH / 100 * psat(T))
    return 1.006 * T + W * (2501 + 1.86 * T)          # kJ/kg

T0, RH0 = 23.9, 55.0
Q_tot = COP * E_rate * 3.6e6 / 86400                  # W average heat removed at rating
m_inf = split['inf'] * Q_tot / ((h_air(T0, RH0) - h_air(T_safe, 100.0)) * 1000)   # kg/s through the curtain
k_rad = split['rad'] * Q_tot / ((T0 + 273.15) ** 4 - (T_safe + 273.15) ** 4)        # W/K^4
UA = split['cond'] * Q_tot / (T0 - T_safe)
Q_int = (split['lights'] + split['fans']) * Q_tot                                  # W, electric heat inside
P_el = 1.6 * E_rate * 1000 / 24                        # W compressor power, sized 60 % above the rated average [assume]

def load(T, Ts, RHs, f_inf):
    q_inf = f_inf * m_inf * (h_air(Ts, RHs) - h_air(T, 100.0)) * 1000
    return q_inf, k_rad * ((Ts + 273.15) ** 4 - (T + 273.15) ** 4) + UA * (Ts - T) + Q_int

dt = 10.0; t = np.arange(0, 86400 + dt, dt)
def run(Ts, RHs, f_inf, schedule=None):
    T = np.empty_like(t); T[0] = T_set; s = np.zeros_like(t); on = False; spill = 0.0
    for k in range(1, len(t)):
        if schedule is None:
            on = True if T[k-1] > T_set + band else (False if T[k-1] < T_set - band else on); s[k] = on
        else:
            s[k] = schedule[k]
        qi, qo = load(T[k-1], Ts, RHs, f_inf); spill += qi * dt
        T[k] = T[k-1] + (qi + qo - COP * P_el * s[k]) * dt / C
    return T, s, spill / 86400

kWh = lambda s: P_el * s.sum() * dt / 3.6e6 + Q_int * 24 / 1000 / 1.0 * 0   # compressor electricity (lights/fans counted below)
E_aux = Q_int * 24 / 1000                                                     # lights + fans electricity, kWh/day
T1, s1, sp1 = run(T0, RH0, 1.0)
T2, _, sp2 = run(25.0, 60.0, 1.2, schedule=s1)
T3, s3, sp3 = run(25.0, 60.0, 1.2)
cases = []
for name, T, s, sp in (('1 calm aisle', T1, s1, sp1), ('2 draft + summer, same electricity', T2, s1, sp2),
                       ('3 draft + summer, own thermostat', T3, s3, sp3)):
    E = P_el * s.sum() * dt / 3.6e6
    hrs = float((T > T_safe).sum() * dt / 3600)
    safe_frac = 1 - hrs / 24
    cases.append(dict(case=name, compressor_kWh_day=E, hours_above=hrs, T_max=float(T.max()),
                      spill_to_aisle_kW=sp / 1000, safe_display_ft2_per_kWh=TDA * safe_frac / E,
                      energy_only_kWh_per_ft2=E / TDA))
out = dict(TDA_ft2=TDA, DOE_limit_kWh_day=E_rate, Q_tot_W=Q_tot, m_inf_kg_s=m_inf, P_el_W=P_el, cases=cases)
json.dump(out, open('open_case_results.json', 'w'), indent=1)
for c in cases:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})
print('Q_tot', round(Q_tot), 'W; P_el', round(P_el), 'W; m_inf', round(m_inf, 4), 'kg/s')

# figure: (a) electricity and efficiency, (b) hourly product temperature
h = t / 3600
hourly = lambda T: np.array([T[(h >= i) & (h < i + 1)].mean() for i in range(24)])
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(width_ratios=[1.15, 2]))
lab = ['1: calm aisle', '2: draft + summer,\nsame electricity', '3: draft + summer,\nmore electricity']
col = ['C0', 'C3', 'C2']
E = [c['compressor_kWh_day'] for c in cases]; eff = [c['safe_display_ft2_per_kWh'] for c in cases]
b = ax[0].bar(range(3), E, color=col, width=0.6)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(lab, fontsize=8); ax[0].set_ylabel('compressor electricity (kWh/day)')
for i, (e, f) in enumerate(zip(E, eff)):
    ax[0].text(i, e + 1, f'ε = {f:.2f}', ha='center', fontsize=8)
ax[0].set_ylim(0, max(E) * 1.18); ax[0].set_title('(a) electricity, and ε = safe display ft² per kWh/day', fontsize=9)
hh = np.arange(24) + 0.5
for T, c_, l in ((T1, 'C0', 'case 1'), (T2, 'C3', 'case 2 (same electricity as 1)'), (T3, 'C2', 'case 3')):
    ax[1].plot(hh, hourly(T), color=c_, marker='o', ms=3, lw=1.5, label=l)
ax[1].axhline(T_safe, color='k', ls='--', lw=0.8); ax[1].text(0.3, T_safe + 0.2, '−18 °C food-safety line', fontsize=8)
ax[1].fill_between(hh, T_safe, hourly(T2), where=hourly(T2) > T_safe, color='C3', alpha=0.15)
ax[1].set_xlim(0, 24); ax[1].set_xlabel('time of day (h)'); ax[1].set_ylabel('product temperature, hourly mean (°C)')
ax[1].set_title('(b) product temperature', fontsize=9); ax[1].legend(fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig('open_case.pdf'); fig.savefig('open_case.png', dpi=140)
