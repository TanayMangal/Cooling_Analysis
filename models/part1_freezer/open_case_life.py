"""Part 1 (final): the same open freezer case run in different ways. Each way uses a different amount of
electricity and gives the food a different storage life. Efficiency = (storage life gained) / (cost).
Storage life of frozen beef steaks: 8 / 18 / 24 months at -12 / -18 / -24 C (Engineering ToolBox, Frozen
Food - Storage Life), log-linear in between. Baseline (case switched off): the beef is at store temperature
and must be thrown out within hours, so its storage life is taken as 0.
Loads: open vertical low-temperature case, DOE limit 2.2*TDA + 6.85 kWh/day at -18 C (10 CFR 431.66),
load split 73 % air curtain / 12 % radiation / 4 % walls / 11 % lights+fans (Sarhadian et al. 2004)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

TDA = 25.0; E_doe = 2.2 * TDA + 6.85            # kWh/day at -18 C
c1 = 0.14                                       # $/kWh (A5)
T_s, RH_s = 23.9, 55.0                          # store, standard test conditions [verify AHRI 1200]
eta, dT_evap, T_cond = 0.4, 8.0, 40.0           # COP = eta * Carnot(T_p - 8 C, 40 C) [assume]
def cop(Tp):
    # Khurmi & Gupta: (C.O.P.)_Carnot = T2 / (T1 - T2); relative C.O.P. = actual / theoretical  ->  actual = eta * Carnot
    T2 = Tp - dT_evap + 273.15; T1 = T_cond + 273.15; return eta * T2 / (T1 - T2)
def psat(T):
    return np.where(T >= 0, 610.94 * np.exp(17.625 * T / (T + 243.04)), 611.15 * np.exp(22.452 * T / (T + 272.55)))
def h_air(T, RH):
    # Khurmi & Gupta psychrometry: W = 0.622 p_v / (p_b - p_v);  h = 1.022 t_d + W (2500 + 1.88 t_d)  kJ/kg dry air
    pv = RH / 100 * psat(T); W = 0.622 * pv / (101325 - pv); return 1.022 * T + W * (2500 + 1.88 * T)
life_T = np.array([-24.0, -18.0, -12.0]); life_m = np.array([24.0, 18.0, 8.0])
def life_days(Tp):
    return float(np.exp(np.interp(-Tp, -life_T[::-1] * -1 * -1, np.log(life_m[::-1])))) * 30.4 if False else \
        float(np.exp(np.interp(Tp, life_T, np.log(life_m)))) * 30.4

# calibrate loads at -18 C so that total electricity = DOE limit
f = dict(inf=0.73, rad=0.12, cond=0.04, int=0.11)
# total heat at -18: Q*(1-f_int)/COP + Q*f_int (lights/fans electricity) + Q*f_int/COP (their heat removed) = E_doe
c18 = cop(-18.0)
Q18 = E_doe * 1000 / 24 / (1 / c18 + f['int'])
m_inf = f['inf'] * Q18 / ((h_air(T_s, RH_s) - h_air(-18.0, 100)) * 1000)
k_r = f['rad'] * Q18 / ((T_s + 273.15) ** 4 - (255.15) ** 4)
UA = f['cond'] * Q18 / (T_s + 18.0)
Q_int = f['int'] * Q18
def daily_kWh(Tp, inf_factor=1.0):
    q = inf_factor * m_inf * (h_air(T_s, RH_s) - h_air(Tp, 100)) * 1000 + k_r * ((T_s + 273.15) ** 4 - (Tp + 273.15) ** 4) \
        + UA * (T_s - Tp) + Q_int
    return (q / cop(Tp) + Q_int) * 24 / 1000

DOOR = 0.2      # glass doors let in 20 % of the open case's infiltration [assume]
L_BASE = 4.0    # days: raw beef steaks kept chilled at about 0-4 C (USDA refrigerator guidance 3-5 days) [verify]
rows = [dict(case='Baseline: chilled at 0 C, no freezer case', T=0.0, door=None, kWh_day=0.0, cost_day=0.0, life_days=L_BASE, eff=None)]
for door, fi in (('open', 1.0), ('glass doors', DOOR)):
    for Tp in (-12.0, -18.0, -24.0):
        E = daily_kWh(Tp, fi); L = life_days(Tp)
        rows.append(dict(case=f'{door}, held at {Tp:.0f} C', T=Tp, door=door, kWh_day=E, cost_day=c1 * E, life_days=L,
                         eff=(L - L_BASE) / (c1 * E), TR=(E / 24 - Q_int / 1000) * cop(Tp) / 3.517))
json.dump(dict(rows=rows, Q18_W=Q18, COP18=c18, L_base=L_BASE), open('open_case_life.json', 'w'), indent=1)
for r in rows:
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
temps = [-12, -18, -24]; x = np.arange(3); w = 0.38
get = lambda d, key: [r[key] for r in rows[1:] if r['door'] == d]
for a, key, yl, t, fmt in ((ax[0], 'kWh_day', 'electricity (kWh/day)', '(a) resources spent', '{:.0f}'),
                           (ax[1], 'life_days', 'storage life of the beef (days)', '(b) result: storage life', '{:.0f}'),
                           (ax[2], 'eff', '(life − baseline) days per $ per day', '(c) efficiency', '{:.0f}')):
    for k, (d, c) in enumerate((('open', 'C1'), ('glass doors', 'C2'))):
        v = get(d, key); a.bar(x + (k - 0.5) * w, v, w, color=c, label=d)
        for i, vi in enumerate(v):
            a.text(x[i] + (k - 0.5) * w, vi * 1.02, fmt.format(vi), ha='center', fontsize=7)
    if key == 'life_days':
        a.axhline(L_BASE, color='k', ls='--', lw=0.8); a.text(-0.45, L_BASE + 15, f'baseline (chilled, 0 °C): {L_BASE:.0f} days', fontsize=7)
    a.set_xticks(x); a.set_xticklabels([f'{T} °C' for T in temps]); a.set_ylabel(yl); a.set_title(t, fontsize=9)
    a.set_ylim(0, max(get('open', key) + get('glass doors', key)) * 1.15)
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig('open_case_life.pdf'); fig.savefig('open_case_life.png', dpi=140)
