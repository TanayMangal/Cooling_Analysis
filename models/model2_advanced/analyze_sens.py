"""Round 7 analysis: Part 2b sensitivity tables and figure from out/rail_sens_results.jsonl.
Outputs (out/):
  rail_sensitivity.csv        every setting x system x city: dC, cost, eps, change vs the base setting
  rail_sensitivity_rank.csv   per setting and city: systems ranked by eps and by dC, and whether B stays on top
  rail_cost_sensitivity.csv   no-rerun checks on the base runs: electricity price and the spot AC's assumed EER
  rail_sensitivity.png        dC and eps per system for every setting, both cities"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
fn = 'out/rail_sens_results.jsonl'
if not os.path.exists(fn):
    raise SystemExit('no out/rail_sens_results.jsonl yet')
d = pd.DataFrame([json.loads(l) for l in open(fn) if l.strip()])
# Round 8: the base setting is not rerun; it is the production run of the same city and system
# (rail_results.jsonl: wind from the track, dt 0.1 s, 5-min wait, 600 s warm-up, same code).
if os.path.exists('out/rail_results.jsonl'):
    p_ = pd.DataFrame([json.loads(l) for l in open('out/rail_results.jsonl') if l.strip()])
    if 'eq_tmax' not in p_:
        p_['eq_tmax'] = 600.0
    p_ = p_[(p_.wind == 'front') & (p_.dt == 0.1) & (p_.wait_min == 5.0) & (p_.scheme == 'bwd') & (p_.eq_tmax.fillna(600.0) == 600.0)]
    have = set(zip(d.climate[d.sens_param == 'base'], d.system[d.sens_param == 'base']))
    p_ = p_[np.array([(c, s_) not in have for c, s_ in zip(p_.climate, p_.system)], bool) & p_.climate.isin(set(d.climate)).values]
    if len(p_):
        p_ = p_.drop_duplicates(['climate', 'system'], keep='last').assign(sens_param='base', sens_value=np.nan)
        d = pd.concat([d, p_], ignore_index=True)
d['sens_value'] = d['sens_value'].astype(float)
d = d.drop_duplicates(['climate', 'system', 'sens_param', 'sens_value'], keep='last')
d['setting'] = np.where(d.sens_param == 'base', 'base', d.sens_param + '=' + d.sens_value.map(lambda v: f'{v:g}'))
C1, C2 = 0.14, 0.002
rows = []
for (cl, setting), g in d.groupby(['climate', 'setting']):
    par = g.sens_param.iloc[0]
    if par == 'eta':      # pad efficiency does not change the no-device platform: use the base no-device run
        base = d[(d.climate == cl) & (d.sens_param == 'base') & (d.system == 'None')]
    else:
        base = g[g.system == 'None']
    if not len(base):
        print('no baseline for', cl, setting); continue
    Cb = float(base.C.iloc[0])
    for _, r in g[g.system != 'None'].iterrows():
        dC = Cb - r.C
        rows.append(dict(climate=cl, setting=setting, param=par, value=r.sens_value, system=r.system, C_base=Cb, C=r.C,
                         dC=dC, kW=r.kW, water_L_h=r.water_L_h, cost=r.cost, eps=dC / r.cost if r.cost > 0 else np.nan,
                         t_eq=r.t_eq, converged=r.eq_converged))
s = pd.DataFrame(rows)
ref = s[s.setting == 'base'].set_index(['climate', 'system'])[['dC', 'eps']]
s = s.join(ref, on=['climate', 'system'], rsuffix='_base_setting')
s['dC_change'] = s.dC - s.dC_base_setting; s['eps_change'] = s.eps - s.eps_base_setting
s.sort_values(['climate', 'param', 'value', 'system']).round(3).to_csv('out/rail_sensitivity.csv', index=False)
print(s.sort_values(['climate', 'param', 'value', 'system'])[['climate', 'setting', 'system', 'dC', 'eps', 'dC_change', 'eps_change']].round(2).to_string())

rk = []
for (cl, setting), g in s.groupby(['climate', 'setting']):
    by_eps = g.sort_values('eps', ascending=False).system.str.split().str[0].tolist()
    by_dC = g.sort_values('dC', ascending=False).system.str.split().str[0].tolist()
    rk.append(dict(climate=cl, setting=setting, order_by_eps=' > '.join(by_eps), order_by_dC=' > '.join(by_dC),
                   B_top_eps=by_eps[0] == 'B', B_top_dC=by_dC[0] == 'B'))
rk = pd.DataFrame(rk); rk.to_csv('out/rail_sensitivity_rank.csv', index=False); print(rk.to_string())

# no-rerun checks on the base runs: electricity price and the spot AC's assumed EER (10 -> 6 kW per unit)
b = d[d.sens_param == 'base']; cost_rows = []
for cl, g in b.groupby('climate'):
    Cb = float(g[g.system == 'None'].C.iloc[0])
    for c1 in (0.10, 0.14, 0.20, 0.30):
        for eer in (8.0, 10.0, 12.0):
            for _, r in g[g.system != 'None'].iterrows():
                kW = r.kW
                if r.system.startswith('E'):                     # 2 MovinCool K63, 60,000 BTU/h each
                    kW = kW - 2 * 6.0 + 2 * 60000 / eer / 1000
                cost = c1 * kW + C2 * r.water_L_h; dC = Cb - r.C
                cost_rows.append(dict(climate=cl, c1=c1, EER=eer, system=r.system, kW=kW, cost=cost, dC=dC, eps=dC / cost))
c = pd.DataFrame(cost_rows); c.round(3).to_csv('out/rail_cost_sensitivity.csv', index=False)
top = c.loc[c.groupby(['climate', 'c1', 'EER']).eps.idxmax()][['climate', 'c1', 'EER', 'system', 'eps']]
print('\nbest eps under each price / EER:'); print(top.round(2).to_string())

sets = sorted(s.setting.unique(), key=lambda x: (x != 'base', x)); sysn = sorted(s.system.unique())
fig, ax = plt.subplots(2, 2, figsize=(15, 8)); wb = 0.8 / len(sets); x = np.arange(len(sysn))
for j, cl in enumerate(sorted(s.climate.unique())):
    for i, v in enumerate(('dC', 'eps')):
        a_ = ax[i, j]
        for k, st in enumerate(sets):
            g = s[(s.climate == cl) & (s.setting == st)].set_index('system').reindex(sysn)
            a_.bar(x + (k - (len(sets) - 1) / 2) * wb, g[v], wb, label=st)
        a_.set_xticks(x); a_.set_xticklabels(sysn, fontsize=8); a_.axhline(0, color='k', lw=0.6)
        a_.set_title(f'{cl}: {"comfort gain dC (°C)" if v == "dC" else "efficiency eps (°C per $/h)"}', fontsize=9)
ax[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle('Part 2b sensitivity: rail platform, wind from the track (600 s warm-up, 5-min wait)')
fig.savefig('out/rail_sensitivity.png', dpi=140, bbox_inches='tight')
print('rail sensitivity analysed:', len(d), 'runs')
