"""Problem 3 tables and figures from out/<site>_results.jsonl for every venue that has results
(stands, market2 = CAD venues; stadium, market = earlier hand-built venues).
Every device run is compared with the no-device run that has the same wind, time step, wait, warm-up
cap, gate/grill/queue settings, grid, body-update interval and crowd setting. Outputs (out/):
  <site>_p3_table.csv         every run with dC overall and per group, eps, crowd eps
  <site>_p3_by_wind.csv       production runs: dC per system x wind direction
  <site>_p3_checks.csv        the same case at a finer grid / smaller dt / shorter body step / grill off:
                              how much the gains move, and whether the ranking of systems changes
  <site>_p3_bars.png          main wind: comfort gain per group and eps per system
  <site>_p3_maps.png          main wind: map of time-averaged UTCI drop vs the baseline (people's height)
  stands_worth_it.csv         stands only: the 'is AC in the stands worth it' table"""
import json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from m3venues import MAIN_WIND, VENUE_SYSTEMS
from m3cad import CAD_MAIN_WIND, CAD_SYSTEMS, SECTOR, CAD_BASE
BASE = dict(stadium='None', market='None', **CAD_BASE)
BASE_LABEL = {'market2': 'existing ceiling fans'}            # what the baseline means, for figure text

MAINW = dict(MAIN_WIND, **CAD_MAIN_WIND); SYSTEMS = dict(VENUE_SYSTEMS, **CAD_SYSTEMS)
DEF = {  # production settings per site
    'stadium': dict(dt=0.1, h=0.5, body_dt=10.0, grill=True, n_queue=12, gates='open', crowd=True),
    'market': dict(dt=0.1, h=0.5, body_dt=10.0, grill=True, n_queue=10, gates='open', crowd=True),
    'stands': dict(wait_min=15.0, eq_tmax=900.0, dt=0.2, h=2.5, body_dt=10.0, grill=True, n_queue=0, gates='cad', crowd=True),
    'market2': dict(wait_min=5.0, eq_tmax=900.0, dt=0.1, h=0.5, body_dt=2.0, grill=True, n_queue=8, gates='cad', crowd=True),
}
KEY = ['wind', 'dt', 'wait_min', 'eq_tmax', 'gates', 'grill', 'n_queue', 'h', 'body_dt', 'crowd', 'u10_set']
MAIN_GROUP = {'stands': 'sector', 'market2': 'queue', 'stadium': 'queue', 'market': 'queue'}


def load(site):
    d = pd.DataFrame([json.loads(l) for l in open(f'out/{site}_results.jsonl') if l.strip()])
    for k, v in (('h', 0.5), ('body_dt', 10.0), ('crowd', True)):
        d[k] = d[k].fillna(v) if k in d else v
    d['crowd'] = d['crowd'].astype(bool); d['grill'] = d['grill'].astype(bool)
    d['u10_set'] = d['u10_set'].fillna(-1.0).astype(float) if 'u10_set' in d else -1.0   # -1 = normal breeze
    d['rev'] = d['rev'].fillna(1) if 'rev' in d else 1
    # rev-1 runs with devices are superseded (device placement fixed in rev 2); runs without devices stay valid
    keep_old = {'market2': 'No fans', 'stands': 'None'}.get(site)
    if keep_old is not None:
        d = d[(d.rev >= 2) | (d.system == keep_old)]
    # a rerun with a newer model revision replaces the older row of the same case
    return d.sort_values('rev', kind='stable').drop_duplicates(KEY + ['system'], keep='last')


def with_gains(d, site):
    groups = sorted(c[2:] for c in d.columns if c.startswith('C_') and not c.endswith('_base'))
    base = d[d.system == BASE[site]][KEY + ['C', 'cost'] + [f'C_{g}' for g in groups]]
    d = d.merge(base, on=KEY, how='left', suffixes=('', '_base'))
    d['dC'] = d['C_base'] - d['C']
    for g in groups:
        d[f'dC_{g}'] = d[f'C_{g}_base'] - d[f'C_{g}']
    # efficiency uses the EXTRA running cost over the baseline (market2: the ceiling fans are already paid for)
    d['dcost'] = d['cost'] - d['cost_base'].fillna(0)
    d['eps'] = np.where(d.dcost.abs() > 1e-9, d.dC / d.dcost.replace(0, np.nan), np.nan)
    npeople = lambda g: (d['n_queue_people'].fillna(d['n_queue']) if (g == 'queue' and 'n_queue_people' in d) else d[f'n_{g}'])
    d['relief_people_C'] = sum(d[f'dC_{g}'].fillna(0) * npeople(g).fillna(0) for g in groups)
    d['eps_people'] = np.where(d.dcost.abs() > 1e-9, d.relief_people_C / d.dcost.replace(0, np.nan), np.nan)
    return d, groups


def is_prod(d, site, **over):
    s = {'eq_tmax': 600.0, 'u10_set': -1.0, **DEF[site], **over}; m = np.ones(len(d), bool)
    for k, v in s.items():
        m &= (d[k] == v).values
    return m


def checks(d, site):
    """each sensitivity setting vs the production run of the same system and wind"""
    g = MAIN_GROUP[site]; rows = []
    prod = d[is_prod(d, site)].set_index(['system', 'wind'])
    for name, over in (('grid h/2', dict(h=DEF[site]['h'] / 2 if site != 'stands' else 2.0)), ('dt/2', dict(dt=DEF[site]['dt'] / 2)),
                       ('body step 1 s', dict(body_dt=1.0)), ('grill off', dict(grill=False)), ('no crowd heat', dict(crowd=False)), ('stay 45 min', dict(wait_min=45.0)), ('calm day (0.5 m/s)', dict(u10_set=0.5))):
        v = d[is_prod(d, site, **over)]
        if not len(v) or (name == 'grid h/2' and site == 'stands' and not (v.h == 2.0).any()):
            continue
        for _, r in v.iterrows():
            if r.system == BASE[site] or (r.system, r.wind) not in prod.index:
                continue
            p = prod.loc[(r.system, r.wind)]
            rows.append(dict(check=name, system=r.system, wind=r.wind, dC_prod=p['dC'], dC_check=r['dC'],
                             **{f'dC_{g}_prod': p.get(f'dC_{g}'), f'dC_{g}_check': r.get(f'dC_{g}')},
                             change=r[f'dC_{g}'] - p[f'dC_{g}'] if f'dC_{g}' in r else np.nan))
    c = pd.DataFrame(rows)
    if len(c):  # does the order of systems change under each check?
        for (nm, wd), sub in c.groupby(['check', 'wind']):
            a = sub.sort_values(f'dC_{g}_prod', ascending=False).system.tolist()
            b = sub.sort_values(f'dC_{g}_check', ascending=False).system.tolist()
            c.loc[sub.index, 'ranking_same'] = a == b
    return c


def tag(site, s, wind, **over):
    p = dict(DEF[site], **over)
    if site in ('stands', 'market2'):
        return (f"{s.split()[0]}_{wind}_dt{p['dt']}_gcad_grill{int(p['grill'])}_q{p['n_queue']}_eq{p.get('eq_tmax', 600):g}_h{p['h']:g}"
                f"_b{p['body_dt']:g}_c{int(p['crowd'])}" + (f"_w{p['wait_min']:g}" if p['wait_min'] != (15.0 if site == 'stands' else 5.0) else '')
                + (f"_u{p['u10_set']:g}" if p.get('u10_set', -1) >= 0 else ''))
    return f"{s.split()[0]}_{wind}_dt0.1_gates-open_grill1_q{p['n_queue']}_eq600_h0.5"


def person_layer(F, site):
    """UTCI where people are: first air layer above the stand surface (stands) or 1.25 m (others)"""
    U = F['avg_UTCI']
    if site == 'stands':
        h = float(F['zc'][1] - F['zc'][0]); k = np.clip((F['top'] / h).astype(int), 0, U.shape[2] - 1)
        I, J = np.indices(k.shape); return U[I, J, k]
    return U[:, :, int(np.argmin(np.abs(F['zc'] - 1.25)))]


for site in ('stands', 'market2', 'stadium', 'market'):
    if not os.path.exists(f'out/{site}_results.jsonl'):
        continue
    d, groups = with_gains(load(site), site)
    d.to_csv(f'out/{site}_p3_table.csv', index=False)
    prod = d[is_prod(d, site)]
    if len(prod):
        prod.pivot_table(index='system', columns='wind', values=['dC'] + [f'dC_{g}' for g in groups] + ['eps', 'eps_people']
                         ).round(3).to_csv(f'out/{site}_p3_by_wind.csv')
    if site in ('market2', 'market'):   # how much the grill itself heats each group (same weather-based walk-in)
        on, off = d[is_prod(d, site)], d[is_prod(d, site, grill=False)]
        if len(off):
            m_ = on.merge(off, on=['system', 'wind'], suffixes=('_on', '_off'))
            cols = ['system', 'wind'] + [f'C_{g}_{k}' for g in groups for k in ('on', 'off')]
            for g in groups:
                m_[f'grill_heats_{g}'] = m_[f'C_{g}_on'] - m_[f'C_{g}_off']
            m_[[c_ for c_ in cols if c_ in m_] + [f'grill_heats_{g}' for g in groups]].round(3).to_csv(f'out/{site}_grill_effect.csv', index=False)
    if len(prod):   # which system helps which group most (queue vs people moving vs ...), per wind
        rows = []
        for wd, sub in prod[~prod.system.isin([BASE[site], 'No fans'])].groupby('wind'):
            for g in groups:
                if sub[f'dC_{g}'].notna().any():
                    b_ = sub.loc[sub[f'dC_{g}'].idxmax()]
                    rows.append(dict(wind=wd, group=g, best_system=b_.system, dC=b_[f'dC_{g}'], extra_cost_per_h=b_.dcost,
                                     eps_group=b_[f'dC_{g}'] / b_.dcost if b_.dcost > 0 else np.nan))
        pd.DataFrame(rows).round(3).to_csv(f'out/{site}_best_by_group.csv', index=False)
    if site == 'market2':   # walkers: time in the open corners of the O vs under the sheds
        rows = []
        for _, r in prod.iterrows():
            f = f'out/{site}_hist/{tag(site, r.system, r.wind)}.json'
            if not os.path.exists(f):
                continue
            H = json.load(open(f)); gr = np.array(H['groups']) == 'walk'
            xy = np.array([h_['xy'] for h_ in H['hist']])[:, gr]; C = np.array([h_['C_people'] for h_ in H['hist']])[:, gr]
            corner = (np.abs(xy[..., 0]) > 8.25) & (np.abs(xy[..., 1]) > 8.25)
            rows.append(dict(system=r.system, wind=r.wind, C_walk_open_corners=C[corner].mean(), C_walk_under_sheds=C[~corner].mean(),
                             frac_time_in_corners=corner.mean()))
        if rows:
            w_ = pd.DataFrame(rows); b_ = w_[w_.system == BASE[site]].set_index('wind')
            for k_ in ('C_walk_open_corners', 'C_walk_under_sheds'):
                w_['d' + k_[1:]] = w_.wind.map(b_[k_]) - w_[k_]
            w_.round(3).to_csv(f'out/{site}_walk_split.csv', index=False)
    # was the space steady while people were in it? (zone-mean temperature and comfort over the stay)
    rows = []
    for _, r in prod.iterrows():
        f = f'out/{site}_hist/{tag(site, r.system, r.wind)}.json'
        if os.path.exists(f):
            H = json.load(open(f))['hist']; t = np.array([x['t'] for x in H])
            T = np.array([x['T_occ'] for x in H]); Cc = np.array([x['C'] for x in H])
            blocks = [Cc[(t >= a) & (t < b)].mean() for a, b in zip(np.linspace(0, t[-1], 6)[:-1], np.linspace(0, t[-1], 6)[1:] + 1e-9)]
            rows.append(dict(system=r.system, wind=r.wind, T_occ_trend_C_per_min=np.polyfit(t, T, 1)[0],
                             C_trend_C_per_min=np.polyfit(t, Cc, 1)[0], C_block_spread=np.ptp(blocks),
                             steady_stay=bool(abs(np.polyfit(t, T, 1)[0]) < 0.03 and np.ptp(blocks) < 0.3)))
    if rows:
        pd.DataFrame(rows).round(4).to_csv(f'out/{site}_steady.csv', index=False)
    c = checks(d, site)
    if len(c):
        c.round(3).to_csv(f'out/{site}_p3_checks.csv', index=False)
    if site == 'stands' and len(prod):
        w = prod[prod.system != BASE[site]].copy()
        w['cost_per_h'] = w['cost']
        P = w['covered_people'].fillna(w['sector_people']) if 'covered_people' in w else w['sector_people']
        w['covered_people'] = P                  # spectators whose seats the system covers (whole bowl for G)
        w['spectator_C_per_$'] = w['dC_sector'] * P / w['cost']          # degree-spectators per dollar
        w['$_per_spectator_h'] = w['cost'] / P
        w['$_per_spectator_C_h'] = w['cost'] / (P * w['dC_sector'].where(w['dC_sector'] > 0))
        cols = ['system', 'wind', 'n_devices', 'kW', 'cooling_kW', 'water_L_h', 'cost_per_h', 'sector_people', 'covered_people', 'dC_sector',
                'dC_west', 'dC_north', 'eps', 'spectator_C_per_$', '$_per_spectator_h', '$_per_spectator_C_h', 'Tcore_max']
        w[[c_ for c_ in cols if c_ in w]].sort_values(['wind', 'system']).round(3).to_csv('out/stands_worth_it.csv', index=False)
    mw = prod[(prod.wind == MAINW[site]) & (prod.system != BASE[site])]
    if len(mw):
        sysn = [s for s in SYSTEMS[site] if s in set(mw.system)]
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.5)); x = np.arange(len(sysn)); wb = 0.8 / max(len(groups), 1)
        for i, g in enumerate(groups):
            ax[0].bar(x + (i - (len(groups) - 1) / 2) * wb, mw.set_index('system').reindex(sysn)[f'dC_{g}'], wb, label=g)
        ax[0].set_ylabel(f"comfort gain vs {BASE_LABEL.get(site, 'no device')} (°C UTCI, body-weighted)"); ax[0].legend(fontsize=8)
        ax[1].bar(x, mw.set_index('system').reindex(sysn)['eps'], 0.6, color='C7')
        ax[1].set_ylabel('ε = ΔC (all people) / (extra $/h over baseline)')
        for a_ in ax:
            a_.set_xticks(x); a_.set_xticklabels(sysn, rotation=15, fontsize=8); a_.axhline(0, color='k', lw=0.6)
        fig.suptitle(f'Problem 3 – {site}, wind {MAINW[site]}')
        fig.savefig(f'out/{site}_p3_bars.png', dpi=140, bbox_inches='tight'); plt.close(fig)
    bf = f'out/{site}_hist/{tag(site, BASE[site], MAINW[site])}_fields.npz'
    if os.path.exists(bf):
        B = np.load(bf); hb = float(B['xc'][1] - B['xc'][0])
        sysn = [s for s in SYSTEMS[site][1:] if os.path.exists(f'out/{site}_hist/{tag(site, s, MAINW[site])}_fields.npz')]
        ncol = 2; nrow = int(np.ceil((len(sysn) + 1) / ncol))
        fig, ax = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 5.2 * nrow), squeeze=False); ax = ax.ravel()
        ext = [B['xc'][0] - hb / 2, B['xc'][-1] + hb / 2, B['yc'][0] - hb / 2, B['yc'][-1] + hb / 2]
        UB = person_layer(B, site)
        if site == 'stands':
            UB = np.where(B['top'] > 0.5 * hb, UB, np.nan)                         # stands only (not the field/outside)
        im0 = ax[0].imshow(UB.T, origin='lower', extent=ext, cmap='inferno')
        bl = BASE_LABEL.get(site, 'no device')
        ax[0].set_title(f'{bl[0].upper() + bl[1:]}: time-averaged UTCI where people are (°C)', fontsize=9); fig.colorbar(im0, ax=ax[0], shrink=0.7)
        for a_, s in zip(ax[1:], sysn):
            F = np.load(f'out/{site}_hist/{tag(site, s, MAINW[site])}_fields.npz')
            dU = UB - person_layer(F, site)
            im = a_.imshow(dU.T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-4, vmax=4)
            a_.set_title(f'{s}: UTCI drop vs {bl} (°C, red = cooler)', fontsize=9); fig.colorbar(im, ax=a_, shrink=0.7)
        for a_ in ax[:len(sysn) + 1]:
            if site == 'stands':
                a_.add_patch(plt.Rectangle((SECTOR['x'][0], SECTOR['y'][0]), np.ptp(SECTOR['x']), np.ptp(SECTOR['y']), fill=False, ec='c'))
            if 'solid' in B:
                a_.contour(B['xc'], B['yc'], B['top'].T, levels=[0.5 * hb], colors='0.5', linewidths=0.4)
        for a_ in ax[len(sysn) + 1:]:
            a_.axis('off')
        fig.savefig(f'out/{site}_p3_maps.png', dpi=120, bbox_inches='tight'); plt.close(fig)
    print(site, len(d), 'runs analysed')
