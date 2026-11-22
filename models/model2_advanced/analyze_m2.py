"""Tables and figures for Model 2 results (rail platform 2b, bus-shelter check, dt study)."""
import json, os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from m2core import rail_canopy, bus_shelter, utci_full, Model2
from shelter_model import rh_from_W
OUT = 'out'
pd.set_option('display.width', 250)


KEYS = ['site', 'climate', 'wind', 'dt', 'wait_min', 'scheme']     # a device run is compared only to a baseline with all of these equal
PRODS = dict(rail=dict(dt=0.1, wait_min=5.0, scheme='bwd'),        # production settings used for rankings
             bus=dict(dt=0.05, wait_min=15.0, scheme='bwd'))          # bus: 15-min wait as in Part 2a


def load(site):
    fn = f'{OUT}/{site}_results.jsonl'
    if not os.path.exists(fn):
        return None
    d = pd.DataFrame([json.loads(l) for l in open(fn)])
    d['eq_tmax'] = d['eq_tmax'].fillna(600.0) if 'eq_tmax' in d else 600.0
    if 'wait_min' not in d:
        d['wait_min'] = 5.0
    # a rerun with the SAME warm-up cap replaces the older row; runs with a longer warm-up are KEPT
    # alongside the 600 s rows so the change in relief can be reported (warmup_check)
    d = d.drop_duplicates(subset=KEYS + ['eq_tmax', 'system'], keep='last')
    base = d[d.system == 'None'].sort_values('eq_tmax').drop_duplicates(KEYS, keep='last')
    dup = base[base.duplicated(KEYS, keep=False)]
    if len(dup):
        raise ValueError(f'duplicate baselines for {site}:\n{dup[KEYS]}')
    d = d.merge(base[KEYS + ['C']].rename(columns={'C': 'C_base'}), on=KEYS, how='left', validate='many_to_one')
    miss = d[d.C_base.isna()]
    if len(miss):
        print(f'WARNING {site}: {len(miss)} run(s) have no matching baseline, dC left blank:')
        print(miss[KEYS + ['system']].to_string())
    d['dC'] = d.C_base - d.C
    d['eps'] = np.where(d.cost > 0, d.dC / d.cost.replace(0, np.nan), np.nan)
    # runs made before the exhaust-counter fix divided warm-up + wait exhaust energy by the wait only
    if 'exhaust_kW_removed' in d:
        old = ~d.get('exhaust_wait_only', pd.Series(False, index=d.index)).fillna(False).astype(bool)
        wait_s = d.wait_min * 60
        d.loc[old, 'exhaust_kW_removed'] = d.loc[old, 'exhaust_kW_removed'] * wait_s[old] / (d.t_eq[old] + wait_s[old])
    return d


def table(site):
    d = load(site)
    if d is None:
        return None
    cols = ['climate', 'wind', 'system', 'dt', 'wait_min', 'eq_tmax', 't_eq', 'eq_converged', 'T_occ', 'RH_occ', 'v_occ', 'utci_occ', 'C', 'dC', 'kW', 'water_L_h',
            'water_evap_frac', 'cost', 'eps', 'Tsk_end', 'Tcore_end', 'Tcore_max', 'sweat_g_per_person']
    d = d.reindex(columns=cols); d.to_csv(f'{OUT}/{site}_table.csv', index=False)
    print(d.round(3).to_string())
    return d


def production(d, site):
    PROD = PRODS[site]
    m = np.ones(len(d), bool)
    for k, v in PROD.items():
        m &= np.isclose(d[k], v) if isinstance(v, float) else (d[k] == v).values
    return d[m]


def latest(d):
    """one row per case: the run with the longest warm-up (the best available estimate)"""
    return d.sort_values('eq_tmax').drop_duplicates(KEYS + ['system'], keep='last')


def warmup_check(d, site):
    """Cases run with more than one warm-up cap: relief at each cap, the change, and whether each run met
    the equilibrium criterion. Finishing the longer warm-up is NOT the same as meeting the criterion."""
    p = production(d[d.system != 'None'], site)
    multi = p.groupby(KEYS + ['system']).eq_tmax.transform('nunique') > 1
    p = p[multi]
    if not len(p):
        return None
    rows = []
    for k, g in p.groupby(['climate', 'wind', 'system']):
        g = g.sort_values('eq_tmax'); r = dict(climate=k[0], wind=k[1], system=k[2])
        for _, x in g.iterrows():
            t = int(x.eq_tmax); r[f'dC_eq{t}'] = x.dC; r[f'converged_eq{t}'] = bool(x.eq_converged)
            ch = x.eq_change if isinstance(x.eq_change, dict) else {}
            r[f'dTmean_last_window_eq{t}'] = ch.get('dT_mean'); r[f'dTmax_last_window_eq{t}'] = ch.get('dT_max')
        r['change_first_to_last'] = g.dC.iloc[-1] - g.dC.iloc[0]
        r['change_last_two'] = g.dC.iloc[-1] - g.dC.iloc[-2]   # still moving between the two longest warm-ups?
        r['settled'] = bool(g.eq_converged.iloc[-1]); rows.append(r)
    w = pd.DataFrame(rows); w.round(3).to_csv(f'{OUT}/{site}_warmup_check.csv', index=False)
    print(f'\n{site}: warm-up check ({(~w.settled).sum()} of {len(w)} cases still do NOT meet the criterion '
          f'at their longest warm-up)'); print(w.round(3).to_string())
    return w


def rank_flags(d, site, w=None):
    """Order of systems per climate/wind (longest warm-up rows). A neighbouring pair is flagged
    'order_unsettled' when either run missed the equilibrium criterion and their gap is smaller than the
    warm-up drift seen for those systems (or, if a system was never rerun, the largest drift seen)."""
    p = latest(production(d[d.system != 'None'], site))
    drift = {}
    if w is not None and len(w):
        for _, r in w.iterrows():
            drift[(r.climate, r.wind, r.system)] = abs(r.change_last_two)
    dmax = max(drift.values()) if drift else 0.4
    out = []
    for (cl, wd), g in p.groupby(['climate', 'wind']):
        g = g.sort_values('dC', ascending=False).reset_index(drop=True)
        for i, r in g.iterrows():
            dr = drift.get((cl, wd, r.system), 0.0 if r.eq_converged else dmax)
            flag = False
            for j in (i - 1, i + 1):
                if 0 <= j < len(g):
                    o = g.loc[j]; dro = drift.get((cl, wd, o.system), 0.0 if o.eq_converged else dmax)
                    if (not r.eq_converged or not o.eq_converged) and abs(r.dC - o.dC) < max(dr, dro):
                        flag = True
            out.append(dict(climate=cl, wind=wd, rank=i + 1, system=r.system, dC=r.dC, eq_tmax=r.eq_tmax,
                            converged=bool(r.eq_converged), drift=dr, order_unsettled=flag))
    o = pd.DataFrame(out); o.round(3).to_csv(f'{OUT}/{site}_rank_flags.csv', index=False)
    print(f'\n{site}: rankings ({o.order_unsettled.sum()} positions depend on unsettled runs)'); print(o.round(3).to_string())
    return o


def scheme_check(d, site):
    """The team's forward (scatter) transport vs the backward (semi-Lagrangian + BFECC) one, same cases."""
    f = d[d.scheme == 'fwd']
    if not len(f):
        return None
    k = ['climate', 'wind', 'system', 'dt', 'wait_min']
    m = f.merge(d[d.scheme == 'bwd'], on=k, suffixes=('_fwd', '_bwd'))
    cols = k + [c for c in ('C_fwd', 'C_bwd', 'dC_fwd', 'dC_bwd', 'T_occ_fwd', 'T_occ_bwd', 'v_occ_fwd', 'v_occ_bwd',
                            'eq_converged_fwd', 'eq_converged_bwd') if c in m]
    m[cols].round(3).to_csv(f'{OUT}/{site}_scheme_check.csv', index=False)
    print(f'\n{site}: forward vs backward transport'); print(m[cols].round(3).to_string())
    return m


def bars(d, site):
    """Production runs only (one dt / wait / scheme). Within each climate every system is averaged over
    the SAME wind directions (those all systems have); whiskers = best/worst of those directions.
    Hatched bars: at least one averaged run did not meet the equilibrium criterion."""
    d = latest(production(d[d.system != 'None'], site))
    sysn = list(dict.fromkeys(d.system))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6)); notes = []
    clims = sorted(d.climate.unique())
    common = {}
    for cl in clims:
        dc = d[d.climate == cl]
        sets = [set(dc[dc.system == s].wind) for s in sysn]
        common[cl] = sorted(set.intersection(*sets)) if all(sets) else []
        notes.append(f'{cl.split()[0]}: {", ".join(common[cl]) or "none"}')
    for a, col, lab in zip(ax, ['dC', 'eps'], ['comfort gain \u0394C (\u00b0C UTCI, body-weighted)', '\u03b5 = \u0394C / ($/h)']):
        x = np.arange(len(sysn)); wdt = 0.38
        for i, cl in enumerate(clims):
            dc = d[(d.climate == cl) & d.wind.isin(common[cl])]
            g = dc.groupby('system')
            mean, lo, hi = (g[col].mean().reindex(sysn), g[col].min().reindex(sysn), g[col].max().reindex(sysn))
            unconv = (~g.eq_converged.all()).reindex(sysn).fillna(False)
            bars_ = a.bar(x + (i - 0.5) * wdt, mean, wdt, label=cl, color=f'C{i}')
            for b, u in zip(bars_, unconv):
                if u:
                    b.set_hatch('///'); b.set_edgecolor('k')
            a.errorbar(x + (i - 0.5) * wdt, mean, yerr=[mean - lo, hi - mean], fmt='none', ecolor='k', capsize=3, lw=1)
        a.set_xticks(x); a.set_xticklabels(sysn, rotation=12, fontsize=8); a.set_ylabel(lab); a.axhline(0, color='k', lw=0.6)
        a.legend(fontsize=8)
    PROD = PRODS[site]
    fig.suptitle(f'Model 2 \u2013 {site} (dt {PROD["dt"]} s, {PROD["wait_min"]:.0f}-min wait). Mean over the same wind directions for every system '
                 f'[{"; ".join(notes)}]\nwhiskers = best/worst direction; longest warm-up per case; hatched = equilibrium criterion not met in at least one run', fontsize=9)
    fig.savefig(f'{OUT}/{site}_bars.png', dpi=140, bbox_inches='tight')


def per_wind(d, site):
    """Every production run, one row per system, one column per climate/wind (nothing averaged)."""
    p = latest(production(d[d.system != 'None'], site)).pivot_table(index='system', columns=['climate', 'wind'], values='dC')
    p.round(2).to_csv(f'{OUT}/{site}_dC_by_wind.csv'); print(p.round(2).to_string())
    return p


def field_maps(site, climate='Phoenix', z=1.1):
    files = sorted(glob.glob(f'{OUT}/{site}_hist/{climate}_front_*_fields.npz'))
    if not files:
        return
    geom = rail_canopy() if site == 'rail' else bus_shelter()
    from m2run import CLIMATES
    cl = next(v for k_, v in CLIMATES.items() if k_.startswith(climate))
    m = Model2(geom, cl, 'front')
    k = int((z - geom.lo[2]) / geom.h)
    n = len(files); fig, ax = plt.subplots(n, 1, figsize=(10, 2.6 * n), squeeze=False); fig.subplots_adjust(hspace=0.55)
    for a, f in zip(ax[:, 0], files):
        F = np.load(f)
        U = utci_full(F['T'], F['Tmrt'], rh_from_W(F['T'], F['W']), np.linalg.norm(F['U'], axis=-1))
        im = a.imshow(U[:, :, k].T, origin='lower', extent=[geom.lo[0], geom.hi[0], geom.lo[1], geom.hi[1]],
                      cmap='inferno', vmin=cl['T_out'] - 7, vmax=cl['T_out'] + 9, aspect='equal')
        sp = F['U'][:, :, k]; X, Y = np.meshgrid(m.xc, m.yc, indexing='ij'); s = 3
        a.quiver(X[::s, ::s], Y[::s, ::s], sp[::s, ::s, 0], sp[::s, ::s, 1], color='w', scale=60, width=0.0015)
        (x0, x1), (y0, y1), _ = geom.box
        a.plot([x0, x1], [y0, y0], 'c', lw=2); a.set_xlim(geom.lo[0], geom.hi[0])
        name = os.path.basename(f).replace(f'{climate}_front_', '').replace('_fields.npz', '').split('_dt')[0].replace('_', ' ')
        a.set_title(f'{name}: full UTCI at z = {z} m (end of wait)', fontsize=9); a.set_ylabel('y (m)')
    ax[-1, 0].set_xlabel('x (m)  (back wall at y = 0, track side open at y = 4)')
    fig.colorbar(im, ax=ax, label='UTCI (°C)', shrink=0.6)
    fig.savefig(f'{OUT}/{site}_{climate}_utci_maps.png', dpi=130, bbox_inches='tight')


def dt_table():
    fn = f'{OUT}/dt_study.jsonl'
    if not os.path.exists(fn):
        return
    d = pd.DataFrame([json.loads(l) for l in open(fn)])
    p = d.pivot_table(index=['scheme', 'dt'], columns='system', values=['T_occ', 'utci_occ', 'v_occ'])
    p[('drop', 'B')] = p[('T_occ', 'None')] - p[('T_occ', 'B Evap')]
    print(p.round(3)); p.to_csv(f'{OUT}/dt_study_table.csv')


if __name__ == '__main__':
    for site in ['rail', 'bus']:
        d = load(site)
        if d is not None and len(d):
            table(site); bars(d, site); per_wind(d, site)
            w = warmup_check(d, site); rank_flags(d, site, w); scheme_check(d, site)
            for cl in sorted({c.split()[0] for c in d.climate}):
                try:
                    field_maps(site, cl)
                except Exception as e:                    # a figure must never stop the tables
                    print('field map skipped for', site, cl, ':', e)
    dt_table()
