"""5-minute demo: v1 vs v2 device model on the 2a bus shelter (Phoenix, wind into the front,
one Jetstream 250), 20 s of simulated time at dt = 0.1 and 0.05 s."""
import time, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from m2core import *
from m2run import SPEC, CLIMATES
from shelter_model import RHO, CP, H_FG, rh_from_W, wet_bulb_stull

class V1(Model2):
    """v1 behaviour: outlet cells RESET to supply state every step, supply air also carried
    across the outlet; plain (first-order) transport."""
    def _apply_devices(self, dt):
        Tf = self.T.ravel(); Wf = self.W.ravel(); self.dev_log = []
        for st in self.dev_state:
            dv = st['dev']; up, dn = st['up'], st['dn']
            Ti, Wi = Tf[up].mean(), Wf[up].mean()
            Twb = float(wet_bulb_stull(Ti, float(rh_from_W(Ti, Wi))))
            Ts = Ti - dv.spec['eta'] * (Ti - Twb); Ws = Wi + CP * (Ti - Ts) / H_FG
            reset_W = RHO * CP * self.g.h ** 3 * (Tf[dn] - Ts).sum() / dt      # energy the reset removes
            self.reset_kW = getattr(self, 'reset_kW', []) + [reset_W / 1000]
            Tf[dn] = Ts; Wf[dn] = Ws
            self.dev_vals[0] = (Ts, Ws)
        self.T = Tf.reshape(self.T.shape); self.W = Wf.reshape(self.W.shape)

cl = CLIMATES['Phoenix (hot-dry)']
dev = lambda: Device('Jetstream 250', 'evap', (1.5, 0.75, 0.95), 1, +1, (0.625, 0.625), SPEC['evap_js250']['flow'], 1.2, eta=0.75)
runs = {}
t0 = time.time()
for label, cls, kw in [('v1 (reset, 1st-order)', V1, dict(bfecc=False)), ('v2 (exact source, BFECC)', Model2, dict(bfecc=True))]:
    for dt in (0.1, 0.05):
        m = cls(bus_shelter(), cl, 'front', devices=[dev()], **kw); occ = m.occupied(); hist = []
        m0 = Model2(bus_shelter(), cl, 'front') if False else None
        while m.time < 20 - 1e-9:
            m.step(dt)
            if int(round(m.time / dt)) % int(round(0.5 / dt)) == 0:
                hist.append((m.time, m.T[occ].mean()))
        runs[(label, dt)] = (m, np.array(hist))
        rk = np.mean(m.reset_kW[-50:]) if hasattr(m, 'reset_kW') else None
        print(f'{label:26s} dt={dt}: occupied air {m.T[occ].mean():.2f} C, min air {m.T.min():.1f} C'
              + (f', reset alone removes {rk:.0f} kW (device output ~56 kW)' if rk else '')
              + (f', cooling placed {100*np.mean(m.placed_frac[-50:]):.0f}% of spec' if hasattr(m, 'placed_frac') else ''), f'[{time.time()-t0:.0f}s]', flush=True)
# uncooled reference
m = Model2(bus_shelter(), cl, 'front'); occ = m.occupied()
while m.time < 20 - 1e-9: m.step(0.1)
Tref = m.T[occ].mean(); print(f'no device: occupied air {Tref:.2f} C', flush=True)

fig = plt.figure(figsize=(14, 8.5)); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15], hspace=0.35, wspace=0.3)
for r, label in enumerate(['v1 (reset, 1st-order)', 'v2 (exact source, BFECC)']):
    for c, dt in enumerate((0.1, 0.05)):
        m = runs[(label, dt)][0]; G = m.g; i = int((1.5 - G.lo[0]) / G.h)
        a = fig.add_subplot(gs[r, c])
        im = a.imshow(m.T[i].T, origin='lower', extent=[G.lo[1], G.hi[1], G.lo[2], G.hi[2]], cmap='coolwarm', vmin=24, vmax=46, aspect='auto')
        U = m.speed()
        a.plot([0, 0], [0, 2.5], 'k', lw=3); a.plot([0, 1.5], [2.5, 2.5], 'k', lw=3)
        a.add_patch(plt.Rectangle((0.0, 0.5), 0.75, 0.0, fill=False))
        a.axhspan(0.5, 1.75, xmin=(0 - G.lo[1]) / (G.hi[1] - G.lo[1]), xmax=(1.5 - G.lo[1]) / (G.hi[1] - G.lo[1]), color='k', alpha=0.06)
        a.set_title(f'{label}\nΔt = {dt} s: occupied air {m.T[m.occupied()].mean():.1f} °C', fontsize=9)
        a.set_xlabel('y (m): back wall 0, open front 1.5'); a.set_ylabel('z (m)')
fig.colorbar(im, ax=fig.axes, shrink=0.6, label='air temperature (°C), side view through the cooler', location='left', pad=0.08)
a = fig.add_subplot(gs[:, 2])
for (label, dt), (m, h) in runs.items():
    a.plot(h[:, 0], h[:, 1], ls='-' if dt == 0.1 else '--', color='C0' if label.startswith('v1') else 'C3', label=f'{label}, Δt {dt}')
a.axhline(Tref, color='k', lw=1, ls=':', label='no device')
a.set_xlabel('time after switch-on (s)'); a.set_ylabel('occupied-zone air temperature (°C)')
a.set_title('Bus shelter, Phoenix, Jetstream 250', fontsize=10); a.legend(fontsize=8)
fig.suptitle('Device model v1 vs v2 (20 s demo, 0.125 m cells). Shaded band = occupied zone 0.5–1.75 m', fontsize=11)
fig.savefig('out/demo_v1_v2.png', dpi=140, bbox_inches='tight'); print('saved', f'[{time.time()-t0:.0f}s]')
