"""Device energy checks. (1) One full step: total enthalpy change caused by the device vs.
its spec output, at three dt. (2) After the jet develops: air leaving the outlet cells vs. the
device's supply temperature."""
import numpy as np, copy
from m2core import *
from m2run import SPEC, CLIMATES
from shelter_model import RHO, CP, H_FG
cl = CLIMATES['Phoenix (hot-dry)']
cases = {
 'Spot AC (K63)': Device('K63', 'spotac', (1.5, 0.375, 1.375), 1, +1, (0.5, 0.25), SPEC['spotac_k63']['flow'], 6.0,
                         Q_kW=SPEC['spotac_k63']['Q_kW'], exhaust_at=(1.5, 0.75, 3.0)),
 'Evap (Jetstream 250)': Device('JS250', 'evap', (1.5, 0.75, 0.95), 1, +1, (0.625, 0.625), SPEC['evap_js250']['flow'], 1.2, eta=0.75),
}
V = None
for name, dev in cases.items():
    m = Model2(bus_shelter(), cl, 'front', devices=[dev]); V = m.g.h ** 3
    while m.time < 15: m.step(0.05)
    st = m.dev_state[0]; dn = st['dn']
    for dt in (0.1, 0.05, 0.01):
        H = []
        for on in (False, True):
            mm = copy.deepcopy(m)
            if not on:
                mm.dev_state = []; mm.devices = []
                # keep the same fixed outlet velocities (flow field identical), only drop the device's effect
            mm.step(dt); H.append((mm.T.copy(), mm.W.copy()))
        dS = RHO * CP * V * (H[0][0] - H[1][0]).sum() / dt / 1000
        dL = RHO * H_FG * V * (H[0][1] - H[1][1]).sum() / dt / 1000
        Ts, Ws, Qs, Mw = list(mm.dev_supply.values())[0]
        print(f'{name:22s} dt={dt:5.2f}  sensible removed {dS:7.2f} kW (spec {Qs/1000:6.2f})  latent {dL:+7.2f} kW (spec {-Mw*H_FG/1000:+6.2f})  total {dS+dL:6.2f} kW', flush=True)
    # outlet air temperature after development
    Tout = m.T.ravel()[dn].mean(); Ts = list(m.dev_supply.values())[0][0]
    print(f'{name:22s} outlet cells T = {Tout:.2f} C vs device supply T_sup = {Ts:.2f} C', flush=True)
