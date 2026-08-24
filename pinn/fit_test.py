"""Can the field be fitted before the parameter is released?

The first attempt at the network experiment let the field and the parameter
train together from the start. The parameter converged, the reconciliation
was satisfied to 8e-4, and the answer was wrong by 27 percentage points,
because the data loss was still twenty-five times its noise floor: with the
field free and unfitted, the parameter absorbs the field error rather than
the deterioration. That is a confound in the protocol, not a property of
either objective, and it has to be removed before the two can be compared.

This script measures what removing it costs. It fits the strain and
displacement fields on measurements and compatibility alone, with theta held
fixed, and reports how close the data loss gets to the floor set by the
measurement noise. Nothing can be concluded from the objective comparison
until that ratio is near one.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import pinn_experiment as E                                                # noqa: E402
import pinn_inverse as P                                                   # noqa: E402
from problem import DeepBeam                                               # noqa: E402


def fit(prob, meas, width, depth, n_iter, lr=3e-3, seed=0, n_coll=512):
    torch.manual_seed(seed)
    e_net = P.StrainPINN(width=width, depth=depth, n_out=3)
    u_net = P.StrainPINN(width=width, depth=depth, n_out=2)
    opt = torch.optim.Adam(list(e_net.parameters()) + list(u_net.parameters()),
                           lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter, eta_min=lr / 30)
    gen = torch.Generator().manual_seed(seed + 1)
    t0 = time.time()
    for it in range(n_iter):
        opt.zero_grad()
        x, y = prob.interior(n_coll, gen)
        x.requires_grad_(True); y.requires_grad_(True)
        ld = P.data_loss(e_net, prob, meas)
        lc = P.compat_loss(e_net, u_net, prob, x, y)
        (ld + lc).backward()
        opt.step(); sched.step()
        if it % 2000 == 0 or it == n_iter - 1:
            print(f"    it {it:6d}  data {float(ld):.5f}  "
                  f"({float(ld)/meas['data_floor']:6.1f}x floor)  "
                  f"compat {float(lc):.5f}  [{time.time()-t0:.0f} s]", flush=True)
    return e_net, u_net, float(ld) / meas["data_floor"]


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    prob = DeepBeam()
    fine = E.fine_from_solver(prob, d["xy"], d["u_0.30_3.5"])
    meas = P.make_measurements(prob, fine, 240, 0.02, seed=7)
    print(f"{meas['n']} gauges, floor = {meas['data_floor']:.5f}\n")
    for width, depth, n_iter in ((64, 4, 12000), (96, 4, 6000)):
        print(f"  width {width}, depth {depth}, {n_iter} iterations")
        _, _, ratio = fit(prob, meas, width, depth, n_iter)
        verdict = "USABLE" if ratio < 3.0 else "NOT FITTED"
        print(f"    -> {ratio:.1f}x floor   {verdict}\n", flush=True)


if __name__ == "__main__":
    main()
