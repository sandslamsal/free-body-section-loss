"""The identification solved jointly over the field and the parameter.

Section 3 evaluates both objectives on a field held fixed at the generating
value. That is the right setting for an observable computed from a
measurement, but it is not what an inverse formulation does when the field
is recovered together with the parameter: there the field moves to
accommodate a trial value, anchored by a data term, and a false minimum
found on a frozen field need not survive.

This script tests that directly. It solves

    minimize over (eps, theta):  || eps - eps_meas ||^2 + lam * R(eps, theta)

with eps a free field on the grid the reference is given on, theta a free
scalar, and R the pointwise squared equilibrium residual formed through the
constitutive map. No network is involved: the field is its own
parameterisation, which removes the approximator as a confounder and makes
the outcome a property of the objective. The residual weight is swept over
four decades, because the honest form of the claim concerns
residual-dominated objectives and where that behavior sets in is part of it.

Run:  python joint_inverse.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

from csfm_constitutive import membrane                                     # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "joint_inverse.json"
NX, NY, BAND, DELTA = 40, 20, 150.0, "3.5"
THETA_MAX = 0.70


def cell_grid(xy, u):
    cx, cy, ex, ey, gxy = element_strains(xy, u, NX, NY)
    return [0.5 * (a[0::2] + a[1::2]).reshape(NY, NX)
            for a in (cx, cy, ex, ey, gxy)]


def residual(eps, cy, theta, prob):
    ex, ey, gxy = eps[0], eps[1], eps[2]
    in_band = (cy < BAND).to(ex.dtype)
    rho_x = prob.rho_min + (prob.rho_tie * (1.0 - theta) - prob.rho_min) * in_band
    rho_y = torch.full_like(ex, prob.rho_min + prob.rho_stirrup)
    st = membrane(ex, ey, gxy, rho_x, rho_y, prob.mat, soften=True)
    sx, sy, txy = st["sigma_x"], st["sigma_y"], st["tau_xy"]
    dx, dy = prob.L / NX, prob.H / NY
    dsx = (sx[1:-1, 2:] - sx[1:-1, :-2]) / (2 * dx)
    dtxy_y = (txy[2:, 1:-1] - txy[:-2, 1:-1]) / (2 * dy)
    dtxy_x = (txy[1:-1, 2:] - txy[1:-1, :-2]) / (2 * dx)
    dsy = (sy[2:, 1:-1] - sy[:-2, 1:-1]) / (2 * dy)
    return ((dsx + dtxy_y) ** 2 + (dtxy_x + dsy) ** 2).mean()


def solve(prob, cy_t, eps_meas, lam, n_iter=2500, seed=0):
    torch.manual_seed(seed)
    eps = eps_meas.clone().requires_grad_(True)
    z = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([{"params": [eps], "lr": 2e-5},
                            {"params": [z], "lr": 2e-2}])
    for _ in range(n_iter):
        opt.zero_grad()
        th = THETA_MAX * torch.sigmoid(z)[0]
        data = ((eps - eps_meas) ** 2).mean() / (eps_meas ** 2).mean()
        (data + lam * residual(eps, cy_t, th, prob)).backward()
        opt.step()
    with torch.no_grad():
        return (float(THETA_MAX * torch.sigmoid(z)[0]),
                float(((eps - eps_meas) ** 2).mean() / (eps_meas ** 2).mean()))


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    lams = [1e-2, 1e-1, 1e0, 1e1, 1e2]
    rows = []
    print("joint minimisation over the field and the parameter\n")
    print(f"{'true':>6}" + "".join(f"{f'lam={l:g}':>11}" for l in lams)
          + "     data misfit at largest lam")
    for th_true in [float(t) for t in d["theta_true"]]:
        k = f"u_{th_true:.2f}_{DELTA}"
        if k not in d.files:
            continue
        cx, cy, ex, ey, gxy = cell_grid(d["xy"], d[k])
        cy_t = torch.tensor(cy)
        eps_meas = torch.tensor(np.stack([ex, ey, gxy]))
        out, mis = [], None
        for lam in lams:
            th, data = solve(prob, cy_t, eps_meas, lam)
            out.append(th); mis = data
            rows.append({"true": th_true, "lam": lam, "theta_hat": th,
                         "data_misfit": data})
        print(f"{th_true:>6.2f}" + "".join(f"{v:>11.3f}" for v in out)
              + f"{mis:>18.2e}", flush=True)
    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
