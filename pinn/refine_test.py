"""Does mesh refinement repair the sectional shortfall?

Section 6.3 reports that the moment of the element stresses on a cut falls
short of what statics requires, by 8 % at the intact state and 16 % at 40 %
section loss, and that this shortfall is what makes the band-strain recovery
biased. The first explanation offered, that a low-order element cannot carry
the through-depth gradient, is contradicted by the quadrilateral fields:
a bilinear element represents a linear gradient exactly and does not help.

The alternative reading is that the shortfall belongs to the recovery
operation rather than to the element, since both discretizations form the
sectional quantity by integrating stresses recovered from elements and
neither does so through the assembly that satisfies equilibrium. That
reading predicts the shortfall will NOT vanish under refinement. This script
tests the prediction on the one axis the element comparison could not reach,
by solving the same two states on a mesh with 2.25 times the elements and
measuring the same quantity.

Run:  python refine_test.py
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

from arclength_oracle import build_mesh                                     # noqa: E402
from arclength_oracle_crisfield import newton_displacement_control          # noqa: E402
from csfm_constitutive import membrane                                      # noqa: E402
from identify import rho_x_of_theta                                         # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                          # noqa: E402
from problem import DeepBeam                                                # noqa: E402
from recover_utils import element_strains                                   # noqa: E402

X_CUT, BAND, DELTA = 700.0, 150.0, 3.5


def shortfall(prob_t, xy, u, theta, lam, nx, ny):
    """Sectional moment from element stresses against what statics needs."""
    cx, cy, ex, ey, gxy = element_strains(xy, u, nx, ny)
    col = np.abs(cx - X_CUT) < (2000.0 / nx)
    X = torch.tensor(cx[col]).unsqueeze(-1)
    Y = torch.tensor(cy[col]).unsqueeze(-1)
    st = membrane(torch.tensor(ex[col]).unsqueeze(-1),
                  torch.tensor(ey[col]).unsqueeze(-1),
                  torch.tensor(gxy[col]).unsqueeze(-1),
                  rho_x_of_theta(prob_t, X, Y, torch.tensor(float(theta))),
                  prob_t.rho_y(X, Y), prob_t.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy()
    ys = cy[col]
    dy = 1000.0 / ny
    n_col = max(1, int(round(col.sum() / ny)))
    wT = np.clip(sx * (ys < BAND), 0.0, None)
    yT = float((wT * ys).sum() / max(wT.sum(), 1e-9))
    M_int = float((sx * (ys - yT) * dy * prob_t.t).sum()) / n_col / 1e6
    M_req = lam * prob_t.P / 2.0 * (X_CUT - 370.0) / 1e6
    return abs(M_int), M_req, 100.0 * (abs(M_int) - M_req) / M_req


def main() -> None:
    prob_t = DeepBeam()
    base = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    print(f"{'mesh':>9}{'theta':>7}{'|M_int|':>10}{'M_req':>9}{'shortfall':>11}")
    for th in (0.0, 0.40):
        lam = float(base[f"lam_{th:.2f}_{DELTA}"][0])
        Mi, Mr, sh = shortfall(prob_t, base["xy"], base[f"u_{th:.2f}_{DELTA}"],
                               th, lam, 40, 20)
        print(f"{'40x20':>9}{th:>7.2f}{Mi:>10.1f}{Mr:>9.1f}{sh:>10.1f} %",
              flush=True)

    for th in (0.0, 0.40):
        p = dataclasses.replace(deepbeam_rho(RHO_NOM * (1.0 - th)),
                                nx=60, ny=30)
        mesh = build_mesh(p)
        t0 = time.time()
        hist = newton_displacement_control(p, mesh, delta_max=DELTA,
                                           n_steps=max(6, int(DELTA * 8)),
                                           verbose=False)
        last = hist[-1]
        u = np.asarray(last.u).ravel()
        lam = float(last.lam)
        Mi, Mr, sh = shortfall(prob_t, mesh.xy, u.reshape(-1, 2), th, lam,
                               60, 30)
        print(f"{'60x30':>9}{th:>7.2f}{Mi:>10.1f}{Mr:>9.1f}{sh:>10.1f} %"
              f"   [{time.time()-t0:.0f} s]", flush=True)
    print("\nIf the shortfall is unchanged under refinement the recovery")
    print("operation is implicated, not the element. If it shrinks, it is")
    print("discretization after all and the text must say so.")


if __name__ == "__main__":
    main()
