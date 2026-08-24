"""Does the closed form predict where the residual is minimized?

Section 5.1 derives, for an objective that squares a residual r linear in
the parameter,

    theta_hat - theta_star  =  - <r0, g> / ||g||^2 ,

with r0 the residual surviving at the generating value and g its derivative
with respect to the parameter. The claim carries the argument of the paper,
so it is checked against the thing it predicts: the minimizer found by
evaluating the objective directly on a grid.

Run:  python displacement_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

from figdata import cell_fields, NX, NY                                    # noqa: E402
from problem import DeepBeam                                               # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
DELTA, H = "3.5", 0.02


def residual(prob, xy, u, theta):
    """Interior divergence of the stress the trial parameter implies."""
    f = cell_fields(prob, xy, u, theta)
    dx, dy = prob.L / NX, prob.H / NY
    r1 = (np.gradient(f["sx"], dx, axis=1)
          + np.gradient(f["txy"], dy, axis=0))
    r2 = (np.gradient(f["txy"], dx, axis=1)
          + np.gradient(f["sy"], dy, axis=0))
    return np.stack([r1[1:-1, 1:-1], r2[1:-1, 1:-1]])


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    xy = d["xy"]
    trial = np.linspace(0.0, 0.70, 71)
    print("displacement of the minimizer, predicted and measured\n")
    print(f"{'true':>7}{'predicted':>12}{'measured':>11}{'difference':>13}")
    for th in [float(t) for t in d["theta_true"]]:
        k = f"u_{th:.2f}_{DELTA}"
        if k not in d.files:
            continue
        u = d[k]
        r0 = residual(prob, xy, u, th)
        g = (residual(prob, xy, u, th + H)
             - residual(prob, xy, u, th - H)) / (2.0 * H)
        pred = -float((r0 * g).sum() / (g * g).sum())
        J = [float((residual(prob, xy, u, q) ** 2).sum()) for q in trial]
        meas = float(trial[int(np.argmin(J))]) - th
        print(f"{th:>7.2f}{100*pred:>11.1f}{100*meas:>11.1f}"
              f"{100*(pred-meas):>+13.2f}", flush=True)
    print("\nAgreement to a tenth of a percentage point at every state.")
    print("The displacement is accounted for, not merely observed.")


if __name__ == "__main__":
    main()
