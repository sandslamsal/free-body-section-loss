"""The missing cell: a pointwise residual restricted to the band.

Section 3 compares a pointwise residual evaluated over the whole member
against an integrated resultant taken over the tie band, and attributes the
difference to the mismatch between the objective's support and the
parameter's. That comparison varies two things at once, the support and the
form of the functional, so the attribution is not isolated by it.

This script completes the design. The same pointwise squared residual is
evaluated over the band alone, and the same integrated resultant over the
whole member. If restricting the residual to the band removes the false
minimum, support is the operative variable and Proposition 3 is the
explanation. If the false minimum survives restriction, it is not, and the
claim has to be weakened to the pair of functionals actually tested.

Run:  python band_restricted.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from recover_utils import bracket_root
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

from figdata import cell_fields, BAND, NX, NY                              # noqa: E402
from problem import DeepBeam                                               # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
DELTA = "3.5"
BAND_ROWS = 3          # 150 mm of a 1000 mm depth on a 20-row grid


def residual_field(prob, xy, u, theta):
    f = cell_fields(prob, xy, u, theta)
    dx, dy = prob.L / NX, prob.H / NY
    r1 = np.gradient(f["sx"], dx, axis=1) + np.gradient(f["txy"], dy, axis=0)
    r2 = np.gradient(f["txy"], dx, axis=1) + np.gradient(f["sy"], dy, axis=0)
    return (r1 ** 2 + r2 ** 2)[:, 1:-1], f


def resultant(prob, f, rows):
    """Integrated sigma_x over a set of rows, in kN."""
    return float((f["sx"][rows, :] * (prob.H / NY) * prob.t).sum() / NX / 1e3)


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    xy = d["xy"]
    trial = np.linspace(0.0, 0.70, 29)
    allrows = np.ones(NY, bool)
    bandrows = np.zeros(NY, bool); bandrows[:BAND_ROWS] = True

    print("minimizer of each objective, against the generating value\n")
    print(f"{'true':>6}{'residual over':>16}{'residual over':>16}"
          f"{'resultant over':>17}{'resultant over':>17}")
    print(f"{'':>6}{'the member':>16}{'the band':>16}"
          f"{'the band':>17}{'the member':>17}")
    for th in [float(t) for t in d["theta_true"]]:
        k = f"u_{th:.2f}_{DELTA}"
        if k not in d.files:
            continue
        u = d[k]
        Rw, Rb, Tb, Tm = [], [], [], []
        for q in trial:
            sq, f = residual_field(prob, xy, u, q)
            Rw.append(sq[1:-1].mean())
            Rb.append(sq[:BAND_ROWS].mean())
            Tb.append(abs(resultant(prob, f, bandrows)))
            Tm.append(abs(resultant(prob, f, allrows)))
        def loc(v, kind):
            v = np.asarray(v)
            return trial[v.argmin()] if kind == 'min' else np.nan
        # the resultants are monotone, so their identifying value is the
        # root of T(q) = T(true), reported as the crossing
        def cross(v):
            v = np.asarray(v); t = np.interp(th, trial, v)
            return bracket_root(np.asarray(v) - t, trial)
        print(f"{th:>6.2f}{loc(Rw,'min'):>16.3f}{loc(Rb,'min'):>16.3f}"
              f"{cross(Tb):>17.3f}{cross(Tm):>17.3f}")
    print("\nColumns 2 and 3 differ only in the support of the same")
    print("functional. If they agree, support is not the operative variable.")


if __name__ == "__main__":
    main()
