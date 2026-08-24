"""Instance B: the criterion applied to a real instrumented structure.

Every other result in this directory is measured on fields a solver
generated. The criterion's sharpest claim is that the difficulty it
describes is invisible in verification against exact solutions and appears
only against real data, so the argument is incomplete until it has been put
to a real measurement.

The instance is a prediction rather than a fit. The criterion says a
functional carries information about a parameter in proportion to the
overlap between its support and the parameter's. An instrument is a
functional's support, so the same statement applies to instrumentation
before it applies to any objective: measurement that does not reach the
parameter cannot identify it, however the objective is written and however
well the optimizer converges. That is checkable from a sensor layout and a
reinforcement drawing, at no cost, before any data is processed.

Applied here it predicts failure, and the prediction is exact. It also says
what would have to change, which is the useful half.

Run:  python instance_b_wall.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from csfm_constitutive import CsfmMaterial, membrane                       # noqa: E402

DATA = HERE.parent / "data" / "wall_strain_gauges.csv"
META = HERE.parent / "data" / "wall_meta.npy"

# specimen, from Fernandez, Berrocal and Rempling (2023); see data/README.md
L, H, T_THK = 800.0, 500.0, 100.0
BAND = 50.0                       # depth the two 12 mm tie bars smear over
AS_TIE = 2.0 * np.pi * 6.0 ** 2   # 226 mm^2
RHO_TIE = AS_TIE / (BAND * T_THK)
RHO_GRID = 2.0 * np.pi * 3.0 ** 2 / (100.0 * T_THK)
MAT = CsfmMaterial(fc=32.0, fy=500.0)
THETA_MAX = 0.70


def load_grid():
    a = np.loadtxt(DATA, delimiter=",", skiprows=1)
    meta = np.load(META, allow_pickle=True).item()
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], float(meta["P_kN"])


def sx_at(y, ex, ey, gxy, rho_x):
    t = lambda v: torch.tensor(np.asarray(v, float)).unsqueeze(-1)   # noqa: E731
    st = membrane(t(ex), t(ey), t(gxy), t(rho_x),
                  torch.full_like(t(ex), RHO_GRID), MAT, soften=True)
    return st["sigma_x"].squeeze().numpy()


def main() -> None:
    x, y, ex, ey, gxy, P = load_grid()
    rows = np.unique(np.round(y))
    print(f"Fernandez wall, {x.size} gauge points, P = {P:.1f} kN")
    print(f"wall {L:.0f} x {H:.0f} x {T_THK:.0f} mm, "
          f"{len(rows)} gauge rows from y = {y.min():.0f} to {y.max():.0f} mm\n")

    # ---- 1. the check that can be made before any data is processed ----
    print("1. Support overlap, from the sensor layout and the drawing\n")
    n_band = int((y < BAND).sum())
    print(f"   tie bars smeared over y < {BAND:.0f} mm, "
          f"rho_tie = {RHO_TIE:.4f}")
    print(f"   gauges in that band: {n_band} of {x.size}")
    print(f"   lowest gauge row: y = {y.min():.0f} mm, "
          f"{y.min() - BAND:.0f} mm above the band\n")
    print("   Overlap is zero, so the measurement carries no information")
    print("   about tie section loss. No objective and no optimizer can")
    print("   recover it from this dataset.\n")

    # ---- 2. confirm it on the data, since a prediction should be tested --
    print("2. The same statement, measured\n")
    sel = np.abs(x - 240.0) < 40.0
    d = []
    for th in (0.0, 0.35, 0.70):
        rho = np.where(y[sel] < BAND, RHO_TIE * (1.0 - th), RHO_GRID)
        d.append(float(np.abs(sx_at(y[sel], ex[sel], ey[sel],
                                    gxy[sel], rho)).sum()))
    print(f"   sum |sigma_x| on a cut at theta = 0, 0.35, 0.70:  "
          f"{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f} MPa")
    print(f"   variation across the whole admissible range: "
          f"{100*(max(d)-min(d))/max(d[0], 1e-12):.2e} %\n")

    # The quantitative comparison against the other two parameters, and
    # the scoring of the criterion's predictions, is in
    # criterion_prediction.py; it is kept there so one definition of signal
    # fraction serves all three questions.

    # ---- 3. how far the instrument would have to reach -------------------
    print("3. How far the fiber would have to reach\n")
    print(f"   {'assumed band depth':>20}{'gauges inside':>15}{'coverage':>11}")
    for h in (50.0, 75.0, 100.0, 150.0, 200.0):
        inside = int((y < h).sum())
        cov = max(0.0, (min(h, y.max()) - y.min())) / h if h > y.min() else 0.0
        print(f"   {h:>18.0f}{inside:>15d}{100*cov:>10.0f} %")
    print("\n   The tie steel sits 21 mm from the soffit, so no realistic")
    print("   smearing of it reaches the lowest gauge row. Identifying tie")
    print("   corrosion by this route requires fiber bonded along the bars,")
    print("   not a grid cast at mid-thickness, and that is a statement about")
    print("   where to put the instrument rather than about what to minimize.")

    print("\n4. What the same dataset does support\n")
    print("   The companion study recovered the applied load from these same")
    print("   gauges to within 6.4 per cent. The criterion accounts for the")
    print("   difference without appeal to either method: the load acts")
    print("   everywhere the instrument reaches and scales the whole field,")
    print("   so it meets both conditions, while a reinforcement parameter")
    print("   meets neither on this layout. Same structure, same sensors,")
    print("   same data, and the question decides the answer.")


if __name__ == "__main__":
    main()
