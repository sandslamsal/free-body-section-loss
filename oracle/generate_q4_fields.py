"""Regenerate the reference family on quadrilaterals.

Section 3.6.5 of Kaufmann et al. (2020) specifies CQUAD4 shell elements
with four integration points for the concrete, and the diagnostics support
that choice on grounds independent of the book: on the same problem the
quadrilateral returns an axial resultant of exactly zero on a vertical
section, where the constant-strain triangle leaves 23 to 86 kN, and it
halves the discrepancy between the measured tension resultant and statics,
from -7.5 to -3.4 per cent.

Two corrections established while diagnosing the triangular reference are
carried into the identification that uses these fields, and are recorded
here because neither is visible in the code:

  * the concrete carries no tension in the CSFM, so at the strains reached
    in the tie band the yielded bar carries essentially all of it, and the
    deterioration parameter scales close to the whole tension resultant
    rather than a small share of it;
  * the bearing reaction is not centred on the support. It concentrates
    towards the inner edge, placing its centroid near 363 mm rather than
    the nominal 250 mm, which is a moment-arm error of 113 mm and was the
    entire apparent deficit in sectional moment.

Output: q4_fields_theta.npz, same schema as the triangular family.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q4_oracle as q                                                      # noqa: E402
from arclength_oracle import Material                                      # noqa: E402

RHO_NOM = 0.012
THETA = [0.0, 0.10, 0.20, 0.30, 0.40]
DELTAS = [2.0, 3.5, 5.0]
NX, NY = 40, 20


def main() -> None:
    mat = Material(fc=30.0)
    mesh = q.build_q4(2000.0, 1000.0, NX, NY, 250.0, 200.0, 800e3)
    out = {"xy": mesh.xy, "quads": mesh.quads,
           "theta_true": np.array(THETA), "deltas": np.array(DELTAS),
           "rho_nom": np.array([RHO_NOM])}
    t0 = time.time()
    for th in THETA:
        for dt in DELTAS:
            u, hist = q.solve_dc(mesh, RHO_NOM * (1.0 - th), mat, 300.0,
                                 800e3, dt, n_steps=max(8, int(dt * 4)),
                                 verbose=False)
            key = f"{th:.2f}_{dt:.1f}"
            out[f"u_{key}"] = u
            out[f"lam_{key}"] = np.array([hist[-1][1]])
            out[f"resid_{key}"] = np.array([hist[-1][2]])
            print(f"  theta={th:.2f}  delta={dt:.1f}  lam={hist[-1][1]:.4f}"
                  f"  resid={hist[-1][2]:.2e}  "
                  f"[{(time.time()-t0)/60:.0f} min]", flush=True)
    np.savez(HERE / "q4_fields_theta.npz", **out)
    print(f"\n-> q4_fields_theta.npz  ({len(THETA)*len(DELTAS)} fields, "
          f"{(time.time()-t0)/60:.0f} min)")


if __name__ == "__main__":
    main()
