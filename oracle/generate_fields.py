"""Strain fields at known deterioration states, across load level.

Instance A needs what no experiment supplies: fields on a structure whose
deterioration is known exactly. The displacement-controlled solver gives
them.

The sweep is two-dimensional by design. The first attempt generated fields
only at a 2 mm service deflection, chosen because an owner instruments a
structure that is standing rather than one being loaded to failure. That
choice turned out to matter more than expected: at 2 mm the section is
largely uncracked, the steel carries a small share of the tension, and the
deterioration is nearly invisible. Whether identification is possible is
therefore not a property of the method alone but of the load level at
which the measurement is taken, so load level is swept as an independent
variable rather than fixed.

The solver is the consistent-tangent Newton scheme under displacement
control, NOT the secant Picard iteration that the surrounding study uses
for its load-deflection curves. The distinction is not cosmetic here.
Identification reconciles measured strain against statics on a free body,
so it presumes the field it is given actually satisfies equilibrium; the
secant scheme converges to a clipped fixed point that does not, and
checking sectional equilibrium on its fields showed the internal moment
falling 29 % short of the applied moment. A field carrying that error
cannot support an identification whose whole content is a force balance.

Output: fields_theta.npz, indexed by (theta, delta).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arclength_oracle import build_mesh                                   # noqa: E402
from arclength_oracle_crisfield import newton_displacement_control        # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                        # noqa: E402

THETA_TRUE = [0.0, 0.10, 0.20, 0.30, 0.40]
DELTAS = [1.0, 2.0, 3.5, 5.0, 7.0]      # service through to near the peak
N_STEPS_PER_MM = 8


def main() -> None:
    here = Path(__file__).resolve().parent
    out, meta = {}, []
    for th in THETA_TRUE:
        prob = deepbeam_rho(RHO_NOM * (1.0 - th))
        mesh = build_mesh(prob)
        for dt in DELTAS:
            n = max(6, int(dt * N_STEPS_PER_MM))
            hist = newton_displacement_control(prob, mesh, delta_max=dt,
                                               n_steps=n, verbose=False)
            last = hist[-1]
            key = f"{th:.2f}_{dt:.1f}"
            out[f"u_{key}"] = np.asarray(last.u).reshape(-1, 2)
            out[f"lam_{key}"] = np.array([last.lam])
            out[f"delta_{key}"] = np.array([last.delta])
            out[f"resid_{key}"] = np.array([last.resid])
            meta.append((th, dt, last.lam))
            print(f"  theta={th:.2f}  delta={last.delta:5.2f} mm  "
                  f"lam={last.lam:.3f}  resid={last.resid:.2e}  "
                  f"conv={last.converged}", flush=True)
    out["xy"] = mesh.xy
    out["theta_true"] = np.array(THETA_TRUE)
    out["deltas"] = np.array(DELTAS)
    out["rho_nom"] = np.array([RHO_NOM])
    np.savez(here / "fields_theta.npz", **out)
    print(f"\n-> fields_theta.npz  ({len(meta)} fields)")


if __name__ == "__main__":
    main()
