"""Is the displaced minimizer a mesh artefact?

The manuscript claims that refining the mesh does not rescue the pointwise
objective, but until now it supported that claim by citing a companion
manuscript reviewers cannot obtain. This study produces the number in this
repository: the closed-form displacement of the minimizer,

    theta_hat - theta_star  =  - <r0, g> / ||g||^2 ,

evaluated on a field solved at 60x30, that is 2.25 times the elements of
the 40x20 reference, at theta = 0.20 and delta = 3.5. Every definition is
identical to displacement_check.py apart from the grid: the same cell
averaging of the two triangles, the same interior divergence of the
interpolated stress, the same central difference in the parameter. If the
displacement were a discretization error it would shrink materially under
refinement; if it belongs to the objective it will not.

The 60x30 solve costs about half an hour, so the field is written to
oracle/field_60x30_020.npz on first run and loaded ever after.

Run:  python refine_displacement.py
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

from arclength_oracle import build_mesh                                    # noqa: E402
from arclength_oracle_crisfield import newton_displacement_control         # noqa: E402
from csfm_constitutive import membrane                                     # noqa: E402
from identify import rho_x_of_theta                                        # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                         # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains                                  # noqa: E402

FIELDS_40 = HERE.parent / "oracle" / "fields_theta.npz"
FIELD_60 = HERE.parent / "oracle" / "field_60x30_020.npz"
THETA, DELTA, H = 0.20, 3.5, 0.02


def cell_fields(prob, xy, u, theta, nx, ny):
    """figdata.cell_fields with the grid as an argument, nothing else changed."""
    cx, cy, ex, ey, gxy = element_strains(xy, u, nx, ny)
    X = torch.tensor(cx).unsqueeze(-1)
    Y = torch.tensor(cy).unsqueeze(-1)
    st = membrane(torch.tensor(ex).unsqueeze(-1), torch.tensor(ey).unsqueeze(-1),
                  torch.tensor(gxy).unsqueeze(-1),
                  rho_x_of_theta(prob, X, Y, torch.tensor(float(theta))),
                  prob.rho_y(X, Y), prob.mat, soften=True)
    out = {}
    for k, v in (("sx", st["sigma_x"]), ("sy", st["sigma_y"]),
                 ("txy", st["tau_xy"])):
        out[k] = v.squeeze().numpy()
    out.update(cx=cx, cy=cy, ex=ex, ey=ey, gxy=gxy)
    for k in list(out):
        a = out[k]
        out[k] = 0.5 * (a[0::2] + a[1::2])
    n = out["cx"].size
    assert n == nx * ny, n
    for k in list(out):
        out[k] = out[k].reshape(ny, nx)
    return out


def residual(prob, xy, u, theta, nx, ny):
    """displacement_check.residual with the grid as an argument."""
    f = cell_fields(prob, xy, u, theta, nx, ny)
    dx, dy = prob.L / nx, prob.H / ny
    r1 = (np.gradient(f["sx"], dx, axis=1)
          + np.gradient(f["txy"], dy, axis=0))
    r2 = (np.gradient(f["txy"], dx, axis=1)
          + np.gradient(f["sy"], dy, axis=0))
    return np.stack([r1[1:-1, 1:-1], r2[1:-1, 1:-1]])


def displacement(prob, xy, u, nx, ny):
    """Predicted and directly measured displacement, plus the norms behind it."""
    r0 = residual(prob, xy, u, THETA, nx, ny)
    g = (residual(prob, xy, u, THETA + H, nx, ny)
         - residual(prob, xy, u, THETA - H, nx, ny)) / (2.0 * H)
    pred = -float((r0 * g).sum() / (g * g).sum())
    trial = np.linspace(0.0, 0.70, 71)
    J = [float((residual(prob, xy, u, q, nx, ny) ** 2).sum()) for q in trial]
    meas = float(trial[int(np.argmin(J))]) - THETA
    return (pred, meas,
            float(np.sqrt((r0 ** 2).sum())), float(np.sqrt((g ** 2).sum())))


def field_60x30():
    """The theta = 0.20 state at 60x30, solved once and cached."""
    if FIELD_60.exists():
        z = np.load(FIELD_60)
        print(f"loaded cached 60x30 field ({float(z['seconds']):.0f} s solve)",
              flush=True)
        return z["xy"], z["u"], float(z["lam"])
    p = dataclasses.replace(deepbeam_rho(RHO_NOM * (1.0 - THETA)),
                            nx=60, ny=30)
    mesh = build_mesh(p)
    print("solving 60x30 at theta = 0.20 ...", flush=True)
    t0 = time.time()
    hist = newton_displacement_control(p, mesh, delta_max=DELTA,
                                       n_steps=max(6, int(DELTA * 8)),
                                       verbose=False)
    secs = time.time() - t0
    last = hist[-1]
    u = np.asarray(last.u).reshape(-1, 2)
    np.savez_compressed(FIELD_60, xy=mesh.xy, u=u,
                        lam=float(last.lam), seconds=secs)
    print(f"solved in {secs:.0f} s, lam = {float(last.lam):.4f}, "
          f"wrote {FIELD_60}", flush=True)
    return mesh.xy, u, float(last.lam)


def main() -> None:
    prob = DeepBeam()

    d = np.load(FIELDS_40)
    p40, m40, r40, g40 = displacement(prob, d["xy"], d[f"u_{THETA:.2f}_{DELTA}"],
                                      40, 20)
    print(f"40x20:  predicted {100*p40:6.1f} pp   measured {100*m40:6.1f} pp"
          f"   ||r0|| {r40:.4e}   ||g|| {g40:.4e}", flush=True)

    xy60, u60, lam60 = field_60x30()
    p60, m60, r60, g60 = displacement(prob, xy60, u60, 60, 30)
    print(f"60x30:  predicted {100*p60:6.1f} pp   measured {100*m60:6.1f} pp"
          f"   ||r0|| {r60:.4e}   ||g|| {g60:.4e}", flush=True)

    print("\nWhatever the refined number says, it is the number: if the")
    print("displacement shrinks materially the text must say refinement")
    print("helps; if it stays or grows, the objective is implicated.")


if __name__ == "__main__":
    main()
