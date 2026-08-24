"""Go/no-go: is the deterioration parameter identifiable from strain at all?

Before any training, one question decides whether this paper is possible.
theta enters the formulation only through the smeared reinforcement ratio
in the constitutive map, so it can be recovered from measurements only if
the equilibrium residual actually responds to it. If dR/dtheta is
negligible, the parameter is invisible to the physics and no amount of
optimization or data recovers it.

The test uses a strain field representative of a cracked deep beam rather
than a trained network, deliberately: the question is a property of the
formulation, not of any particular fit, and asking it of an analytic field
removes training quality as a confounder. The field carries tension along
the soffit tie band, where the steel works and where theta therefore acts,
and an inclined compression band between the load point and the supports.

Reported: R(theta) across the admissible range, its exact derivative by
automatic differentiation, and the relative sensitivity that decides
identifiability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import DeepBeam                                              # noqa: E402
from csfm_constitutive import membrane                                    # noqa: E402
from identify import rho_x_of_theta, THETA_MAX                            # noqa: E402


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def representative_strain(prob, x, y):
    """A cracked deep-beam strain field, smooth and analytic.

    Tension along the soffit band, decaying with height, so the tie steel is
    engaged; a diagonal compressive band from the loaded patch toward each
    support; shear coupling between them. Magnitudes are those of a beam
    near its service load, with the tie strain past yield onset so the
    reinforcement term is active.
    """
    xn = x / prob.L
    yn = y / prob.H
    # tie: tension near the soffit, strongest at midspan, decaying upward
    ex = 2.5e-3 * torch.exp(-6.0 * yn) * torch.sin(np.pi * xn)
    # strut: compression on the diagonals from midspan to the supports
    d = torch.exp(-8.0 * (torch.abs(xn - 0.5) - 0.85 * yn) ** 2)
    ey = -1.1e-3 * d - 2.0e-4 * yn
    gxy = 9.0e-4 * torch.sign(0.5 - xn) * d * (1.0 - yn)
    return ex, ey, gxy


def residual_at(prob, theta_t, n=6000, seed=0):
    """Equilibrium residual of the representative field at a given theta."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(n, 1, generator=g) * prob.L).requires_grad_(True)
    y = (torch.rand(n, 1, generator=g) * prob.H).requires_grad_(True)
    ex, ey, gxy = representative_strain(prob, x, y)
    st = membrane(ex, ey, gxy, rho_x_of_theta(prob, x, y, theta_t),
                  prob.rho_y(x, y), prob.mat, soften=True)
    r1 = grad(st["sigma_x"], x) + grad(st["tau_xy"], y)
    r2 = grad(st["tau_xy"], x) + grad(st["sigma_y"], y)
    return ((r1 * prob.L / prob.mat.fc) ** 2
            + (r2 * prob.L / prob.mat.fc) ** 2).mean()


def main() -> None:
    prob = DeepBeam()
    print("Is the deterioration parameter visible to the physics?\n")
    print(f"{'theta':>7}{'rho_tie':>10}{'residual R':>13}"
          f"{'dR/dtheta':>13}{'rel. sens.':>12}")
    rows = []
    for tv in np.linspace(0.0, THETA_MAX, 8):
        th = torch.tensor(float(tv), requires_grad=True)
        R = residual_at(prob, th)
        (g,) = torch.autograd.grad(R, th)
        R0, g0 = float(R.detach()), float(g)
        rel = abs(g0) * THETA_MAX / max(R0, 1e-30)
        rows.append((tv, R0, g0, rel))
        print(f"{tv:>7.2f}{prob.rho_tie*(1-tv):>10.5f}{R0:>13.5f}"
              f"{g0:>13.5f}{rel:>12.2f}")

    R = np.array([r[1] for r in rows])
    rel = np.mean([r[3] for r in rows])
    span = (R.max() - R.min()) / R.mean() * 100
    print(f"\nresidual varies by {span:.1f} % across the admissible range")
    print(f"mean relative sensitivity |dR/dtheta| * theta_max / R = {rel:.2f}")
    if rel > 0.2 and span > 5:
        print("\nVERDICT: IDENTIFIABLE. The residual responds to theta, so a "
              "measured\n         strain field constrains it. Proceed.")
    else:
        print("\nVERDICT: WEAK. The residual barely moves with theta; the "
              "parameter is\n         close to invisible and recovery would "
              "be ill-conditioned.")


if __name__ == "__main__":
    main()
