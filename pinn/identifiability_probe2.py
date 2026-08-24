"""Second identifiability test: is theta visible in the RIGHT quantity?

The first probe asked whether the pointwise equilibrium residual
div(sigma) responds to theta, and found it barely does: 1.8 % across the
whole admissible range. That result is real but it indicts the choice of
observable rather than the parameter. The residual at a point is
dominated by the concrete stresses, which do not depend on theta at all;
the steel enters as a small additive term in one band, so its variation
is swamped.

The quantity that actually carries the information is the FORCE the tie
band carries. Given a measured strain field, the tension resultant over
the band is

    T(theta) = integral over band of sigma_x(eps_measured; theta) dA,

and because the steel contribution is proportional to rho_tie, T depends
on theta close to linearly rather than marginally. Physically: if a
corroded tie is strained to a measured value, it carries less force than
a sound tie strained the same amount, and it is that force deficit which
must be reconciled against the applied load.

This mirrors what the P3 study found for equilibrium: a global,
integrated condition discriminates where a pointwise residual does not.
The test reports both observables on the identical strain field so the
comparison is like for like.
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
from identifiability_probe import representative_strain                   # noqa: E402


def tie_force(prob, theta_t, n=4000):
    """Tension resultant carried by the soffit tie band, in kN."""
    g = torch.Generator().manual_seed(3)
    x = torch.rand(n, 1, generator=g) * prob.L
    y = torch.rand(n, 1, generator=g) * prob.band          # band only
    x.requires_grad_(True); y.requires_grad_(True)
    ex, ey, gxy = representative_strain(prob, x, y)
    st = membrane(ex, ey, gxy, rho_x_of_theta(prob, x, y, theta_t),
                  prob.rho_y(x, y), prob.mat, soften=True)
    # mean sigma_x over the band times the band area
    area = prob.band * prob.t
    return st["sigma_x"].mean() * area / 1e3


print("Is theta visible in the tie FORCE rather than the pointwise residual?\n")
prob = DeepBeam()
print(f"{'theta':>7}{'rho_tie':>10}{'T (kN)':>11}{'dT/dtheta':>12}{'rel.sens':>10}")
Ts, sens = [], []
for th in np.linspace(0.0, THETA_MAX, 8):
    t = torch.tensor(float(th), requires_grad=True)
    T = tie_force(prob, t)
    (g,) = torch.autograd.grad(T, t)
    Ts.append(float(T)); sens.append(abs(float(g)) * THETA_MAX / abs(float(T)))
    print(f"{th:>7.2f}{prob.rho_tie*(1-th):>10.5f}{float(T):>11.1f}"
          f"{float(g):>12.1f}{sens[-1]:>10.2f}")
rng = (max(Ts) - min(Ts)) / abs(np.mean(Ts)) * 100
print(f"\ntie force varies by {rng:.0f} % across the admissible range")
print(f"mean relative sensitivity = {np.mean(sens):.2f}")
print(f"  (pointwise residual, for comparison: 1.8 % and 0.02)")
print("\nVERDICT:", "STRONG - theta is identifiable from the force the tie carries"
      if np.mean(sens) > 0.3 else "still weak")
