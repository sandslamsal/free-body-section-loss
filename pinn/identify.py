"""Identify a deterioration parameter from sparse measured strain.

P2 answers: given measured strain on a structure that is as-built, what is
the internal stress field and the applied load? P4 asks the question an
asset owner actually has: the structure is old and its condition is
unknown, so given the same measurements, HOW DETERIORATED IS IT?

Formally, the reinforcement enters the cracked-membrane map through the
smeared ratio rho_x. Corrosion of the main tension tie reduces it,

    rho_tie(theta) = rho_nom * (1 - theta),     theta in [0, 1),

and theta is not known. The inverse problem of P2 is therefore extended by
one scalar unknown, and the identification is posed as

    minimize over (E, U, theta):
        w_data  * || E(x_g) - eps_measured ||^2          measurement anchor
      + w_eq    * || div sigma(E; theta) ||^2            equilibrium
      + w_compat* || E - grad U ||^2                     compatibility
      + boundary terms

where theta enters ONLY through the constitutive map. Nothing else changes:
the measurement anchor still makes the problem well posed, and the physics
still supplies what the sparse gauges cannot.

The reason this is a natural problem for a network rather than for a solver
is that theta is recovered by the SAME automatic differentiation that
trains the fields. A finite-element treatment of the same question needs an
adjoint solve per parameter to obtain d(residual)/d(theta); here the
gradient already exists because theta is a leaf of the computational graph.
That is the one advantage claim of this program that is unambiguous.

Parametrization. theta is carried as an unconstrained latent z with
theta = theta_max * sigmoid(z), so the optimizer is unconstrained while the
physical parameter stays in range and the reinforcement ratio can never go
negative.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csfm_constitutive import membrane                                     # noqa: E402
from problem import DeepBeam                                              # noqa: E402

THETA_MAX = 0.70          # widest section loss the parametrization admits


class DeteriorationParameter(torch.nn.Module):
    """One scalar unknown, held so that gradients reach it.

    theta = THETA_MAX * sigmoid(z). The map keeps theta in (0, THETA_MAX)
    for any real z, so the optimizer never has to respect a bound and the
    smeared ratio cannot go negative. The initial value is deliberately the
    midpoint rather than zero: starting at an undamaged guess places the
    parameter where the gradient of the residual is smallest, which is the
    slowest place to start from.
    """

    def __init__(self, theta_init: float = 0.35,
                 theta_max: float = THETA_MAX):
        super().__init__()
        self.theta_max = theta_max
        t0 = min(max(theta_init / theta_max, 1e-3), 1 - 1e-3)
        z0 = torch.log(torch.tensor(t0 / (1.0 - t0)))
        self.z = torch.nn.Parameter(z0)

    def forward(self) -> Tensor:
        return self.theta_max * torch.sigmoid(self.z)

    @property
    def value(self) -> float:
        with torch.no_grad():
            return float(self.forward())


def rho_x_of_theta(prob: DeepBeam, x: Tensor, y: Tensor,
                   theta: Tensor) -> Tensor:
    """Smeared horizontal ratio with the tie band scaled by (1 - theta).

    Mirrors DeepBeam.rho_x but keeps theta in the graph, so d/dtheta of any
    downstream quantity is available by automatic differentiation. Only the
    band is deteriorated; the mesh-reinforcement floor is a property of the
    detailing and is left alone.
    """
    in_band = (y < prob.band).to(x.dtype)
    rho_tie = prob.rho_tie * (1.0 - theta)
    return prob.rho_min + (rho_tie - prob.rho_min) * in_band


def stress_with_theta(e_net, prob, x, y, theta, soften: bool = True):
    """CSFM stress from the strain network at a given deterioration state."""
    from pinn_inverse import strain_net                                   # noqa: E402
    ex, ey, gxy = strain_net(e_net, prob, x, y)
    st = membrane(ex, ey, gxy, rho_x_of_theta(prob, x, y, theta),
                  prob.rho_y(x, y), prob.mat, soften=soften)
    return st["sigma_x"], st["sigma_y"], st["tau_xy"]


def equilibrium_loss_theta(e_net, prob, x, y, theta, soften: bool = True):
    """Equilibrium residual, differentiable in theta.

    This is the only term through which theta acts: the measurements
    constrain the strain field, and theta is whatever value makes that
    measured field equilibrate. If the residual were insensitive to theta
    the parameter would be unidentifiable, which is exactly what the
    sensitivity check below tests for.
    """
    from pinn_inverse import grad                                         # noqa: E402
    sx, sy, txy = stress_with_theta(e_net, prob, x, y, theta, soften)
    r1 = grad(sx, x) + grad(txy, y)
    r2 = grad(txy, x) + grad(sy, y)
    return ((r1 * prob.L / prob.mat.fc) ** 2
            + (r2 * prob.L / prob.mat.fc) ** 2).mean()


@dataclass
class Identifiability:
    """How strongly the residual responds to the parameter."""
    theta: float
    residual: float
    dR_dtheta: float

    def __str__(self) -> str:
        return (f"theta={self.theta:.3f}  R={self.residual:.5f}  "
                f"dR/dtheta={self.dR_dtheta:+.5f}")


def identifiability(e_net, prob, x, y, theta_val: float,
                    soften: bool = True) -> Identifiability:
    """Residual and its exact derivative with respect to theta.

    Obtained by automatic differentiation in one backward pass, with no
    finite differencing and no adjoint solve. A parameter is identifiable
    only where this derivative is appreciable; reporting it is therefore
    the honest precondition for claiming recovery, and it is cheap enough
    to report at every theta rather than at one.
    """
    th = torch.tensor(float(theta_val), requires_grad=True)
    R = equilibrium_loss_theta(e_net, prob, x, y, th, soften)
    (g,) = torch.autograd.grad(R, th, retain_graph=False)
    return Identifiability(float(theta_val), float(R.detach()), float(g))


# --------------------------------------------------------------------------
# the identifying condition
# --------------------------------------------------------------------------
def tie_resultant(e_net, prob, theta: Tensor, n: int = 3000,
                  gen: torch.Generator | None = None,
                  soften: bool = True) -> Tensor:
    """Tension resultant carried by the soffit tie band.

    Integrates sigma_x over the band by Monte-Carlo quadrature. This is the
    observable through which theta is identifiable: probing the formulation
    before training showed the pointwise equilibrium residual varies by
    1.8 % across the admissible range of theta while this resultant varies
    by 146 %, because the steel contribution is proportional to rho_tie
    whereas the pointwise residual is dominated by concrete stresses that
    carry no information about the parameter (see notes/identifiability.md).
    """
    from pinn_inverse import strain_net                                   # noqa: E402
    g = gen if gen is not None else torch.Generator().manual_seed(0)
    x = (torch.rand(n, 1, generator=g) * prob.L).requires_grad_(True)
    y = (torch.rand(n, 1, generator=g) * prob.band).requires_grad_(True)
    ex, ey, gxy = strain_net(e_net, prob, x, y)
    st = membrane(ex, ey, gxy, rho_x_of_theta(prob, x, y, theta),
                  prob.rho_y(x, y), prob.mat, soften=soften)
    return st["sigma_x"].mean() * (prob.band * prob.t)


def force_reconciliation_loss(e_net, prob, theta: Tensor, P_applied: float,
                              lever_arm: float | None = None,
                              **kw) -> Tensor:
    """Squared relative mismatch between the force the tie carries and the
    force statics says it must carry.

    For the simply supported deep beam under a central load the midspan
    moment is P*a/2 with a the support inset, and the internal couple that
    resists it is T*z. The lever arm is NOT assumed: it defaults to the
    distance from the centroid of the tie band to the centroid of the
    compression resultant implied by the strain field itself, which is what
    makes the treatment valid in a region where plane sections do not hold.
    """
    T = tie_resultant(e_net, prob, theta, **kw)
    z = lever_arm if lever_arm is not None else compression_lever(e_net, prob)
    M_applied = P_applied * prob.a / 2.0
    T_required = M_applied / z
    return ((T - T_required) / T_required.abs().clamp(min=1e-6)) ** 2


def compression_lever(e_net, prob, n: int = 3000,
                      gen: torch.Generator | None = None) -> Tensor:
    """Distance from the tie-band centroid to the centroid of compression.

    Computed from the strain field rather than assumed, so no plane-section
    hypothesis enters. The compression centroid is the sigma_x-weighted mean
    height over the region where sigma_x is compressive.
    """
    from pinn_inverse import strain_net, stress_net                       # noqa: E402
    g = gen if gen is not None else torch.Generator().manual_seed(1)
    x = (torch.rand(n, 1, generator=g) * prob.L).requires_grad_(True)
    y = (torch.rand(n, 1, generator=g) * prob.H).requires_grad_(True)
    sx, _sy, _t = stress_net(e_net, prob, x, y, soften=True)
    w = torch.relu(-sx)                       # compression only, positive
    y_c = (w * y).sum() / w.sum().clamp(min=1e-9)
    return (y_c - prob.band / 2.0).clamp(min=1e-3)
