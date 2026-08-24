"""Reconstruct the displacement field from gauge strains with a network.

Every formulation that routed through recovered element stresses has
failed, and the reason is now clear: on a constant-strain discretization
the sectional moment of the recovered stresses is short by about a quarter,
because the element is exact for uniform strain and poor for the linear
through-depth gradient that carries the moment. Force balances survive
that; moment balance does not. The formulation that works evaluates the
assembled internal force vector, which is the quantity the solver actually
converged.

That fixes what the network must produce. It must return a DISPLACEMENT
field, so the assembled forces remain available, rather than a strain field
from which stresses would have to be recovered. Two consequences follow.
Compatibility holds by construction, since the strains are obtained by
differentiating the field. And the sparse problem inherits the identifying
condition that recovers the parameter exactly from a complete field, so any
error is attributable to the reconstruction alone.

Rigid-body modes are not determined by strain data, so the support
conditions are imposed as an additional term; without them the fit is
defined only up to a translation and rotation, which the assembly would
then see as a spurious state.

Data alone is not enough, and the reason is instructive. With Fourier
features the field reproduces a few hundred gauge readings to eight
significant figures and still yields no admissible parameter, because a
few hundred readings leave the great majority of the domain unconstrained
and the interpolant between them is arbitrary. What closes the gap is the
pointwise equilibrium residual, imposed here as a reconstruction prior.

That gives the residual a role, having been shown useless in another. By
Proposition 3 it is nearly blind to the parameter and cannot serve as the
identifying objective; but blindness to the parameter is no impediment to
constraining the field between gauges, which is a different task. The two
functionals therefore divide the labour: the residual reconstructs, and
the integrated condition identifies. Neither substitutes for the other.

The baseline for comparison is the finite-element least-squares fit, which
is also displacement-based and also compatible, so the comparison isolates
what the network representation adds rather than what compatibility adds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))
from arclength_oracle import build_mesh                                    # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                         # noqa: E402
from recover_nodal import band_imbalance, THETA_MAX                        # noqa: E402

torch.set_default_dtype(torch.float64)
L, H = 2000.0, 1000.0
U_SCALE = 15.0         # mm; the benchmark field reaches 9.2 mm, so a
                       # smaller scale leaves the fit fighting its own
                       # parametrization rather than the data


class DispNet(nn.Module):
    """(x, y) -> (u_x, u_y) through Fourier features.

    A plain coordinate MLP is slow to resolve the sharp transition between
    the compression fan and the cracked tie, which is where the informative
    strain lives. Random Fourier features supply that bandwidth directly
    and cost nothing at this size.
    """

    def __init__(self, width=96, depth=4, n_freq=32, sigma=2.5, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.B = nn.Parameter(torch.randn(2, n_freq, generator=g) * sigma,
                              requires_grad=False)
        layers = [nn.Linear(2 * n_freq, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        z = torch.cat([x / L, y / H], dim=-1)
        p = 2 * torch.pi * z @ self.B
        return self.net(torch.cat([torch.sin(p), torch.cos(p)], -1)) * U_SCALE


def strains_of(net, x, y):
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u = net(x, y)
    ux, uy = u[:, 0:1], u[:, 1:2]
    g = lambda a, b: torch.autograd.grad(                                # noqa: E731
        a, b, torch.ones_like(a), create_graph=True)[0]
    return g(ux, x), g(uy, y), g(ux, y) + g(uy, x)


def physics_residual(net, prob, n_col, gen):
    """Mean squared divergence of the CSFM stress implied by the field.

    The constitutive here must be the one the structure actually obeys. An
    earlier version used a linear elastic surrogate, on the reasoning that
    the term only has to penalize fields that cannot be equilibrated; that
    is wrong, and measurably so. The true field is cracked, so elastic
    equilibrium is inconsistent with it, and penalising the elastic
    divergence pushed the reconstruction away from the data: the gauge
    misfit stalled some three orders of magnitude above what the same
    network reaches with no prior at all. A prior must be drawn from the
    same physics as the measurements it is regularising.

    The nominal reinforcement is used, since Proposition 3 established that
    this residual is nearly blind to the deterioration parameter; that
    blindness, fatal for identification, is harmless here.
    """
    from csfm_constitutive import membrane as memb_t, CsfmMaterial
    mat = CsfmMaterial(fc=30.0)
    x = (torch.rand(n_col, 1, generator=gen) * L).requires_grad_(True)
    y = (torch.rand(n_col, 1, generator=gen) * H).requires_grad_(True)
    ex, ey, gxy = strains_of(net, x, y)
    rho_x = torch.where(y < 150.0, torch.full_like(y, RHO_NOM),
                        torch.full_like(y, 0.0010))
    rho_y = torch.full_like(y, 0.0025)
    st = memb_t(ex, ey, gxy, rho_x, rho_y, mat, soften=True)
    g = lambda a, b: torch.autograd.grad(                                # noqa: E731
        a, b, torch.ones_like(a), create_graph=True)[0]
    rx = g(st["sigma_x"], x) + g(st["tau_xy"], y)
    ry = g(st["tau_xy"], x) + g(st["sigma_y"], y)
    return ((rx * L / 30.0) ** 2 + (ry * L / 30.0) ** 2).mean()


def train(gx, gy, ge, prob, mesh, iters=8000, w_supp=1e3, w_phys=3e-3,
          seed=0):
    """Fit the field to gauge strains, pinned at the supports and
    regularized by the equilibrium residual between them."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = DispNet()
    X = torch.tensor(gx).unsqueeze(-1); Y = torch.tensor(gy).unsqueeze(-1)
    E = torch.tensor(ge)
    fixed = np.asarray(mesh.fixed, dtype=bool)
    sn = sorted({n for n in range(mesh.n_node) if fixed[2 * n] or fixed[2 * n + 1]})
    SX = torch.tensor(mesh.xy[sn, 0]).unsqueeze(-1)
    SY = torch.tensor(mesh.xy[sn, 1]).unsqueeze(-1)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    sc = float(np.abs(ge[:, 0]).mean())
    for it in range(iters):
        opt.zero_grad()
        ex, ey, gxy = strains_of(net, X, Y)
        data = (((ex - E[:, 0:1]) / sc) ** 2 + ((ey - E[:, 1:2]) / sc) ** 2
                + ((gxy - E[:, 2:3]) / sc) ** 2).mean()
        us = net(SX, SY)
        supp = (us ** 2).mean() / U_SCALE ** 2
        phys = physics_residual(net, prob, 256, gen)
        (data + w_supp * supp + w_phys * phys).backward()
        opt.step()
        sched.step()
        if it % 100000 == 0:
            print(f"    it {it:5d}  data {float(data):.3e}  "
                  f"phys {float(phys):.3e}", flush=True)
    return net


def nodal_field(net, mesh):
    X = torch.tensor(mesh.xy[:, 0]).unsqueeze(-1)
    Y = torch.tensor(mesh.xy[:, 1]).unsqueeze(-1)
    with torch.no_grad():
        u = net(X, Y).numpy()
    out = np.zeros(mesh.ndof)
    out[0::2] = u[:, 0]; out[1::2] = u[:, 1]
    return out


def recover(u, prob, mesh, lam):
    g0 = band_imbalance(u, prob, mesh, 0.0, lam)
    g7 = band_imbalance(u, prob, mesh, THETA_MAX, lam)
    if g0 * g7 > 0:
        return np.nan
    return -g0 / (g7 - g0) * THETA_MAX


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    th_true, dt = 0.30, 3.5
    prob = deepbeam_rho(RHO_NOM * (1.0 - th_true))
    mesh = build_mesh(prob)
    u_true = d[f"u_{th_true:.2f}_{dt:.1f}"].ravel()
    lam = float(d[f"lam_{th_true:.2f}_{dt:.1f}"][0])
    print(f"exact from the complete field: "
          f"{recover(u_true, prob, mesh, lam):.4f}   (true {th_true})\n")

    def dofs(e):
        n = mesh.tris[e][0]
        return [2*n[0], 2*n[0]+1, 2*n[1], 2*n[1]+1, 2*n[2], 2*n[2]+1]
    eps = np.array([mesh.B[e] @ u_true[dofs(e)] for e in range(len(mesh.tris))])
    cen = np.array([mesh.xy[list(mesh.tris[e][0])].mean(axis=0)
                    for e in range(len(mesh.tris))])
    sc = np.abs(eps[:, 0]).mean()

    for n_g, noise in ((300, 0.0), (300, 0.02), (100, 0.0), (100, 0.02)):
        rng = np.random.default_rng(0)
        el = rng.choice(len(mesh.tris), n_g, replace=False)
        ge = eps[el] + noise * sc * rng.standard_normal((n_g, 3))
        print(f"  gauges {n_g}, noise {noise:.0%}")
        net = train(cen[el, 0], cen[el, 1], ge, prob, mesh)
        th = recover(nodal_field(net, mesh), prob, mesh, lam)
        print(f"    -> theta {th:.4f}   error {(th-th_true)*100:+.2f} pp\n")


if __name__ == "__main__":
    main()
