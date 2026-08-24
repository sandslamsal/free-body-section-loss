"""Mixed continuum CSFM PINN -- decisive pure-PINN attempt.

Two networks (cf. Balmer, Kaufmann & Kraus 2024): a displacement network
u(x, y) and a stress network sigma(x, y). Their advantage here is the order
of differentiation:

  * Equilibrium  div(sigma) = 0  is FIRST-order autodiff on the stress net --
    no double-backward through the stiff, non-smooth CSFM constitutive map,
    which is what wrecked the conditioning of the single-net runs.
  * Compatibility: strain = sym(grad u) from the displacement net.
  * A constitutive-consistency loss ties them:
        sigma_net  ==  C(strain(u_net)).
  At convergence the field is simultaneously equilibrated, compatible and
  constitutive -- which is exactly the Compatible Stress Field Method.

Run:  python pinn_mixed.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from csfm_constitutive import membrane
from model import DisplacementPINN
from problem import DeepBeam

torch.set_default_dtype(torch.float32)
torch.set_num_threads(12)

U0 = 0.8          # displacement scale (mm)
SEED = 20260517


class StressPINN(nn.Module):
    """(x_n, y_n) -> (sigma_x, sigma_y, tau_xy), nondimensional (units of f_c)."""

    def __init__(self, width: int = 96, depth: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, 3)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, xy: Tensor) -> Tensor:
        return self.net(xy)


def grad(out: Tensor, inp: Tensor) -> Tensor:
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]


def grad_norm(loss: Tensor, params: list[Tensor]) -> float:
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = sum((gi ** 2).sum() for gi in g if gi is not None)
    return float(torch.sqrt(sq + 1e-30))


def strain(u_net: DisplacementPINN, prob: DeepBeam, x: Tensor, y: Tensor):
    xn, yn = x / prob.L, y / prob.H
    raw = u_net(torch.cat([xn, yn], dim=1))
    ux, uy = U0 * raw[:, 0:1], U0 * raw[:, 1:2]
    ex = grad(ux, x)
    ey = grad(uy, y)
    gxy = grad(ux, y) + grad(uy, x)
    return ux, uy, ex, ey, gxy


def stress(s_net: StressPINN, prob: DeepBeam, x: Tensor, y: Tensor):
    """Stress-net stress (MPa); output is nondimensionalised by f_c."""
    xn, yn = x / prob.L, y / prob.H
    s = prob.mat.fc * s_net(torch.cat([xn, yn], dim=1))
    return s[:, 0:1], s[:, 1:2], s[:, 2:3]


def train(prob: DeepBeam, iters: int = 8000, n_int: int = 1500,
          n_bc: int = 350, lr: float = 1.5e-3):
    gen = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    u_net = DisplacementPINN(width=96, depth=6)
    s_net = StressPINN(width=96, depth=6)
    params = list(u_net.parameters()) + list(s_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    fc = prob.mat.fc
    # adaptive weights; equilibrium term is the reference (weight 1)
    w = {"const": 1.0, "supp": 5.0, "load": 5.0, "free": 2.0}

    def leaf(t: Tensor) -> Tensor:
        return t.clone().requires_grad_(True)

    for it in range(iters):
        ramp = min(1.0, 0.15 + 0.85 * it / (0.4 * iters))
        p = ramp * prob.pressure

        xi, yi = prob.interior(n_int, gen)
        xs, ys = prob.supports(n_bc, gen)
        xl, yl = prob.loaded_patch(n_bc, gen)
        xf, yf, nf = prob.free_edges(n_bc, gen)
        xi, yi = leaf(xi), leaf(yi)
        xl, yl = leaf(xl), leaf(yl)
        xf, yf = leaf(xf), leaf(yf)

        # --- equilibrium: first-order divergence of the stress net ---------
        sx, sy, txy = stress(s_net, prob, xi, yi)
        r1 = grad(sx, xi) + grad(txy, yi)
        r2 = grad(txy, xi) + grad(sy, yi)
        l_eq = ((r1 * prob.L / fc) ** 2 + (r2 * prob.L / fc) ** 2).mean()

        # --- constitutive consistency: sigma_net == C(strain(u_net)) -------
        _, _, ex, ey, gxy = strain(u_net, prob, xi, yi)
        st = membrane(ex, ey, gxy, prob.rho_x(xi, yi), prob.rho_y(xi, yi),
                      prob.mat)
        l_const = (((sx - st["sigma_x"]) / fc) ** 2
                   + ((sy - st["sigma_y"]) / fc) ** 2
                   + ((txy - st["tau_xy"]) / fc) ** 2).mean()

        # --- supports: u = 0 (u_x only at the left support) ----------------
        us_x, us_y, *_ = strain(u_net, prob, leaf(xs), leaf(ys))
        l_supp = (us_y / U0).pow(2).mean() \
            + (us_x[xs < prob.L / 2] / U0).pow(2).mean()

        # --- loaded patch: stress-net traction (0, -p) ---------------------
        slx, sly, tlxy = stress(s_net, prob, xl, yl)
        l_load = (((sly + p) / fc) ** 2 + (tlxy / fc) ** 2).mean()

        # --- free edges: stress-net traction = 0 ---------------------------
        sfx, sfy, tfxy = stress(s_net, prob, xf, yf)
        nx, ny = nf[:, 0:1], nf[:, 1:2]
        tfx = sfx * nx + tfxy * ny
        tfy = tfxy * nx + sfy * ny
        l_free = ((tfx / fc) ** 2 + (tfy / fc) ** 2).mean()

        if it % 150 == 0:
            ge = grad_norm(l_eq, params)
            for key, lk in (("const", l_const), ("supp", l_supp),
                            ("load", l_load), ("free", l_free)):
                tgt = ge / (grad_norm(lk, params) + 1e-12)
                w[key] = 0.9 * w[key] + 0.1 * min(max(tgt, 0.1), 1.0e3)

        loss = (l_eq + w["const"] * l_const + w["supp"] * l_supp
                + w["load"] * l_load + w["free"] * l_free)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if it % 250 == 0 or it == iters - 1:
            rms = float((r1 ** 2 + r2 ** 2).mean().detach().sqrt())
            print(f"it {it:5d}  ramp {ramp:.2f}  loss {float(loss):.3e}  "
                  f"eq {float(l_eq):.2e}  const {float(l_const):.2e}  "
                  f"supp {float(l_supp):.1e}  load {float(l_load):.1e}  "
                  f"free {float(l_free):.1e}  |div s| {rms:.3e}", flush=True)
    return u_net, s_net


def plot_field(s_net: StressPINN, prob: DeepBeam, path: str) -> None:
    nx, ny = 160, 80
    xs = torch.linspace(0, prob.L, nx)
    ys = torch.linspace(0, prob.H, ny)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    x = gx.reshape(-1, 1)
    y = gy.reshape(-1, 1)
    with torch.no_grad():
        sx, sy, txy = stress(s_net, prob, x, y)
    sx = sx.reshape(ny, nx); sy = sy.reshape(ny, nx); txy = txy.reshape(ny, nx)
    av = 0.5 * (sx + sy)
    rad = torch.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    s2 = (av - rad).numpy()

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    im = ax.contourf(gx.numpy(), gy.numpy(), s2, levels=24, cmap="RdBu")
    fig.colorbar(im, ax=ax, label=r"minor principal stress $\sigma_2$ (MPa)")
    for xc in prob.x_supp:
        ax.plot([xc], [0], "k^", ms=9)
    ax.plot([prob.x_load], [prob.H], "rv", ms=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"Mixed continuum PINN compression field "
                 f"-- deep beam, P = {prob.P/1e3:.0f} kN")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}  (min sigma_2 = {s2.min():.2f} MPa)")


def main() -> None:
    prob = DeepBeam()
    print(f"MIXED PINN  deep beam {prob.L:.0f}x{prob.H:.0f} mm, "
          f"P = {prob.P/1e3:.0f} kN, fc = {prob.mat.fc:.0f} MPa")
    u_net, s_net = train(prob, iters=16000, n_int=2000, n_bc=450)
    os.makedirs("runs", exist_ok=True)
    torch.save({"u": u_net.state_dict(), "s": s_net.state_dict()},
               "runs/deepbeam_mixed.pt")
    plot_field(s_net, prob, "deepbeam_mixed_field.png")


if __name__ == "__main__":
    main()
