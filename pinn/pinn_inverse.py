"""Inverse continuum CSFM PINN -- reconstruct the internal stress field of a
concrete D-region from sparse, noisy strain measurements.

The forward problem (given the load, solve for the field) is what a finite
element solver does, and a pure forward PINN cannot beat it: our Deep-Energy
campaign plateaued near 40% error. The *inverse* problem is different, and it
is where a physics-informed network has a genuine advantage. A structure
already stands; it is measured at a limited number of points; the applied
load is not known; the question is the internal stress state. A finite element
solver cannot answer that -- it needs the load and the full boundary data.

A single displacement network does not suffice: its strain is the gradient of
the network, and differentiating a network amplifies fitting error. This study
uses a *mixed* formulation, after Balmer, Kaufmann and Kraus (2024):

  * a strain network  E(x)  outputs the strain field directly -- a smooth
    quantity, not a derivative, so the measurements anchor it cleanly and the
    equilibrium residual is first order and well conditioned;
  * a displacement network  U(x)  enforces kinematic compatibility, the
    reconstructed strain being tied to grad(U) by a consistency loss.

The reference field is a fine-mesh (12.5 mm) continuum CSFM analysis with
Gaussian stress recovery: the constant-strain-triangle solver carries heavy
element-scale discretization noise, so the true (smooth) continuum field is
estimated by smoothing the fine solution. Measurements are drawn from this
smooth field and the reconstruction is validated against it.

  loss = w_data*data + w_compat*compatibility + w_eq*equilibrium
       + w_free*free-edge + w_tau*load-shear + w_supp*support

Run (after scripts/p2_oracle.ts has generated oracle_deepbeam_ref.json):
  python pinn_inverse.py --gauges 48 --noise 0.03
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from csfm_constitutive import membrane
from model import DisplacementPINN
from pinn_mixed import grad
from problem import DeepBeam, Corbel, WallPier

torch.set_default_dtype(torch.float32)
torch.set_num_threads(12)              # full speed -- all CPU cores
SEED = 20260517
U_SCALE = 3.0e0                        # displacement scale (mm)
# per-component strain scales (eps_x, eps_y, gamma_xy): eps_y is ~4x smaller,
# so a common scale leaves its network output under-resolved
STRAIN_SCALE = (1.0e-3, 3.0e-4, 1.0e-3)
# applied-load scale (N). The unknown load is co-optimized as a scalar
# parameter P = P_SCALE * P_raw; the initial guess (raw = 0.5 -> 500 kN)
# is deliberately wrong, so the recovered P at the end measures the
# method's load identification.
P_SCALE = 1.0e6
REF_PATH = "oracle_deepbeam_ref.json"  # fine 160x80 mesh
SMOOTH_R = 40.0                        # stress-recovery smoothing radius (mm)


class StrainPINN(nn.Module):
    """Plain SiLU MLP on raw coordinates (x_n, y_n) -> n_out outputs. Used as
    the strain net (n_out=3) and displacement net (n_out=2) for the deep beam,
    whose smooth field needs no Fourier encoding; the corbel and wall pier use
    FourierMLP instead. Selected by arch='plain' in train()."""

    def __init__(self, width: int = 128, depth: int = 6, n_out: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(xy)


class FourierMLP(nn.Module):
    """MLP on random-Fourier-feature-encoded coordinates. The sinusoidal
    encoding lets the network represent sharper field structure -- notably
    the shear-strain sign change across the strut -- that a plain MLP, with
    its spectral bias toward smooth functions, tends to round off."""

    def __init__(self, n_out: int, width: int = 192, depth: int = 6,
                 n_freq: int = 48, scale: float = 6.0):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.register_buffer(
            "freqs", torch.randn(n_freq, 2, generator=g) * scale)
        layers: list[nn.Module] = [nn.Linear(2 + 2 * n_freq, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * np.pi * (xy @ self.freqs.t())
        feat = torch.cat([xy, torch.cos(proj), torch.sin(proj)], dim=1)
        return self.net(feat)


# --------------------------------------------------------------------------
# reference field and synthetic measurements
# --------------------------------------------------------------------------
def load_fine(path: str) -> dict:
    """Fine-mesh continuum CSFM element field: centroids and strain."""
    import json
    el = json.load(open(path))["elements"]

    def col(key: str, i: int) -> np.ndarray:
        return np.array([e[key][i] for e in el], dtype=np.float64)

    return {"cx": col("c", 0), "cy": col("c", 1),
            "strain": np.column_stack([col("eps", 0), col("eps", 1),
                                       col("eps", 2)])}


def gaussian_smooth(qx, qy, sx, sy, vals, R):
    """Gaussian stress recovery: smoothed `vals` (N,k) sampled at the query
    points (qx, qy). Windowed at 3R for speed."""
    cut = (3.0 * R) ** 2
    out = np.zeros((len(qx), vals.shape[1]))
    for i in range(len(qx)):
        d2 = (sx - qx[i]) ** 2 + (sy - qy[i]) ** 2
        m = d2 < cut
        w = np.exp(-d2[m] / (R * R))
        out[i] = (w[:, None] * vals[m]).sum(0) / w.sum()
    return out


def membrane_np(prob, x, y, ex, ey, gxy, soften):
    """Constitutive map applied to numpy strain arrays -> numpy stress."""
    t = lambda a: torch.tensor(a, dtype=torch.float32).reshape(-1, 1)
    xt, yt = t(x), t(y)
    st = membrane(t(ex), t(ey), t(gxy), prob.rho_x(xt, yt), prob.rho_y(xt, yt),
                  prob.mat, soften=soften)
    return (st["sigma_x"].numpy().ravel(), st["sigma_y"].numpy().ravel(),
            st["tau_xy"].numpy().ravel())


def build_reference(prob: DeepBeam, fine: dict, n_query: int,
                    soften: bool) -> dict:
    """Smoothed converged reference: strain + stress at validation points."""
    step = max(1, len(fine["cx"]) // n_query)
    qi = np.arange(0, len(fine["cx"]), step)
    qx, qy = fine["cx"][qi], fine["cy"][qi]
    sm = gaussian_smooth(qx, qy, fine["cx"], fine["cy"], fine["strain"],
                         SMOOTH_R)
    ex, ey, gxy = sm[:, 0], sm[:, 1], sm[:, 2]
    sx, sy, txy = membrane_np(prob, qx, qy, ex, ey, gxy, soften)
    return {"qx": qx, "qy": qy, "ex": ex, "ey": ey, "gxy": gxy,
            "sx": sx, "sy": sy, "txy": txy,
            "o2": principal2(sx, sy, txy)}


def make_measurements(prob: DeepBeam, fine: dict, n_gauge: int,
                      noise: float, seed: int) -> dict:
    """Sample a sparse grid of strain gauges from the smooth reference and add
    Gaussian measurement noise. Per-component RMS strains (rx, ry, rg) are
    returned so the data loss is normalized component by component."""
    rng = np.random.default_rng(seed)
    # Bounding-box aspect ratio sets the grid; for non-rectangular domains
    # (e.g. the L-corbel) we drop the gauges that land outside the
    # concrete and report the remaining count.
    aspect = prob.L / prob.H
    ny_g = max(2, int(round((n_gauge / aspect) ** 0.5)))
    nx_g = max(2, int(round(n_gauge / ny_g)))
    mx, my = 0.02 * prob.L, 0.02 * prob.H
    gx, gy = np.meshgrid(np.linspace(mx, prob.L - mx, nx_g),
                         np.linspace(my, prob.H - my, ny_g))
    gx, gy = gx.ravel(), gy.ravel()
    keep = prob.inside(torch.tensor(gx).reshape(-1, 1),
                       torch.tensor(gy).reshape(-1, 1)).bool().numpy().ravel()
    gx, gy = gx[keep], gy[keep]
    sm = gaussian_smooth(gx, gy, fine["cx"], fine["cy"], fine["strain"],
                         SMOOTH_R)
    ex, ey, gxy = sm[:, 0], sm[:, 1], sm[:, 2]
    rx = max(float(np.sqrt(np.mean(ex ** 2))), 1.0e-4)
    ry = max(float(np.sqrt(np.mean(ey ** 2))), 1.0e-4)
    rg = max(float(np.sqrt(np.mean(gxy ** 2))), 1.0e-4)
    scale = float(np.sqrt(np.mean(ex ** 2 + ey ** 2 + gxy ** 2)))
    sd = noise * scale
    nex = ex + sd * rng.standard_normal(ex.shape)
    ney = ey + sd * rng.standard_normal(ey.shape)
    ngxy = gxy + sd * rng.standard_normal(gxy.shape)
    # data-loss floor: the value reached when the network equals the true
    # field and only the measurement noise remains. Training past it over-fits
    # the noise, so it is the early-stopping target.
    data_floor = float((np.mean((nex - ex) ** 2) / rx ** 2
                        + np.mean((ney - ey) ** 2) / ry ** 2
                        + np.mean((ngxy - gxy) ** 2) / rg ** 2) / 3.0)

    def col(a: np.ndarray) -> torch.Tensor:
        return torch.tensor(a, dtype=torch.float32).reshape(-1, 1)

    return {
        "x": col(gx), "y": col(gy),
        "ex": col(nex), "ey": col(ney), "gxy": col(ngxy),
        "rx": rx, "ry": ry, "rg": rg, "data_floor": data_floor,
        "grid": (nx_g, ny_g), "n": len(gx), "noise": noise,
    }


def principal2(sx, sy, txy):
    """Minor (most compressive) principal stress."""
    av = 0.5 * (sx + sy)
    rad = np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    return av - rad


# --------------------------------------------------------------------------
# network fields
# --------------------------------------------------------------------------
def strain_net(e_net, prob, x, y):
    """Strain field, the direct output of the strain network (no autodiff)."""
    raw = e_net(torch.cat([x / prob.L, y / prob.H], dim=1))
    sx, sy, sg = STRAIN_SCALE
    return sx * raw[:, 0:1], sy * raw[:, 1:2], sg * raw[:, 2:3]


def disp_net(u_net, prob, x, y):
    """Displacement field, the direct output of the displacement network."""
    raw = u_net(torch.cat([x / prob.L, y / prob.H], dim=1))
    return U_SCALE * raw[:, 0:1], U_SCALE * raw[:, 1:2]


def compat_strain(u_net, prob, x, y):
    """Strain from the displacement network by differentiation (x, y leaves)."""
    raw = u_net(torch.cat([x / prob.L, y / prob.H], dim=1))
    ux = U_SCALE * raw[:, 0:1]
    uy = U_SCALE * raw[:, 1:2]
    return grad(ux, x), grad(uy, y), grad(ux, y) + grad(uy, x)


def stress_net(e_net, prob, x, y, soften):
    """CSFM stress from the strain network via the constitutive map."""
    ex, ey, gxy = strain_net(e_net, prob, x, y)
    st = membrane(ex, ey, gxy, prob.rho_x(x, y), prob.rho_y(x, y), prob.mat,
                  soften=soften)
    return st["sigma_x"], st["sigma_y"], st["tau_xy"]


# --------------------------------------------------------------------------
# loss terms
# --------------------------------------------------------------------------
def data_loss(e_net, prob, meas):
    """Per-component relative mean-square mismatch of the gauge strains."""
    ex, ey, gxy = strain_net(e_net, prob, meas["x"], meas["y"])
    return (((ex - meas["ex"]) ** 2).mean() / meas["rx"] ** 2
            + ((ey - meas["ey"]) ** 2).mean() / meas["ry"] ** 2
            + ((gxy - meas["gxy"]) ** 2).mean() / meas["rg"] ** 2) / 3.0


def compat_loss(e_net, u_net, prob, x, y):
    """Kinematic compatibility: the strain network equals grad of the
    displacement network."""
    ex, ey, gxy = strain_net(e_net, prob, x, y)
    cx, cy, cg = compat_strain(u_net, prob, x, y)
    sx, sy, sg = STRAIN_SCALE
    return (((ex - cx) / sx) ** 2).mean() / 3.0 \
        + (((ey - cy) / sy) ** 2).mean() / 3.0 \
        + (((gxy - cg) / sg) ** 2).mean() / 3.0


def smooth_loss(e_net, prob, x, y):
    """Tikhonov smoothness prior on the strain field. A network fit to a few
    dozen gauges is wildly under-determined between them; penalising the
    strain gradient damps that over-fit oscillation -- the standard
    regularization of a sparse inverse problem."""
    ex, ey, gxy = strain_net(e_net, prob, x, y)
    sx, sy, sg = STRAIN_SCALE
    out = torch.zeros(())
    for e, s in ((ex, sx), (ey, sy), (gxy, sg)):
        out = out + ((grad(e, x) * prob.L / s) ** 2
                     + (grad(e, y) * prob.H / s) ** 2).mean()
    return out / 3.0


def equilibrium_loss(e_net, prob, x, y, soften):
    """Pointwise CSFM equilibrium residual div(sigma) = 0 on the interior;
    first order in autodiff -- the stress is a direct network output."""
    sx, sy, txy = stress_net(e_net, prob, x, y, soften)
    r1 = grad(sx, x) + grad(txy, y)
    r2 = grad(txy, x) + grad(sy, y)
    fc, length = prob.mat.fc, prob.L
    return ((r1 * length / fc) ** 2 + (r2 * length / fc) ** 2).mean()


def free_edge_loss(e_net, prob, x, y, nrm, soften):
    """Traction-free residual on the unloaded boundary (known physics)."""
    sx, sy, txy = stress_net(e_net, prob, x, y, soften)
    nx, ny = nrm[:, 0:1], nrm[:, 1:2]
    tx = sx * nx + txy * ny
    ty = txy * nx + sy * ny
    return ((tx / prob.mat.fc) ** 2 + (ty / prob.mat.fc) ** 2).mean()


def load_traction_loss(e_net, prob, x, y, soften):
    """Load-patch traction condition: tau_xy = 0 (the load is purely
    vertical). The magnitude follows from the network's sigma_y on the
    patch and is recovered after training by integration. Treating P as
    a co-optimized scalar parameter through a uniform-pressure constraint
    was tested and rejected: at any meaningful weight the constraint
    either dragged the field accuracy or left the parameter decoupled
    from the data."""
    _, _, txy = stress_net(e_net, prob, x, y, soften)
    return ((txy / prob.mat.fc) ** 2).mean()


def load_constraint_loss(e_net, prob, soften, n: int = 2000):
    """Known-load calibration (R2.2 ablation): constrain the resultant of the
    reconstructed sigma_y over the load patch to the applied load P, i.e.
    'supply the load' through the loss. Normalized by P so the term is
    dimensionless. Off by default (w_load = 0); when on, tests whether
    calibrating the load improves the reconstruction."""
    gen = torch.Generator().manual_seed(SEED + 5)
    x, y = prob.loaded_patch(n, gen)
    bearing = getattr(prob, "bearing", None) or float(prob.L)
    _, sy, _ = stress_net(e_net, prob, x, y, soften)
    resultant = -sy.mean() * bearing * prob.t          # positive downward
    return ((resultant - prob.P) / prob.P) ** 2


def support_loss(u_net, prob, x, y):
    """Known support conditions, defined by the problem geometry's
    `support_residual` method (pin + roller for the deep beam, fully
    clamped for the corbel)."""
    ux, uy = disp_net(u_net, prob, x, y)
    return prob.support_residual(ux, uy, x) / U_SCALE ** 2


def outside_penalty(e_net, prob, n: int, gen: torch.Generator):
    """Pin the strain network's output toward zero in the empty part of
    the bounding box (only nonzero for non-rectangular domains).

    For an L-shape (corbel) the bounding box has a sizeable empty
    quadrant. Without this penalty the network is free to oscillate
    there: that oscillation has no data cost but propagates into the
    concrete via the network's spatial continuity, inflating
    smooth_loss near the L-shape boundary and starving the data fit.
    Anchoring the network to zero outside the material gives a clean
    boundary and lets the interior converge.
    """
    if not hasattr(prob, "inside"):
        return torch.zeros(())
    x = torch.rand(n, 1, generator=gen) * prob.L
    y = torch.rand(n, 1, generator=gen) * prob.H
    out = (1.0 - prob.inside(x, y)).squeeze(-1).bool()
    if not out.any():
        return torch.zeros(())
    raw = e_net(torch.cat([x[out].reshape(-1, 1) / prob.L,
                           y[out].reshape(-1, 1) / prob.H], dim=1))
    return (raw ** 2).mean()


def quick_error(e_net, prob, ref, soften) -> float:
    """Minor-principal-stress rel-RMS over the reference points (progress)."""
    x = torch.tensor(ref["qx"], dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(ref["qy"], dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        sx, sy, txy = stress_net(e_net, prob, x, y, soften)
    p2 = principal2(sx.numpy().ravel(), sy.numpy().ravel(), txy.numpy().ravel())
    return float(np.sqrt(np.mean((p2 - ref["o2"]) ** 2))
                 / np.sqrt(np.mean(ref["o2"] ** 2)))


# --------------------------------------------------------------------------
# training: (A) data + compatibility, (B) + equilibrium, (C) L-BFGS polish
# --------------------------------------------------------------------------
def train(prob: DeepBeam, meas: dict, ref: dict, width: int = 192,
          depth: int = 6, n_freq: int = 48, ff_scale: float = 6.0,
          adam_data: int = 3000, adam_phys: int = 5000,
          lbfgs: int = 600, lr: float = 2.0e-3, w_data: float = 10.0,
          w_compat: float = 2.0, w_smooth: float = 5.0e-4, w_eq: float = 0.0,
          w_free: float = 1.0, w_tau: float = 0.5, w_supp: float = 5.0,
          w_outside: float = 0.0, w_load: float = 0.0,
          n_int: int = 2000, n_bc: int = 600,
          soften: bool = True, arch: str = "fourier"):
    gen = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    if arch == "plain":
        e_net = StrainPINN(width=width, depth=depth, n_out=3)
        u_net = StrainPINN(width=width, depth=depth, n_out=2)
    else:
        e_net = FourierMLP(3, width=width, depth=depth, n_freq=n_freq,
                           scale=ff_scale)
        u_net = FourierMLP(2, width=width, depth=depth, n_freq=n_freq,
                           scale=ff_scale)
    params = list(e_net.parameters()) + list(u_net.parameters())

    def leaf(t):
        return t.clone().requires_grad_(True)

    def sample(n_i, n_b):
        xi, yi = (leaf(t) for t in prob.interior(n_i, gen))
        xf, yf, nf = prob.free_edges(n_b, gen)
        xl, yl = prob.loaded_patch(n_b // 2, gen)
        xs, ys = prob.supports(n_b, gen)
        return xi, yi, xf, yf, nf, xl, yl, xs, ys

    def compute(pts, w_eq_now):
        xi, yi, xf, yf, nf, xl, yl, xs, ys = pts
        l_data = data_loss(e_net, prob, meas)
        l_compat = compat_loss(e_net, u_net, prob, xi, yi)
        l_smooth = smooth_loss(e_net, prob, xi, yi)
        l_free = free_edge_loss(e_net, prob, xf, yf, nf, soften)
        l_tau = load_traction_loss(e_net, prob, xl, yl, soften)
        l_supp = support_loss(u_net, prob, xs, ys)
        l_out = (outside_penalty(e_net, prob, n_int, gen)
                 if w_outside > 0.0 else torch.zeros(()))
        l_eq = (equilibrium_loss(e_net, prob, xi, yi, soften)
                if w_eq_now > 0.0 else torch.zeros(()))
        l_load = (load_constraint_loss(e_net, prob, soften)
                  if w_load > 0.0 else torch.zeros(()))
        loss = (w_data * l_data + w_compat * l_compat + w_smooth * l_smooth
                + w_eq_now * l_eq + w_free * l_free + w_tau * l_tau
                + w_supp * l_supp + w_outside * l_out + w_load * l_load)
        return loss, (l_data, l_compat, l_smooth, l_eq, l_free, l_tau, l_supp,
                      l_out)

    # early-stopping target: with noisy gauges, fitting the data below its
    # noise floor over-fits the noise, so training halts at 1.5x the floor
    floor = meas["data_floor"]

    def run_adam(n_iter, lr0, w_eq_fn, tag):
        opt = torch.optim.Adam(params, lr=lr0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter)
        for it in range(n_iter):
            loss, terms = compute(sample(n_int, n_bc), w_eq_fn(it, n_iter))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if it % 500 == 0 or it == n_iter - 1:
                ld, lc, lsm, le, lf, lt, ls, lo = (float(t) for t in terms)
                print(f"[{tag}] it {it:5d}  loss {float(loss):.3e}  "
                      f"data {ld:.3e}  compat {lc:.3e}  smooth {lsm:.2e}  "
                      f"eq {le:.3e}  free {lf:.2e}  tau {lt:.2e}  "
                      f"supp {ls:.2e}  out {lo:.2e}", flush=True)
            if floor > 0.0 and float(terms[0]) < 1.5 * floor:
                print(f"[{tag}] early stop at it {it}  "
                      f"data {float(terms[0]):.3e} (floor {floor:.3e})",
                      flush=True)
                break
        print(f"[{tag}] sigma_2 rel-RMS "
              f"{quick_error(e_net, prob, ref, soften):.3f}", flush=True)

    # ---- Phase A: data, compatibility, boundary conditions (no equilibrium)
    run_adam(adam_data, lr, lambda it, n: 0.0, "A")
    # ---- Phase B: equilibrium ramped in from the data-fit basin ---------
    run_adam(adam_phys, 0.6 * lr,
             lambda it, n: w_eq * min(1.0, 3.0 * it / n), "B")

    # ---- Phase C: L-BFGS polish with adaptive early-stopping. -----------
    # L-BFGS empirically over-tightens the gauge fit on geometries with
    # non-smooth boundary tractions (corbel, deep beam), which then
    # degrades the principal-stress field even as the total loss falls.
    # We chunk L-BFGS into short bursts and monitor the compatibility
    # residual: when it grows persistently above its Adam-end value, the
    # network has started to trade physics consistency for gauge fit;
    # we roll back to the last good checkpoint and stop. This makes the
    # polish self-limiting and removes the need to disable it manually.
    if lbfgs > 0 and floor <= 0.0:
        pts = sample(2 * n_int, 2 * n_bc)

        def evaluate_terms():
            with torch.no_grad():
                pass  # placeholder, compute() needs grads for compat
            loss, terms = compute(pts, w_eq)
            return float(loss), float(terms[1])  # (total_loss, compat)

        def snapshot():
            return {id(p): p.detach().clone() for p in params}

        def restore(snap):
            for p in params:
                if id(p) in snap:
                    with torch.no_grad():
                        p.copy_(snap[id(p)])

        _, compat_init = evaluate_terms()
        best_state = snapshot()
        best_compat = compat_init
        chunk_size = 100
        n_chunks = max(1, lbfgs // chunk_size)
        compat_tol = 1.20   # rollback if compat exceeds 1.2 x initial
        total_iters_used = 0
        hit_rollback = False
        for ci in range(n_chunks):
            opt_l = torch.optim.LBFGS(params, max_iter=chunk_size,
                                      history_size=60,
                                      line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-12,
                                      tolerance_change=1e-14)

            def closure():
                opt_l.zero_grad()
                loss, _ = compute(pts, w_eq)
                loss.backward()
                return loss

            opt_l.step(closure)
            total_iters_used += chunk_size
            _, compat_now = evaluate_terms()
            if compat_now < best_compat:
                # strict improvement on compat: snapshot the new best
                best_state = snapshot()
                best_compat = compat_now
            if compat_now > compat_init * compat_tol:
                # physics has degraded too far: stop accumulating
                hit_rollback = True
                break
        # ALWAYS restore the best-compat state seen so far. Without this
        # restore, slow upward drift of compat (which stays under the
        # rollback threshold) can leave the network in a strictly worse
        # state than its best L-BFGS snapshot.
        restore(best_state)
        final_loss, _ = evaluate_terms()
        msg = (f"[C] L-BFGS  loss {final_loss:.3e}  sigma_2 rel-RMS "
               f"{quick_error(e_net, prob, ref, soften):.3f}  ")
        if hit_rollback:
            msg += (f"(early stop at iter {total_iters_used}; compat "
                    f"{compat_now:.2e} > {compat_tol:.2f} x init "
                    f"{compat_init:.2e}, restored best snapshot)")
        else:
            msg += (f"({total_iters_used} of {lbfgs} L-BFGS iters used; "
                    f"best compat {best_compat:.2e} vs init {compat_init:.2e})")
        print(msg, flush=True)
    return e_net, u_net


# --------------------------------------------------------------------------
# load recovery and validation
# --------------------------------------------------------------------------
def integrate_sigma_y(e_net, prob, which: str, soften: bool) -> float:
    """Resultant vertical force from the reconstructed sigma_y over a
    boundary patch. This is the naive estimate: it samples sigma_y right
    on the boundary, where the field is least resolved (no gauges sit
    there and the spectral bias smooths the discontinuity at the patch
    edges). Prefer integrate_sigma_y_cut, which uses an interior cut.

    which = 'load'    -> the top load patch  (recovers the applied load)
            'support' -> the soffit support(s)
    """
    gen = torch.Generator().manual_seed(SEED + (1 if which == "load" else 2))
    bearing = getattr(prob, "bearing", None) or getattr(prob, "bearing_V",
                                                       float(prob.L))
    if which == "load":
        x, y = prob.loaded_patch(4000, gen)
        width = bearing
    else:
        x, y = prob.supports(4000, gen)
        # deep beam has two bearings; corbel has one, wall pier's base is
        # the full width.
        width = (2.0 * bearing if hasattr(prob, "x_supp")
                 and len(prob.x_supp) > 1 else float(x.max() - x.min()))
    with torch.no_grad():
        _, sy, _ = stress_net(e_net, prob, x, y, soften)
    return float(-sy.mean()) * width * prob.t


def integrate_sigma_y_cut(e_net, prob, y_cut: float, soften: bool,
                          n: int = 4000) -> float:
    """Vertical force across an interior horizontal cut at y = y_cut.

    By global equilibrium, this equals the applied load magnitude (above
    the cut) for any cut that lies between the supports and the load
    patch. Interior cuts sit in a region densely covered by gauges, so
    the network's sigma_y is well resolved there -- unlike the boundary
    integration of integrate_sigma_y, which extrapolates to a zone the
    network's spectral bias smooths over.

    The cut spans prob.cut_xrange(y_cut), which for the deep beam is the
    full width and for the L-corbel is either the arm (above the soffit)
    or only the column (below the soffit).
    """
    x_lo, x_hi = prob.cut_xrange(y_cut)
    gen = torch.Generator().manual_seed(SEED + int(y_cut * 1000) + 7)
    s = torch.rand(n, 1, generator=gen)
    x = x_lo + (x_hi - x_lo) * s
    y = torch.full_like(x, y_cut)
    with torch.no_grad():
        _, sy, _ = stress_net(e_net, prob, x, y, soften)
    return float(-sy.mean()) * (x_hi - x_lo) * prob.t


def recover_load_cuts(e_net, prob, soften: bool) -> tuple[float, float, list]:
    """Recover the applied load by integrating sigma_y over each of
    prob.default_cuts() and averaging. Returns (mean, std, per-cut list)
    in N. The per-cut spread is an honest uncertainty estimate: if global
    equilibrium holds in the reconstruction, all cuts should give nearly
    the same value."""
    cuts = list(prob.default_cuts())
    vals = [integrate_sigma_y_cut(e_net, prob, y, soften) for y in cuts]
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std()), list(zip(cuts, vals))


def integrate_tau_xy_cut(e_net, prob, y_cut: float, soften: bool,
                         n: int = 4000) -> float:
    """Horizontal shear force across an interior horizontal cut at
    y = y_cut.

    Equilibrium of the material above the cut: integrating tau_xy along
    a horizontal slice gives the total horizontal force applied above
    that slice. For a cantilever wall pier with a horizontal load V at
    height h_eff (and no other horizontal loads), any cut at y < h_eff
    integrates to V; cuts at y > h_eff integrate to zero.
    """
    x_lo, x_hi = prob.cut_xrange(y_cut)
    gen = torch.Generator().manual_seed(SEED + int(y_cut * 1000) + 13)
    s = torch.rand(n, 1, generator=gen)
    x = x_lo + (x_hi - x_lo) * s
    y = torch.full_like(x, y_cut)
    with torch.no_grad():
        _, _, txy = stress_net(e_net, prob, x, y, soften)
    # Sign convention: V > 0 means the applied horizontal load points in
    # +x. Above-cut equilibrium then gives tau_xy averaged across the
    # cut positive in the same direction, so we return +tau_xy * width.
    return float(txy.mean()) * (x_hi - x_lo) * prob.t


def recover_shear_cuts(e_net, prob, soften: bool) -> tuple[float, float, list]:
    """Recover the applied horizontal shear by integrating tau_xy at the
    cuts returned by prob.default_cuts(). Cut-to-cut spread is the
    self-consistency check (see recover_load_cuts)."""
    cuts = list(prob.default_cuts())
    vals = [integrate_tau_xy_cut(e_net, prob, y, soften) for y in cuts]
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std()), list(zip(cuts, vals))


def validate(e_net, u_net, prob, ref, meas, soften,
             path="inverse_validation.png"):
    qx, qy = ref["qx"], ref["qy"]
    x = torch.tensor(qx, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(qy, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        pex, pey, pgxy = strain_net(e_net, prob, x, y)
        psx, psy, ptxy = stress_net(e_net, prob, x, y, soften)
    pex, pey, pgxy = (a.numpy().ravel() for a in (pex, pey, pgxy))
    psx, psy, ptxy = (a.numpy().ravel() for a in (psx, psy, ptxy))
    o2 = ref["o2"]
    p2 = principal2(psx, psy, ptxy)

    def rr(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))
                     / (np.sqrt(np.mean(b ** 2)) + 1e-12))

    nx_g, ny_g = meas["grid"]
    print("\n=== inverse PINN reconstruction vs smoothed continuum CSFM ===")
    print(f"  {meas['n']} strain gauges ({nx_g}x{ny_g}, {meas['noise']*100:.0f}% "
          f"noise) -> {len(qx)} reference points")
    print("  -- strain field --")
    for nm, a, b in (("eps_x", pex, ref["ex"]), ("eps_y", pey, ref["ey"]),
                     ("gam_xy", pgxy, ref["gxy"])):
        print(f"    {nm:8s} rel-RMS  {rr(a, b):.3f}")
    print("  -- stress field --")
    for nm, a, b in (("sigma_x", psx, ref["sx"]),
                     ("sigma_y", psy, ref["sy"]),
                     ("tau_xy", ptxy, ref["txy"]), ("sigma_2", p2, o2)):
        print(f"    {nm:8s} rel-RMS  {rr(a, b):.3f}")
    print(f"  peak compression  ref {o2.min():.2f} / PINN {p2.min():.2f} MPa")

    # Naive boundary estimate (kept for reporting honesty).
    p_load_b = integrate_sigma_y(e_net, prob, "load", soften)
    # Interior-cut estimate: averaged over several cuts inside the domain
    # where the network's sigma_y is well constrained by interior gauges.
    p_cut_mean, p_cut_std, p_cut_each = recover_load_cuts(e_net, prob, soften)
    # For wall-pier-type problems (axial + horizontal shear), interpret
    # the sigma_y integral as the AXIAL load N (which is what global
    # vertical equilibrium delivers above any cut) and also recover the
    # horizontal shear V from tau_xy.
    has_shear = hasattr(prob, "N")
    n_true = float(prob.N) if has_shear else float(prob.P)
    p_axial_label = "axial (interior)" if has_shear else "load (interior)"
    print(f"  load (boundary)   {p_load_b/1e3:.1f} kN  (true {prob.P/1e3:.0f} kN, "
          f"error {abs(p_load_b - prob.P)/prob.P*100:.1f}%)")
    print(f"  {p_axial_label} cuts (mean +/- std over {len(p_cut_each)})")
    for y, v in p_cut_each:
        print(f"    y_cut = {y:7.1f} mm  ->  {v/1e3:7.1f} kN")
    err_ax = abs(p_cut_mean - n_true) / n_true * 100
    print(f"  {p_axial_label} {p_cut_mean/1e3:.1f} +/- "
          f"{p_cut_std/1e3:.1f} kN (true {n_true/1e3:.0f} kN, "
          f"error {err_ax:.1f}%)")

    v_cut_mean = v_cut_std = err_v = 0.0
    v_cut_each: list = []
    if has_shear:
        v_cut_mean, v_cut_std, v_cut_each = recover_shear_cuts(
            e_net, prob, soften)
        print(f"  shear (interior cut, mean +/- std over {len(v_cut_each)})")
        for y, v in v_cut_each:
            print(f"    y_cut = {y:7.1f} mm  ->  {v/1e3:7.1f} kN")
        err_v = abs(v_cut_mean - prob.P) / prob.P * 100
        print(f"  shear (interior)  {v_cut_mean/1e3:.1f} +/- "
              f"{v_cut_std/1e3:.1f} kN  (true {prob.P/1e3:.0f} kN, "
              f"error {err_v:.1f}%)")
    err_pct = err_ax  # used by the figure title; for wall pier this is N

    # ---- figure: reference | reconstruction | error --------------------
    # Shared sigma_2 scale across the two compression panels (identical
    # color = identical stress); the error panel has its own scale.
    # Cut lines on the reference panel show where the load-recovery
    # integration is taken; the recovered load per cut sits inside the
    # reconstruction panel, away from the colorbar.
    gx = meas["x"].numpy().ravel()
    gy = meas["y"].numpy().ravel()
    err = np.abs(p2 - o2)
    s2_rms = rr(p2, o2)
    vmin = float(min(o2.min(), p2.min()))
    s2_levels = np.linspace(vmin, 0.0, 25)
    # Robust upper limit for the error panel: clip at 5x the 95th
    # percentile, so a handful of boundary-spike outliers do not make
    # the bulk of the field appear as a flat black background. The peak
    # value is still reported in the per-panel footer.
    err_p95 = float(np.percentile(err, 95))
    err_p99 = float(np.percentile(err, 99))
    err_max_raw = float(err.max() + 1e-9)
    err_vmax = float(max(err_p99, 4.0 * err_p95, 0.5))
    err_levels = np.linspace(0.0, err_vmax, 25)
    cuts = list(prob.default_cuts())
    cut_vals = [v for _, v in p_cut_each]
    err_p95 = float(np.percentile(err, 95))

    # ================= figure: reference | reconstruction | error =========
    # Shared sigma_2 scale across the two compression panels (identical color
    # = identical stress); the error panel has its own scale. Perceptually
    # uniform, greyscale- and CVD-safe maps (R1.11): viridis for the single-
    # signed compressive sigma_2 field (monotonic luminance, so more
    # compression still reads darker in monochrome) and magma for the
    # non-negative error field.
    CMAP_F, CMAP_E = "viridis", "magma"
    FS_SUP, FS_TTL, FS_AX, FS_TICK = 15, 15, 13, 12
    FS_LEG, FS_CB, FS_INFO = 12, 13, 13
    aspect = prob.H / prob.L
    wide = aspect <= 1.5
    density = meas["n"] / (prob.L * prob.H / 1e6)   # gauges per m^2

    def draw_panel(ax, field, ttl, cmap, lvls, lo, hi, is_err, pl,
                   gauges=False, labels=False, xlabels=True):
        tri = ax.tricontourf(qx, qy, field, levels=lvls, cmap=cmap,
                             vmin=lo, vmax=hi)
        # Cut lines on the two sigma_2 panels mark where load recovery is taken;
        # on the reference panel the per-cut recovered force is annotated.
        if not is_err:
            for k, (y_c, v) in enumerate(zip(cuts, cut_vals)):
                x_lo, x_hi = prob.cut_xrange(y_c)
                ax.plot([x_lo, x_hi], [y_c, y_c], color="#222222", ls="--",
                        lw=1.0, alpha=0.55, zorder=3)
                if labels:
                    if has_shear:
                        v_v = v_cut_each[k][1]
                        lt = (f"$\\hat N$={v/1e3:.0f}\n"
                              f"$\\hat V$={v_v/1e3:.0f} kN")
                    else:
                        lt = f"{v/1e3:.0f} kN"
                    ax.text(x_hi - 0.015 * prob.L, y_c, lt, color="#111111",
                            fontsize=FS_LEG, va="center", ha="right", zorder=6,
                            weight="bold",
                            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                                      ec="0.45", lw=0.6, alpha=0.92))
        # Open gauge markers on the reference panel only (field shows through);
        # the marker key lives in the shared bottom legend (cf. Fig. 8).
        if gauges:
            ax.scatter(gx, gy, s=5, marker="o", facecolors="none",
                       edgecolors="0.15", linewidths=0.25, alpha=0.6, zorder=4)
        # Support / load markers matched to each archetype's physics.
        if has_shear:
            for xc in np.linspace(0, prob.L, 9):
                ax.plot([xc], [0], marker="^", ms=9, mfc="#222222",
                        mec="white", mew=0.8, zorder=5, clip_on=False)
            ax.plot([prob.L / 2], [prob.H], marker="v", ms=12, mfc="#c0392b",
                    mec="white", mew=1.2, zorder=5, clip_on=False)
            ax.plot([0], [prob.h_eff], marker=">", ms=12, mfc="#c0392b",
                    mec="white", mew=1.2, zorder=5, clip_on=False)
        else:
            for xc in prob.x_supp:
                ax.plot([xc], [0], marker="^", ms=12, mfc="#222222",
                        mec="white", mew=1.2, zorder=5)
            ax.plot([prob.x_load], [prob.H], marker="v", ms=12, mfc="#c0392b",
                    mec="white", mew=1.2, zorder=5, clip_on=False)
        ax.set_aspect("equal")
        ax.set_title(f"{pl}  {ttl}", fontsize=FS_TTL, weight="bold", pad=3,
                     loc="left")
        if xlabels:
            ax.set_xlabel("x (mm)", fontsize=FS_AX)
        ax.tick_params(labelsize=FS_TICK, labelbottom=xlabels)
        return tri

    # Headline + metrics strings (shared by both layouts).
    if has_shear:
        headline = (f"Inverse-PINN reconstruction "
                    f"({prob.L:.0f}$\\,\\times\\,${prob.H:.0f} mm wall, "
                    f"$N\\,=\\,${n_true/1e3:.0f} kN, "
                    f"$V\\,=\\,${prob.P/1e3:.0f} kN)")
        metrics = (f"$\\sigma_2$ rel-RMS: {s2_rms*100:.1f}%"
                   f"$\\;\\;\\bullet\\;\\;$"
                   f"$\\hat N$ = {p_cut_mean/1e3:.0f}$\\pm$"
                   f"{p_cut_std/1e3:.0f} kN (err {err_ax:.1f}%)"
                   f"$\\;\\;\\bullet\\;\\;$"
                   f"$\\hat V$ = {v_cut_mean/1e3:.0f}$\\pm$"
                   f"{v_cut_std/1e3:.0f} kN (err {err_v:.1f}%)")
    else:
        headline = (f"Inverse-PINN reconstruction of the internal $\\sigma_2$ "
                    f"field ({prob.L:.0f}$\\,\\times\\,${prob.H:.0f} mm domain, "
                    f"$P\\,=\\,${prob.P/1e3:.0f} kN)")
        metrics = (f"$\\sigma_2$ rel-RMS error: {s2_rms*100:.1f}%"
                   f"$\\;\\;\\;\\bullet\\;\\;\\;$recovered load (interior cut): "
                   f"{p_cut_mean/1e3:.0f} $\\pm$ {p_cut_std/1e3:.0f} kN"
                   f" (true {prob.P/1e3:.0f} kN, err {err_pct:.1f}%)")

    # Acquisition + error stats for the wide-domain summary box (2x2 layout).
    info_rows = [
        ("gauge density", f"{density:.0f} per m$^2$"),
        ("gauges, noise", f"{meas['n']}, {meas['noise']*100:.0f}%"),
        ("smoothing radius $R$", f"{SMOOTH_R:.0f} mm"),
        (r"peak $|\Delta\sigma_2|$", f"{err_max_raw:.1f} MPa"),
        (r"95th-pctl $|\Delta\sigma_2|$", f"{err_p95:.2f} MPa"),
        ("color scale clip", f"{err_vmax:.1f} MPa"),
    ]

    # Shared marker legend (applied load / supports / gauges), cf. Fig. 8.
    from matplotlib.lines import Line2D
    sup_lbl = "clamped base" if has_shear else "supports"
    leg_handles = [
        Line2D([0], [0], marker="v", ls="none", mfc="#c0392b", mec="white",
               mew=1.0, ms=12, label="applied load"),
        Line2D([0], [0], marker="^", ls="none", mfc="#222222", mec="white",
               mew=1.0, ms=12, label=sup_lbl),
        Line2D([0], [0], marker="o", ls="none", mfc="none", mec="0.2",
               ms=8, label=f"{meas['n']} strain gauges"),
    ]

    if wide:
        # ---- 2x2: (a) ref | (b) recon ; (c) error | summary box ; legend ----
        panel_w = 5.0
        panel_h = panel_w * aspect
        top_pad, row_gap = 0.72, 0.40
        # bottom margin, inches from panels down: x-labels, gap, legend, floor
        xlabel_h, leg_gap, leg_h, small_bot = 0.46, 0.05, 0.38, 0.04
        bot_pad = xlabel_h + leg_gap + leg_h + small_bot
        hspace = row_gap / panel_h           # so tall panels do not over-gap
        fig_w = 2 * panel_w + 2.2
        fig_h = 2 * panel_h + row_gap + top_pad + bot_pad
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(2, 2, hspace=hspace, wspace=0.13,
                              left=0.065, right=0.86,
                              top=1 - top_pad / fig_h,
                              bottom=bot_pad / fig_h)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[0, 1])
        axC = fig.add_subplot(gs[1, 0])
        axI = fig.add_subplot(gs[1, 1])
        axI.axis("off")
        triA = draw_panel(axA, o2, "CSFM reference", CMAP_F, s2_levels, vmin,
                          0.0, False, "(a)", gauges=True, labels=True,
                          xlabels=False)
        draw_panel(axB, p2, "PINN reconstruction", CMAP_F, s2_levels, vmin,
                   0.0, False, "(b)", xlabels=False)
        triC = draw_panel(axC, err, "absolute error", CMAP_E, err_levels, 0.0,
                          err_vmax, True, "(c)")
        axA.set_ylabel("y (mm)", fontsize=FS_AX)
        axC.set_ylabel("y (mm)", fontsize=FS_AX)
        cb1 = fig.colorbar(triA, ax=[axA, axB], shrink=0.92, pad=0.02, aspect=26)
        cb1.set_label(r"$\sigma_2$ (MPa, compression negative)", fontsize=FS_CB)
        cb1.ax.tick_params(labelsize=FS_TICK)
        cb2 = fig.colorbar(triC, ax=[axC, axI], shrink=0.92, pad=0.02, aspect=26)
        cb2.set_label(r"$|\sigma_2^{\mathrm{PINN}}-\sigma_2^{\mathrm{ref}}|$ (MPa)",
                      fontsize=FS_CB)
        cb2.ax.tick_params(labelsize=FS_TICK)
        axI.add_patch(plt.Rectangle((0.03, 0.04), 0.94, 0.90,
                      transform=axI.transAxes, fill=True, fc="0.965", ec="0.65",
                      lw=1.0, clip_on=False, zorder=0))
        axI.text(0.09, 0.86, "acquisition & error", transform=axI.transAxes,
                 fontsize=FS_TTL, weight="bold", va="top", ha="left")
        y0, dy = 0.68, 0.108
        for i, (k, v) in enumerate(info_rows):
            yy = y0 - i * dy
            axI.text(0.09, yy, k, transform=axI.transAxes, fontsize=FS_INFO,
                     va="center", ha="left")
            axI.text(0.91, yy, v, transform=axI.transAxes, fontsize=FS_INFO,
                     va="center", ha="right", weight="bold")
        leg_cy = (small_bot + leg_h / 2) / fig_h
        lg = fig.legend(handles=leg_handles, loc="center", ncol=3,
                        fontsize=FS_INFO, frameon=True, facecolor="white",
                        framealpha=0.9, bbox_to_anchor=(0.46, leg_cy),
                        borderpad=0.5, columnspacing=1.8, handletextpad=0.5)
        st = fig.suptitle(headline + "\n" + metrics, fontsize=FS_SUP,
                          weight="bold", y=1 - 0.05 / fig_h)
    else:
        # ---- 3-in-a-row (tall VK1): horizontal box + legend below ----------
        fig_w = 13.0
        top_pad = 0.70
        # bottom margin, inches: x-labels, gap, box, gap, legend, floor
        xaxis_h, box_gap, box_h = 0.44, 0.06, 0.82
        mid_gap, leg_h, small_bot = 0.16, 0.40, 0.04
        bot_pad = xaxis_h + box_gap + box_h + mid_gap + leg_h + small_bot
        usable_w = fig_w * (0.975 - 0.045)
        panel_w = (usable_w - 2 * 0.20 * usable_w / 3) / 3
        panel_h = panel_w * aspect
        fig_h = panel_h + top_pad + bot_pad
        fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h),
                                 gridspec_kw={"top": 1 - top_pad / fig_h,
                                              "bottom": bot_pad / fig_h,
                                              "wspace": 0.20, "left": 0.045,
                                              "right": 0.975})
        triA = draw_panel(axes[0], o2, "CSFM reference", CMAP_F, s2_levels,
                          vmin, 0.0, False, "(a)", gauges=True, labels=True)
        draw_panel(axes[1], p2, "PINN reconstruction", CMAP_F, s2_levels,
                   vmin, 0.0, False, "(b)")
        triC = draw_panel(axes[2], err, "absolute error", CMAP_E, err_levels,
                          0.0, err_vmax, True, "(c)")
        for a in axes:
            a.set_anchor("S")
        axes[0].set_ylabel("y (mm)", fontsize=FS_AX)
        cb1 = fig.colorbar(triA, ax=axes[:2].tolist(), shrink=0.82, pad=0.02)
        cb1.set_label(r"$\sigma_2$ (MPa, compression negative)", fontsize=FS_CB)
        cb1.ax.tick_params(labelsize=FS_TICK)
        cb2 = fig.colorbar(triC, ax=axes[2], shrink=0.82, pad=0.02)
        cb2.set_label(r"$|\sigma_2^{\mathrm{PINN}}-\sigma_2^{\mathrm{ref}}|$ (MPa)",
                      fontsize=FS_CB)
        cb2.ax.tick_params(labelsize=FS_TICK)
        # Horizontal "acquisition & error" box: same content and style as the
        # 2x2 summary box (Fig 2/5) -- title at the left, then three columns of
        # key -> bold-value pairs (acquisition on top, error below).
        box_cy = (small_bot + leg_h + mid_gap + box_h / 2) / fig_h
        box_hf = box_h / fig_h
        bx0, bxw = 0.085, 0.83
        axbox = fig.add_axes([bx0, box_cy - box_hf / 2, bxw, box_hf])
        axbox.axis("off")
        axbox.add_patch(plt.Rectangle((0, 0), 1, 1, transform=axbox.transAxes,
                        fill=True, fc="0.965", ec="0.65", lw=1.0, clip_on=False,
                        zorder=0))
        axbox.text(0.018, 0.5, "acquisition\n& error", transform=axbox.transAxes,
                   fontsize=FS_TTL, weight="bold", va="center", ha="left")
        axbox.plot([0.152, 0.152], [0.16, 0.84], transform=axbox.transAxes,
                   color="0.6", lw=1.0)
        for i in range(3):
            k1, v1 = info_rows[i]        # acquisition (top row)
            k2, v2 = info_rows[i + 3]    # error (bottom row)
            kx = 0.185 + i * 0.278
            vx = kx + 0.248
            axbox.text(kx, 0.72, k1, transform=axbox.transAxes, fontsize=FS_INFO,
                       ha="left", va="center")
            axbox.text(vx, 0.72, v1, transform=axbox.transAxes, fontsize=FS_INFO,
                       ha="right", va="center", weight="bold")
            axbox.text(kx, 0.28, k2, transform=axbox.transAxes, fontsize=FS_INFO,
                       ha="left", va="center")
            axbox.text(vx, 0.28, v2, transform=axbox.transAxes, fontsize=FS_INFO,
                       ha="right", va="center", weight="bold")
        leg_cy = (small_bot + leg_h / 2) / fig_h
        lg = fig.legend(handles=leg_handles, loc="center", ncol=3,
                        fontsize=FS_INFO, frameon=True, facecolor="white",
                        framealpha=0.9, bbox_to_anchor=(0.5, leg_cy),
                        borderpad=0.5, columnspacing=1.8, handletextpad=0.5)
        st = fig.suptitle(headline + "\n" + metrics, fontsize=FS_SUP,
                          y=1 - 0.04 / fig_h, weight="bold")

    # Center the suptitle and marker legend over the drawn content (panels +
    # colorbars). In the 2x2 layout that content is not symmetric about the
    # figure mid-line (colorbars sit on the right), so a plain x=0.5 title
    # reads as shifted; measure the content box and recentre on it.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    _bbs = [ax.get_tightbbox(fig.canvas.get_renderer()) for ax in fig.axes]
    _bbs = [b for b in _bbs if b is not None]
    if _bbs:
        _u = Bbox.union(_bbs)
        _inv = fig.transFigure.inverted()
        _xc = 0.5 * (_inv.transform((_u.x0, 0))[0]
                     + _inv.transform((_u.x1, 0))[0])
        st.set_x(_xc)
        lg.set_bbox_to_anchor((_xc, leg_cy))
    fig.savefig(path, dpi=450, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauges", type=int, default=48)
    ap.add_argument("--noise", type=float, default=0.03)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--n-freq", type=int, default=48)
    ap.add_argument("--ff-scale", type=float, default=6.0)
    ap.add_argument("--adam-data", type=int, default=3000)
    ap.add_argument("--adam-phys", type=int, default=5000)
    ap.add_argument("--lbfgs", type=int, default=600)
    ap.add_argument("--n-int", type=int, default=2000)
    ap.add_argument("--n-query", type=int, default=3000)
    ap.add_argument("--w-data", type=float, default=10.0)
    ap.add_argument("--w-compat", type=float, default=2.0)
    ap.add_argument("--w-smooth", type=float, default=5.0e-4)
    ap.add_argument("--w-eq", type=float, default=0.0)
    ap.add_argument("--w-outside", type=float, default=0.0)
    ap.add_argument("--w-load", type=float, default=0.0)
    ap.add_argument("--meas-oracle", default="",
                    help="draw the strain measurements from a DIFFERENT (coarser) "
                         "oracle mesh than the validation reference, to test for "
                         "inverse crime (R1.5); empty = same fine mesh as the reference")
    ap.add_argument("--arch", default="auto",
                    choices=["auto", "plain", "fourier"],
                    help="net architecture; 'auto' = plain for the deep beam, "
                         "Fourier for the corbel and wall pier (as published)")
    ap.add_argument("--soften", type=int, default=1)
    ap.add_argument("--problem", default="deepbeam",
                    choices=["deepbeam", "corbel", "vk1"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    soften = bool(a.soften)
    arch = a.arch if a.arch != "auto" else (
        "plain" if a.problem == "deepbeam" else "fourier")
    tag = f"_{a.tag}" if a.tag else ""

    if a.problem == "corbel":
        prob = Corbel()
        ref_path = "oracle_corbel_ref.json"
    elif a.problem == "vk1":
        prob = WallPier()
        ref_path = "oracle_vk1_ref.json"
    else:
        prob = DeepBeam()
        ref_path = REF_PATH
    fine = load_fine(ref_path)
    ref = build_reference(prob, fine, a.n_query, soften)
    # R1.5 inverse-crime test: optionally draw the measurements from a coarser,
    # independent mesh, while still validating against the fine reference above.
    fine_meas = load_fine(a.meas_oracle) if a.meas_oracle else fine
    if a.meas_oracle:
        print(f"  [inverse-crime test] measurements drawn from {a.meas_oracle} "
              f"({len(fine_meas['cx'])} elem), validated vs {ref_path} "
              f"({len(fine['cx'])} elem)", flush=True)
    meas = make_measurements(prob, fine_meas, a.gauges, a.noise, SEED)
    print(f"INVERSE PINN (mixed)  deep beam {prob.L:.0f}x{prob.H:.0f} mm, "
          f"P = {prob.P/1e3:.0f} kN  [recovered, not given to the network]")
    print(f"  {meas['n']} strain gauges  grid {meas['grid'][0]}x{meas['grid'][1]}"
          f"  noise {a.noise*100:.0f}%  width {a.width}  "
          f"adam {a.adam_data}+{a.adam_phys}  lbfgs {a.lbfgs}  "
          f"n_int {a.n_int}  ref {len(ref['qx'])} pts (R={SMOOTH_R:.0f}mm)  "
          f"w_eq {a.w_eq}  soften {soften}", flush=True)
    e_net, u_net = train(prob, meas, ref, width=a.width,
                         n_freq=a.n_freq, ff_scale=a.ff_scale,
                         adam_data=a.adam_data,
                         adam_phys=a.adam_phys, lbfgs=a.lbfgs,
                         n_int=a.n_int, w_data=a.w_data,
                         w_compat=a.w_compat, w_smooth=a.w_smooth,
                         w_eq=a.w_eq, w_outside=a.w_outside,
                         w_load=a.w_load,
                         soften=soften, arch=arch)
    os.makedirs("runs", exist_ok=True)
    torch.save({"e": e_net.state_dict(), "u": u_net.state_dict()},
               f"runs/{a.problem}_inverse{tag}.pt")
    validate(e_net, u_net, prob, ref, meas, soften,
             path=f"inverse_validation{tag}.png")


if __name__ == "__main__":
    main()
