"""Instance A: recover a known section loss, and find where it is possible.

The identification is a moment reconciliation on a free body. Cut the beam
vertically at x_c, keep the portion towards the support, and the only
external action on it is the reaction R = lambda P / 2. Moment equilibrium
about a point on the cut therefore requires

    M_internal(theta) = R (x_c - a),

with M_internal the moment of the internal normal stresses over the cut,

    M_internal(theta) = integral of sigma_x(eps_measured; theta) (y - y_0) t dy.

theta enters only through the smeared steel in the tie band, so the
equation has one unknown and is solved by root-finding.

This replaces an earlier attempt that equated the applied moment to a
strut-and-tie couple T z. That form presumes the section is cracked deeply
enough for the tension to have localized into the band, which is false at
service: at a 2 mm deflection the beam is largely uncracked, the tension is
spread through the lower half and carried mostly by concrete, and the
required tie force came out thirteen times what the band could supply. The
free-body form above makes no assumption about cracking, so it is valid at
every load level and the load level becomes something to measure rather
than something to get wrong.

Nothing here assumes a plane section or a lever arm, which is what makes
the treatment applicable in a discontinuity region.
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
from recover_utils import element_strains, bracket_root                                 # noqa: E402

HERE = Path(__file__).resolve().parent
X_CUT_FRAC = 0.35          # between support and load, clear of both


def internal_moment(prob, cx, cy, ex, ey, gxy, area, theta, x_cut, band_w):
    """Moment of the internal normal stresses over a vertical cut, in kN m.

    Taken about the mid-height of the cut. Elements within band_w of x_cut
    form the strip; using a strip rather than a mathematical line is what
    a constant-strain discretization permits.
    """
    t = torch.tensor(float(theta))
    sel = np.abs(cx - x_cut) < band_w
    X = torch.tensor(cx[sel]).unsqueeze(-1)
    Y = torch.tensor(cy[sel]).unsqueeze(-1)
    st = membrane(torch.tensor(ex[sel]).unsqueeze(-1),
                  torch.tensor(ey[sel]).unsqueeze(-1),
                  torch.tensor(gxy[sel]).unsqueeze(-1),
                  rho_x_of_theta(prob, X, Y, t), prob.rho_y(X, Y),
                  prob.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy()
    ys = cy[sel]
    # width of the strip actually sampled, so the sum approximates an
    # integral over the cut rather than over a volume
    w = 2.0 * band_w
    dA = area / w * prob.t
    y0 = prob.H / 2.0
    return float((sx * (ys - y0) * dA).sum()) / 1e6      # kN m


def recover(prob, cx, cy, ex, ey, gxy, area, lam, x_cut, band_w):
    """Root of M_internal(theta) - M_applied, or None if none exists."""
    R = lam * prob.P / 2.0
    M_app = R * (x_cut - prob.a) / 1e6                   # kN m
    grid = np.linspace(0.0, THETA_MAX, 57)
    f = np.array([internal_moment(prob, cx, cy, ex, ey, gxy, area,
                                  g, x_cut, band_w) - M_app for g in grid])
    s = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    if not len(s):
        return None, M_app, f
    return bracket_root(f, grid), M_app, f


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    prob = DeepBeam()
    xy = d["xy"]
    nx, ny = 40, 20
    area = (prob.L / nx) * (prob.H / ny) / 2.0
    x_cut = X_CUT_FRAC * prob.L
    band_w = prob.L / nx
    print("Recovery of section loss, as a function of load level\n")
    print(f"{'delta':>7}{'theta true':>11}{'lambda':>8}"
          f"{'M app':>9}{'theta rec':>11}{'error':>8}")
    by_delta = {}
    for dt in d["deltas"]:
        errs = []
        for th in d["theta_true"]:
            key = f"{th:.2f}_{dt:.1f}"
            if f"u_{key}" not in d:
                continue
            u = d[f"u_{key}"]; lam = float(d[f"lam_{key}"][0])
            cx, cy, ex, ey, gxy = element_strains(xy, u, nx, ny)
            th_rec, M_app, _f = recover(prob, cx, cy, ex, ey, gxy, area,
                                        lam, x_cut, band_w)
            if th_rec is None:
                print(f"{dt:>7.1f}{th:>11.2f}{lam:>8.3f}{M_app:>9.1f}"
                      f"{'none':>11}{'--':>8}")
            else:
                e = th_rec - th
                errs.append(abs(e))
                print(f"{dt:>7.1f}{th:>11.2f}{lam:>8.3f}{M_app:>9.1f}"
                      f"{th_rec:>11.3f}{e:>+8.3f}")
        by_delta[float(dt)] = (np.mean(errs) if errs else None, len(errs))
    print(f"\n{'delta (mm)':>12}{'recovered':>11}{'mean |err| (pp)':>18}")
    for dt, (m, n) in by_delta.items():
        print(f"{dt:>12.1f}{n:>11d}"
              f"{('--' if m is None else f'{m*100:.1f}'):>18}")


if __name__ == "__main__":
    main()
