"""Is the loss due to incompatibility, or to the interpolant?

Interpolating the three strain components independently gave a recovered
section loss biased low by two to eighteen percentage points, the bias
one-sided in every case. The suspected cause is that such a field is not
the symmetric gradient of any displacement, so it is kinematically
inadmissible and the tie force computed from it is systematically wrong.

That suspicion is testable without a network, and it should be tested that
way first: if simply imposing compatibility recovers the accuracy, then
compatibility is what matters and a network is one way among several of
imposing it. If a compatible fit still loses most of the accuracy, the
deficiency lies elsewhere.

The compatible reconstruction used here writes the displacement field in a
polynomial basis,

    u_x = sum a_ij x^i y^j ,      u_y = sum b_ij x^i y^j ,

and fits the coefficients by least squares to the measured strains, which
are linear in them. Any field so obtained is a genuine displacement field,
so its strains satisfy compatibility exactly and by construction. The
comparison against the independent interpolation is then like for like:
same gauges, same noise, same identifying condition, differing only in
whether the reconstructed strain is admissible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))
from arclength_oracle import build_mesh, membrane                          # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                         # noqa: E402
from recover_utils import element_strains                                  # noqa: E402
from sparse_study import (gauge_sample, tie_force, THETA_MAX, BAND)        # noqa: E402

DEG = 4          # polynomial degree per displacement component
LAM_REG = 1e-8   # ridge term, so the fit stays defined when gauges are few


def _powers(deg):
    return [(i, j) for i in range(deg + 1) for j in range(deg + 1)
            if i + j <= deg]


def compatible_fit(gx, gy, ex, ey, gxy, cx, cy, deg=DEG):
    """Least-squares displacement field whose strains match the gauges.

    Returns strains evaluated at the element centroids. Compatibility is
    exact: the returned strains are the symmetric gradient of the fitted
    displacement field, not three independently smoothed scalars.
    """
    X, Y = gx / 1000.0, gy / 1000.0          # scaled for conditioning
    P = _powers(deg)
    n, m = len(X), len(P)
    # strain rows: d(ux)/dx, d(uy)/dy, d(ux)/dy + d(uy)/dx
    A = np.zeros((3 * n, 2 * m))
    for k, (i, j) in enumerate(P):
        dx = i * X ** max(i - 1, 0) * Y ** j if i else np.zeros(n)
        dy = j * X ** i * Y ** max(j - 1, 0) if j else np.zeros(n)
        A[0 * n:1 * n, k] = dx            # eps_x from ux
        A[2 * n:3 * n, k] = dy            # gamma from ux
        A[1 * n:2 * n, m + k] = dy        # eps_y from uy
        A[2 * n:3 * n, m + k] = dx        # gamma from uy
    b = np.concatenate([ex, ey, gxy])
    coef = np.linalg.solve(A.T @ A + LAM_REG * np.eye(2 * m), A.T @ b)

    Xc, Yc = cx / 1000.0, cy / 1000.0
    nc = len(Xc)
    B = np.zeros((3 * nc, 2 * m))
    for k, (i, j) in enumerate(P):
        dx = i * Xc ** max(i - 1, 0) * Yc ** j if i else np.zeros(nc)
        dy = j * Xc ** i * Yc ** max(j - 1, 0) if j else np.zeros(nc)
        B[0 * nc:1 * nc, k] = dx
        B[2 * nc:3 * nc, k] = dy
        B[1 * nc:2 * nc, m + k] = dy
        B[2 * nc:3 * nc, m + k] = dx
    out = B @ coef
    return out[:nc], out[nc:2 * nc], out[2 * nc:]


def naive_interp(gx, gy, vals, cx, cy):
    pts = np.column_stack([gx / 1000.0, gy / 1000.0])
    tgt = np.column_stack([cx / 1000.0, cy / 1000.0])
    return RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                           smoothing=1e-8, degree=1)(tgt)


def recover(prob, cx, cy, e1, e2, e3, T_ref, area):
    lo = tie_force(prob, cx, cy, e1, e2, e3, 0.0, area)
    hi = tie_force(prob, cx, cy, e1, e2, e3, THETA_MAX, area)
    if (lo - T_ref) * (hi - T_ref) > 0:
        return np.nan
    return (T_ref - lo) / (hi - lo) * THETA_MAX


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    prob = deepbeam_rho(RHO_NOM)
    nx, ny = 40, 20
    area = (prob.L / nx) * (prob.H / ny) / 2.0
    th_true, dt = 0.30, 3.5
    cx, cy, ex, ey, gxy = element_strains(
        d["xy"], d[f"u_{th_true:.2f}_{dt:.1f}"], nx, ny)
    T_ref = tie_force(prob, cx, cy, ex, ey, gxy, th_true, area)

    print(f"true theta = {th_true:.2f};  complete-field tie force "
          f"{T_ref:.2f} kN;  10 seeds per cell\n")
    print(f"{'gauges':>7}{'noise':>7} | {'independent':>21} | "
          f"{'compatible':>21}")
    print(f"{'':>7}{'':>7} | {'theta':>9}{'err (pp)':>12} | "
          f"{'theta':>9}{'err (pp)':>12}")
    for n_g in (200, 80, 40, 20):
        for noise in (0.0, 0.02, 0.05):
            en, ec = [], []
            for seed in range(10):
                gx, gy, a, b, c = gauge_sample(cx, cy, ex, ey, gxy,
                                               n_g, noise, seed)
                ni = [naive_interp(gx, gy, v, cx, cy) for v in (a, b, c)]
                en.append(recover(prob, cx, cy, *ni, T_ref, area))
                ci = compatible_fit(gx, gy, a, b, c, cx, cy)
                ec.append(recover(prob, cx, cy, *ci, T_ref, area))
            mn, mc = np.nanmean(en), np.nanmean(ec)
            print(f"{n_g:>7d}{noise:>7.0%} | {mn:>9.3f}"
                  f"{(mn-th_true)*100:>12.2f} | {mc:>9.3f}"
                  f"{(mc-th_true)*100:>12.2f}")


if __name__ == "__main__":
    main()
