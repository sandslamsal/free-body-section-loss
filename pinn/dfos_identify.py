"""Identification from a fiber bonded along the reinforcement.

The sparse studies so far scattered gauges over the face and then tried to
rebuild a field from them. That is a harder problem than the one being
solved, and the reconstruction dominated the answer. Real instrumentation
is not scattered: distributed fiber-optic sensing bonds a fiber along the
reinforcement and returns a dense line of longitudinal strain readings,
which is precisely where the deterioration acts.

That layout admits a formulation needing no field reconstruction at all.
At a section x the tie carries

    T(x) = [ sigma_c(eps) + rho_tie (1 - theta) sigma_s(eps_x) ] A_band ,

evaluated from the reading at that section, while statics requires
T(x) = M(x) / z with M(x) = R (x - a) known from the applied load. The
lever arm z is not known in a discontinuity region, which is what defeated
an assumed-couple formulation earlier; but a fiber supplies readings at
many sections, so z need not be assumed. Writing q = 1/z, each reading
gives

    T_i(theta) - M_i q = 0 ,

which is affine in theta and linear in q, so a line of readings determines
both. The lever arm is then an output of the identification rather than an
assumption, and its recovered value is a check on the result: a physically
implausible z indicts the fit.

Only sections between the support and the load are used, where M(x) is
linear and no load is enclosed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))
from arclength_oracle import membrane                                      # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                         # noqa: E402
from recover_utils import element_strains                                  # noqa: E402

BAND = 150.0
THETA_MAX = 0.70


def fiber_readings(cx, cy, ex, ey, gxy, n_read, noise, seed,
                   y_fiber=75.0, x_lo=300.0, x_hi=950.0):
    """Strain sampled along a fiber at the reinforcement level."""
    rng = np.random.default_rng(seed)
    near = np.abs(cy - y_fiber) < 40.0
    inspan = (cx > x_lo) & (cx < x_hi)
    idx = np.where(near & inspan)[0]
    xs = np.linspace(x_lo + 20, x_hi - 20, n_read)
    pick = [idx[np.argmin(np.abs(cx[idx] - x))] for x in xs]
    pick = np.array(sorted(set(pick)))
    s = np.abs(ex[pick]).mean()
    return (cx[pick],
            ex[pick] + noise * s * rng.standard_normal(len(pick)),
            ey[pick] + noise * s * rng.standard_normal(len(pick)),
            gxy[pick] + noise * s * rng.standard_normal(len(pick)))


def solve_theta_z(prob, xr, ex, ey, gxy, lam):
    """Least squares for (theta, 1/z) from the fiber readings."""
    A_band = BAND * prob.thickness
    R = lam * prob.P_ref / 2.0
    M = R * (xr - 250.0)                                   # N mm

    # T(theta) = c0 + c1 * theta, evaluated per reading
    c0 = np.empty(len(xr)); c1 = np.empty(len(xr))
    for i in range(len(xr)):
        s_full, _, _ = membrane(float(ex[i]), float(ey[i]), float(gxy[i]),
                                RHO_NOM, 0.0025, prob.mat)
        s_bare, _, _ = membrane(float(ex[i]), float(ey[i]), float(gxy[i]),
                                0.0, 0.0025, prob.mat)
        sx_full = float(np.asarray(s_full).ravel()[0])
        sx_bare = float(np.asarray(s_bare).ravel()[0])
        c0[i] = sx_full * A_band                            # theta = 0
        c1[i] = (sx_bare - sx_full) * A_band                # slope in theta
    # rows: c0 + c1 theta - M q = 0  ->  [c1, -M] [theta, q]^T = -c0
    G = np.column_stack([c1, -M])
    sol, *_ = np.linalg.lstsq(G, -c0, rcond=None)
    theta, q = sol
    return float(theta), (1.0 / q if q else np.nan)


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    nx, ny = 40, 20
    print("Identification from a fiber bonded along the reinforcement\n")
    print(f"{'delta':>6}{'theta true':>11}{'reads':>7}{'noise':>7}"
          f"{'theta rec':>11}{'err pp':>9}{'z (mm)':>9}")
    for dt in (2.0, 3.5, 5.0):
        for th in (0.0, 0.20, 0.40):
            prob = deepbeam_rho(RHO_NOM * (1.0 - th))
            key = f"{th:.2f}_{dt:.1f}"
            cx, cy, ex, ey, gxy = element_strains(d["xy"], d[f"u_{key}"],
                                                  nx, ny)
            lam = float(d[f"lam_{key}"][0])
            for n_read, noise in ((60, 0.0), (60, 0.05)):
                rec, zz = [], []
                for seed in range(5):
                    xr, a, b, c = fiber_readings(cx, cy, ex, ey, gxy,
                                                 n_read, noise, seed)
                    t, z = solve_theta_z(prob, xr, a, b, c, lam)
                    rec.append(t); zz.append(z)
                m = float(np.mean(rec))
                print(f"{dt:>6.1f}{th:>11.2f}{n_read:>7d}{noise:>7.0%}"
                      f"{m:>11.3f}{(m-th)*100:>9.2f}{np.mean(zz):>9.0f}")


if __name__ == "__main__":
    main()
