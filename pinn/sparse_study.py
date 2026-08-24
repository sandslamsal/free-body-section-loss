"""Does the identification survive sparse, noisy measurement?

The recovery of Section 5 used the complete nodal displacement field, from
which internal forces can be assembled exactly. An inspection supplies
something far weaker: strain at a few dozen scattered points, each carrying
measurement error. This script measures what that costs.

The comparison is deliberately between two ways of getting from gauge
readings to a field, because they differ in a way that matters. Scattered
interpolation of the three strain components treats them as independent
scalar fields, and the result is in general NOT the symmetric gradient of
any displacement field: interpolated strains need not satisfy compatibility.
Assembling internal forces from an incompatible strain field is not a
well-posed operation, and any error it introduces is absorbed into the
recovered parameter.

Stage one, run here, establishes how far plain interpolation carries. It
sweeps the number of gauges and the noise level, and reports the recovered
section loss against the value that generated the field. Whether a network
that enforces compatibility recovers what interpolation loses is the
question the second stage answers; the point of running this first is that
if interpolation were adequate, the network would have nothing to do.
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

THETA_MAX = 0.70
X_CUT = 700.0
BAND = 150.0


def gauge_sample(cx, cy, ex, ey, gxy, n_gauge, noise, seed,
                 band_only=False):
    """Draw gauges from the member face.

    Gauges are placed over the whole face by default rather than only on
    the tie band. Restricting them to the band was tried first, on the
    reasoning that the informative strain lives there, and it makes any
    compatible reconstruction ill-posed: a displacement field cannot be
    determined from data spanning a tenth of the depth, so the fit is
    unconstrained in the through-depth direction. It is also the less
    realistic arrangement, since surface instrumentation covers the
    accessible face.
    """
    rng = np.random.default_rng(seed)
    pool = np.where(cy < BAND)[0] if band_only else np.arange(len(cx))
    pick = rng.choice(pool, size=min(n_gauge, len(pool)), replace=False)
    scale = np.abs(ex).mean()
    out = []
    for a in (ex, ey, gxy):
        out.append(a[pick] + noise * scale * rng.standard_normal(len(pick)))
    return cx[pick], cy[pick], out[0], out[1], out[2]


def interpolate(gx, gy, vals, cx, cy):
    """Scattered interpolation of one strain component onto the elements."""
    pts = np.column_stack([gx / 1000.0, gy / 1000.0])
    tgt = np.column_stack([cx / 1000.0, cy / 1000.0])
    f = RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                        smoothing=1e-8, degree=1)
    return f(tgt)


def tie_force(prob, cx, cy, ex, ey, gxy, theta, area):
    """Tension resultant carried by the band, in kN.

    The band-averaged horizontal stress multiplied by the cross-sectional
    area of the band. Summing the element contributions weighted by plan
    area would integrate over the span as well and return a force times a
    length, not a force.
    """
    sel = np.where(cy < BAND)[0]
    rho = RHO_NOM * (1.0 - theta)
    acc = 0.0
    for i in sel:
        sig, _, _ = membrane(float(ex[i]), float(ey[i]), float(gxy[i]),
                             rho, 0.0025, prob.mat)
        acc += float(np.asarray(sig).ravel()[0])
    return acc / len(sel) * (BAND * prob.thickness) / 1e3


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    prob = deepbeam_rho(RHO_NOM)
    mesh = build_mesh(prob)
    nx, ny = 40, 20
    area = (prob.L / nx) * (prob.H / ny) / 2.0
    th_true, dt = 0.30, 3.5
    u = d[f"u_{th_true:.2f}_{dt:.1f}"]
    cx, cy, ex, ey, gxy = element_strains(d["xy"], u, nx, ny)

    # reference: the tie force the complete field gives at the true state
    T_ref = tie_force(prob, cx, cy, ex, ey, gxy, th_true, area)
    print(f"true theta = {th_true:.2f} at delta = {dt} mm")
    print(f"tie force from the complete field: {T_ref:.2f} kN\n")
    print(f"{'gauges':>8}{'noise':>8}{'T (kN)':>10}{'theta rec':>11}"
          f"{'error (pp)':>12}")
    for n_gauge in (200, 80, 40, 20):
        for noise in (0.0, 0.02, 0.05):
            errs = []
            for seed in range(5):
                gx, gy, a, b, c = gauge_sample(cx, cy, ex, ey, gxy,
                                               n_gauge, noise, seed)
                ei = interpolate(gx, gy, a, cx, cy)
                ej = interpolate(gx, gy, b, cx, cy)
                ek = interpolate(gx, gy, c, cx, cy)
                # recover theta by matching the reference tie force
                lo = tie_force(prob, cx, cy, ei, ej, ek, 0.0, area)
                hi = tie_force(prob, cx, cy, ei, ej, ek, THETA_MAX, area)
                if (lo - T_ref) * (hi - T_ref) > 0:
                    errs.append(np.nan)
                    continue
                th = (T_ref - lo) / (hi - lo) * THETA_MAX
                errs.append(th - th_true)
            m = np.nanmean(errs) if np.any(np.isfinite(errs)) else np.nan
            Tm = tie_force(prob, cx, cy, ei, ej, ek, th_true, area)
            print(f"{n_gauge:>8d}{noise:>8.0%}{Tm:>10.2f}"
                  f"{th_true + m:>11.3f}{m*100:>12.2f}")


if __name__ == "__main__":
    main()
