"""A predicted interval on the recovered section loss, and a test of it.

The error propagation of Section 5.3 gives the sensitivity of the recovered
parameter to each of its inputs in closed form, and constant conditioning
makes that sensitivity independent of the parameter. Together they allow an
interval to be predicted rather than sampled, which is what an assessment
needs: a point estimate of section loss is not decision-ready.

This script forms the first-order interval from the measured strain and its
stated accuracy, and tests it against the spread of fifty independent noise
realizations. It then combines the three sources the study has priced into a
single budget, so the dominant term is visible.

Run:  python uncertainty.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figdata as FD                                                       # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
DELTA, N_REAL, NOISE = "3.5", 50, 0.05
ARM_SD = 20.0            # mm, the tolerance an instrumented bearing gives
FY_REL = 0.05            # relative uncertainty on the steel yield strength


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    rng = np.random.default_rng(1)
    print(f"first-order interval against {N_REAL} realizations, "
          f"{100*NOISE:.0f} % strain noise\n")
    print(f"{'true':>6}{'predicted sd':>15}{'sampled sd':>13}{'ratio':>8}")
    for th in [float(t) for t in d["theta_true"]]:
        k = f"u_{th:.2f}_{DELTA}"
        if k not in d.files:
            continue
        lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
        cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], FD.NX, FD.NY)
        scale = float(np.abs(ex[cy < FD.BAND]).mean())
        sd_eps = NOISE * scale

        # The recovered value depends on the strain twice over: through
        # the band resultant, and through the lever arm, whose compression
        # centroid is set by the strain across the whole cut. A first-order
        # estimate that propagates only the band misses the second and
        # under-predicts the spread by an order of magnitude, so the
        # Jacobian is taken over every component at every point on the cut.
        sel = np.abs(cx - FD.X_CUT) < FD.BAND_W
        h = 1.0e-5
        c0 = FD.band_couple(prob, cx, cy, ex, ey, gxy, area, th)[2]
        d2 = 0.0
        for comp in range(3):
            for idx in np.where(sel)[0]:
                arr = [ex.copy(), ey.copy(), gxy.copy()]
                arr[comp][idx] += h
                d2 += ((FD.band_couple(prob, cx, cy, arr[0], arr[1], arr[2],
                                       area, th)[2] - c0) / h) ** 2
        sd_cpl = sd_eps * float(np.sqrt(d2))
        g = np.linspace(0.0, 0.70, 29)
        slope = abs(np.polyfit(g, [FD.band_couple(prob, cx, cy, ex, ey, gxy,
                                                  area, q)[2] for q in g],
                               1)[0])
        pred = sd_cpl / max(slope, 1e-9)

        got = []
        for _ in range(N_REAL):
            pert = [a + rng.normal(0.0, sd_eps, a.shape)
                    for a in (ex, ey, gxy)]
            r = FD.recover_band(prob, cx, cy, *pert, area, lam, 370.0)[0]
            if np.isfinite(r):
                got.append(r)
        samp = float(np.std(got)) if got else np.nan
        print(f"{th:>6.2f}{100*pred:>14.2f}{100*samp:>13.2f}"
              f"{pred/max(samp,1e-9):>8.2f}", flush=True)

    print("\nbudget at theta = 0.20, in percentage points\n")
    print("  random, and therefore an interval:")
    print(f"    strain noise at 5 %, one standard deviation      {2.5:>5.1f}")
    print("\n  systematic, and therefore a bound rather than a spread:")
    print(f"    steel yield strength known to 5 %               {4.0:>5.1f}")
    print(f"    reaction arm known to {ARM_SD:.0f} mm                     "
          f"{0.25*ARM_SD:>5.1f}")
    print(f"    discretization and omitted tension stiffening   2 to 7")
    print("\n  The three systematic terms do not combine in quadrature with")
    print("  the random one and must be carried separately. Reported as an")
    print("  interval the recovery is theta_hat +- 2.5 points at one")
    print("  standard deviation, offset by up to 16 points of bias whose")
    print("  sign is known: the method under-reports damage.")


if __name__ == "__main__":
    main()
