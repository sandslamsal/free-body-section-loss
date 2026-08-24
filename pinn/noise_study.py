"""What measurement error really costs, under noise a fiber actually gives.

Independent Gaussian noise scaled by the mean band strain flatters the
method, because distributed fiber-optic sensing does not deliver
independent errors: the reading is spatially correlated over a gauge
length, it is disturbed where the fiber crosses a crack, and it drops out
where the bond to the host is lost. This study therefore reports the
identification under three noise models at the same 5 % amplitude, with
fifty realizations each so that a spread can be quoted rather than a
range. Table 2 of the manuscript is the printed output of this script,
and figdata.py caches the same realizations for the recovery figure, so
the table and the figure cannot drift apart.

  independent   Gaussian gauge by gauge, kept for comparison
  correlated    an exponential covariance with a 150 mm correlation length,
                the order of a DFOS gauge length on this member
  dropout       independent noise plus a fraction of gauges removed entirely

Run:  python noise_study.py
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
DELTA, N_REAL, CORR_LEN = "3.5", 50, 150.0
MODELS = ("independent", "correlated", "dropout")


def correlated(rng, cx, cy, sd):
    """One draw from an exponential-covariance field, by distance kernel."""
    n = cx.size
    idx = rng.choice(n, size=min(n, 400), replace=False)
    d2 = ((cx[idx, None] - cx[None, idx]) ** 2
          + (cy[idx, None] - cy[None, idx]) ** 2)
    K = np.exp(-np.sqrt(d2) / CORR_LEN) + 1e-8 * np.eye(idx.size)
    L = np.linalg.cholesky(K)
    z = L @ rng.standard_normal(idx.size)
    # nearest-neighbor extension to the full set, which preserves the
    # correlation structure at the scale that matters here
    from scipy.spatial import cKDTree
    _, nn = cKDTree(np.c_[cx[idx], cy[idx]]).query(np.c_[cx, cy])
    return sd * z[nn]


def run_models(d=None):
    """Every recovery behind Table 2, realization by realization.

    One random stream, seeded once and consumed in a fixed loop order,
    makes the draw sequence part of the definition of the numbers: the
    figure cache calls this function rather than re-simulating with a
    second stream, so a figure panel and a table cell always come from
    the same realization. A realization without an admissible root stays
    NaN and is left out of every average.
    """
    if d is None:
        d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    thetas = [float(t) for t in d["theta_true"]]
    rng = np.random.default_rng(0)
    rec = np.full((len(MODELS), len(thetas), N_REAL), np.nan)
    for mi, model in enumerate(MODELS):
        for ti, th in enumerate(thetas):
            k = f"u_{th:.2f}_{DELTA}"
            if k not in d.files:
                continue
            lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
            cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], FD.NX, FD.NY)
            scale = float(np.abs(ex[cy < FD.BAND]).mean())
            for j in range(N_REAL):
                sd = 0.05 * scale
                if model == "correlated":
                    pert = [a + correlated(rng, cx, cy, sd)
                            for a in (ex, ey, gxy)]
                else:
                    pert = [a + rng.normal(0.0, sd, a.shape)
                            for a in (ex, ey, gxy)]
                keep = np.ones(cx.size, bool)
                if model == "dropout":
                    keep[rng.choice(cx.size, int(0.15 * cx.size),
                                    replace=False)] = False
                rec[mi, ti, j] = FD.recover_band(
                    prob, cx[keep], cy[keep], pert[0][keep], pert[1][keep],
                    pert[2][keep], area, lam, 370.0)[0]
    return np.array(thetas), rec


def main() -> None:
    d = np.load(FIELDS)
    print(f"{N_REAL} realizations, 5 % noise, correlation length "
          f"{CORR_LEN:.0f} mm, 15 % dropout\n")
    thetas, rec = run_models(d)
    print(f"{'model':>13}" + "".join(f"{f'th={t:.2f}':>14}" for t in thetas))
    for mi, model in enumerate(MODELS):
        cells = []
        for ti, th in enumerate(thetas):
            if f"u_{th:.2f}_{DELTA}" not in d.files:
                cells.append("--"); continue
            got = rec[mi, ti][np.isfinite(rec[mi, ti])]
            if got.size:
                cells.append(f"{got.mean():.3f}+-{got.std():.3f}")
            else:
                cells.append("none")
        print(f"{model:>13}" + "".join(f"{c:>14}" for c in cells), flush=True)
    print("\nTable 2 of the manuscript is this table, and figdata.py caches "
          "the same realizations for the recovery figure.")


if __name__ == "__main__":
    main()
