"""Why Table 2 and the POD caption disagree on the sound tie.

Table 2 (noise_study.py, 50 realizations) prints the sound-tie
independent recovery as 0.025 +- 0.010, yet the POD threshold
(pod_study.py, 200 realizations) is the upper 5 % quantile of the same
nominal distribution and comes out at 0.0205, below that mean. This
study resolves the contradiction by replicating one protocol and
flipping the candidate differences one at a time: the treatment of
realizations where recover_band finds no root in [0, 0.70] (noise_study
drops them, pod_study counts them as zero), the seed (0 against 2) and
the realization count (50 against 200). Everything else, noise
amplitude, perturbed components, bracket, gauges, is shared code.

Run:  python pod_table_reconcile.py
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
DELTA = "3.5"


def sound_sample(seed, n_real):
    """Raw recover_band roots on the sound tie, NaN kept as NaN."""
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    lam = float(d[f"lam_0.00_{DELTA}"][0])
    cx, cy, ex, ey, gxy = element_strains(d["xy"], d[f"u_0.00_{DELTA}"],
                                          FD.NX, FD.NY)
    sd = 0.05 * float(np.abs(ex[cy < FD.BAND]).mean())
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(n_real):
        pert = [a + rng.normal(0.0, sd, a.shape) for a in (ex, ey, gxy)]
        raw.append(FD.recover_band(prob, cx, cy, *pert, area, lam,
                                   370.0)[0])
    return np.array(raw)


def report(tag, raw):
    fin = raw[np.isfinite(raw)]
    cens = np.where(np.isfinite(raw), raw, 0.0)
    print(f"{tag}")
    print(f"  n = {raw.size}, no-root (NaN) = {np.sum(~np.isfinite(raw))},"
          f" finite roots = {fin.size},"
          f" of which exactly 0.0 = {int(np.sum(fin == 0.0))}")
    if fin.size:
        print(f"  noise_study convention (drop no-root):   mean "
              f"{fin.mean():.3f} +- {fin.std():.3f}")
    print(f"  pod_study convention (no-root -> 0):      q95 "
          f"{np.quantile(cens, 0.95):.4f}, mean {cens.mean():.4f}")
    print(f"  q95 of the root-conditional sample alone: "
          f"{np.quantile(fin, 0.95):.4f}" if fin.size else "")
    print()


def main() -> None:
    print("A. replicate noise_study exactly: seed 0, 50 realizations")
    a = sound_sample(0, 50)
    report("   (Table 2 prints 0.025 +- 0.010 from this)", a)

    print("B. one change, seed 0 -> 2, still 50 realizations")
    report("", sound_sample(2, 50))

    print("C. one more change, 50 -> 200 realizations, seed 2 "
          "(pod_study's exact sample)")
    c = sound_sample(2, 200)
    report("   (pod.json prints threshold 0.0205 from this)", c)


if __name__ == "__main__":
    main()
