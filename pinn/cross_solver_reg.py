"""The cross-model offset, re-measured against the regularized generator.

The 27.8 pp intercept reported by cross_solver.py was measured on fields
from a fixed-crack model whose compressive descending branch was not
crack-band regularized: its theta = 0 capacity at the 3.5 mm station moved
25 per cent between 40x20 and 80x40, and on 80x40 capacity rose with
damage. This module re-runs the identification, entirely unchanged
(figdata.recover_band, the same wide trial grid, the same per-state
measured reaction arm), on fields_crossmodel_reg.npz, generated after the
compressive branch was regularized on Gc = 250 Gf.

It first prints what the offset is being measured against: capacity versus
theta on every mesh (monotonicity), and the theta = 0 capacity across
40x20, 60x30 and 80x40 (mesh convergence), so the generator's authority is
measured rather than asserted before its fields are identified.

Run:  python cross_solver_reg.py      (writes figures/cross_solver_reg.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import cross_solver as CS                                                  # noqa: E402
import figdata as FD                                                       # noqa: E402
from problem import DeepBeam                                               # noqa: E402

ORACLE = HERE.parent / "oracle"
OUT = HERE.parent / "figures" / "cross_solver_reg.json"
FNAME = "fields_crossmodel_reg.npz"
MESHES = ["40x20", "60x30", "80x40"]
THETA = CS.THETA


def capacity_tables():
    """lam at the 3.5 mm station and the smoothed peak, per mesh and theta."""
    d = np.load(ORACLE / FNAME)
    station, peak = {}, {}
    for tag in MESHES:
        ls, ps = [], []
        for th in THETA:
            key = f"{tag}_{th:.2f}_3.5"
            ls.append(float(d[f"lam_{key}"][0]))
            c = d[f"curve_{key}"]
            k = 25
            lam_s = np.convolve(c[:, 1], np.ones(k) / k, mode="same")
            lam_s[:k], lam_s[-k:] = c[:k, 1], c[-k:, 1]
            ps.append(float(lam_s.max()))
        station[tag], peak[tag] = ls, ps
    return station, peak


def main() -> None:
    station, peak = capacity_tables()

    print("capacity versus theta, regularized model (Gc = 250 Gf)")
    print(f"{'theta':>7}" + "".join(f"{t + ' @3.5':>14}" for t in MESHES)
          + "".join(f"{t + ' peak':>14}" for t in MESHES))
    for i, th in enumerate(THETA):
        print(f"{th:>7.2f}"
              + "".join(f"{station[t][i]:>14.4f}" for t in MESHES)
              + "".join(f"{peak[t][i]:>14.4f}" for t in MESHES))
    for t in MESHES:
        mono = all(a > b for a, b in zip(station[t], station[t][1:]))
        monp = all(a > b for a, b in zip(peak[t], peak[t][1:]))
        print(f"  {t}: station monotone {mono}, peak monotone {monp}")
    l0 = [station[t][0] for t in MESHES]
    print(f"theta=0 station capacity by mesh: "
          + ", ".join(f"{t} {v:.4f}" for t, v in zip(MESHES, l0))
          + f"   (80x40 vs 60x30: {100 * (l0[2] / l0[1] - 1.0):+.2f} %,"
          f" 80x40 vs 40x20: {100 * (l0[2] / l0[0] - 1.0):+.2f} %)")

    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    fams, checks = [], {}
    for tag in ("40x20", "80x40"):
        st, arms, ck = CS.load_cross(tag, fname=FNAME)
        checks[tag] = ck
        fams.append(CS.family(f"regularized model: fixed crack, Q4 {tag}",
                              st, prob, area, arms=arms))
    for f in fams:
        CS.table(f)
        print(f"        affine (wide): {f['affine_wide']}")

    print("\nadmissibility of the regularized fields, on their own stresses")
    print(f"{'mesh':>8}{'theta':>7}{'M close':>9}{'V close':>9}"
          f"{'reactions':>11}{'x_R mm':>8}{'peak lam':>10}{'at mm':>8}"
          f"{'lam/peak':>11}{'unconv':>8}")
    for tag, ck in checks.items():
        for c in ck:
            print(f"{tag:>8}{c['theta']:>7.2f}{c['M_closure']:>9.3f}"
                  f"{c['V_closure']:>9.3f}{c['global_closure']:>11.4f}"
                  f"{c['x_reaction_mm']:>8.0f}{c['peak_lam']:>10.3f}"
                  f"{c['peak_delta_mm']:>8.2f}{c['lam_over_peak']:>11.3f}"
                  f"{c['unconverged_steps'][0]:>5}/"
                  f"{c['unconverged_steps'][1]}")

    out = {"what": "cross-model identification against the crack-band "
                   "regularized fixed-crack generator",
           "gc": "Gc = 250 Gf = 32.5 N/mm (RTD 1016; Nakamura-Higai 2001 "
                 "gives 48 N/mm, same order)",
           "unregularised_baseline_40x20": {"slope": 0.7697,
                                            "intercept": -0.2779},
           "same_solver_baseline": {"slope": 0.8317, "intercept": -0.0092},
           "capacity_station": station, "capacity_peak": peak,
           "families": fams, "field_checks": checks}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
