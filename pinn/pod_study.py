"""Probability of detection, from the machinery the study already has.

Table 2 reports what a reading looks like; an inspection plan needs the
complement, the probability that a deteriorated tie is flagged. With the
identification in hand that is Monte Carlo over the existing code: a
detection is a recovered value exceeding a threshold, the threshold is set
by the false-positive rate the operator will accept on a sound tie, and the
curve of detection probability against true loss is the POD curve of the
non-destructive evaluation literature.

The threshold is taken from the sound-tie distribution itself, as the
recovered value exceeded by 5 % of realizations at theta = 0, so the false
alarm rate is fixed by construction and the POD is read at matched PFA.

Run:  python pod_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figdata as FD                                                       # noqa: E402
from noise_study import correlated                                        # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "pod.json"
DELTA, N_REAL, NOISE, PFA = "3.5", 200, 0.05, 0.05


def draws(model, rng, cx, cy, ex, ey, gxy, sd):
    if model == "correlated":
        return [a + correlated(rng, cx, cy, sd) for a in (ex, ey, gxy)]
    return [a + rng.normal(0.0, sd, a.shape) for a in (ex, ey, gxy)]


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    rng = np.random.default_rng(2)
    thetas = [float(t) for t in d["theta_true"]]
    out = {}
    for model in ("independent", "correlated"):
        # the sound-tie distribution sets the threshold at the chosen PFA
        k = f"u_0.00_{DELTA}"
        lam = float(d[f"lam_0.00_{DELTA}"][0])
        cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], FD.NX, FD.NY)
        scale = float(np.abs(ex[cy < FD.BAND]).mean())
        sd = NOISE * scale
        sound = []
        for _ in range(N_REAL):
            pert = draws(model, rng, cx, cy, ex, ey, gxy, sd)
            r = FD.recover_band(prob, cx, cy, *pert, area, lam, 370.0)[0]
            sound.append(r if np.isfinite(r) else 0.0)
        thr = float(np.quantile(sound, 1.0 - PFA))
        pods = {}
        for th in thetas:
            if th == 0.0:
                continue
            k = f"u_{th:.2f}_{DELTA}"
            lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
            cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], FD.NX,
                                                  FD.NY)
            hits = 0
            for _ in range(N_REAL):
                pert = draws(model, rng, cx, cy, ex, ey, gxy, sd)
                r = FD.recover_band(prob, cx, cy, *pert, area, lam, 370.0)[0]
                if np.isfinite(r) and r > thr:
                    hits += 1
            pods[th] = hits / N_REAL
        out[model] = {"threshold": thr, "pod": pods}
        print(f"{model}: threshold {thr:.3f} at {100*PFA:.0f} % PFA", flush=True)
        for th, pd_ in pods.items():
            print(f"   theta = {th:.2f}: POD = {pd_:.2f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
