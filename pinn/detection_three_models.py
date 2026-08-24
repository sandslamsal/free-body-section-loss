"""Detection under three error models, computed on one random stream.

The study previously reported detection twice: once under independent gauge
noise and an isotropic exponential covariance (the old Section 7.12), and
once under the covariance a gauge length imposes (Section 7.8). The two
were separate scripts with separate streams, so the same quantity, the
threshold on a sound tie, appeared with different values in different
places. This module computes all three on one stream and one convention,
so the three curves are comparable by construction and the paper can quote
one number per case.

The three are not alternatives of equal standing:

  independent      a lower bound. No real instrument delivers it.
  gauge length     the representative case. A fiber averages along its own
                   length, which imposes the triangular correlation
                   1 - |dx|/l exactly, and correlates nothing across the
                   depth of the cut.
  isotropic 150mm  an upper bound. An exponential covariance laid over the
                   field correlates readings across the depth as well as
                   along the tie, which is what a full-field optical method
                   does and what a fiber does not.

Convention, the same one the manuscript states once: the threshold is the
95th percentile of the recovered value over ALL sound-tie draws, a draw
admitting no root counted as zero. It is not the root-conditional mean of
Table 2.

Run:  /usr/local/bin/python3.12 detection_three_models.py
      -> figures/pod.json   (consumed by make_pod_figure.py)
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
from gauge_length_noise import gauge_average, gauge_noise                  # noqa: E402
from noise_study import correlated                                         # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "pod.json"

DELTA, N_REAL, NOISE, PFA, ARM = "3.5", 200, 0.05, 0.05, 370.0
ELL = 260.0
MODELS = ("independent", "gauge length", "isotropic")


def perturb(model, rng, cx, cy, ex, ey, gxy, sd):
    """One draw of the measured field under the named error model."""
    if model == "gauge length":
        base = [gauge_average(cx, cy, ex, ELL), ey, gxy]
        return [b + gauge_noise(rng, cx, cy, sd, ELL) for b in base]
    if model == "isotropic":
        return [a + correlated(rng, cx, cy, sd) for a in (ex, ey, gxy)]
    return [a + rng.normal(0.0, sd, a.shape) for a in (ex, ey, gxy)]


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    thetas = [float(t) for t in d["theta_true"]]
    rng = np.random.default_rng(2)
    out = {}

    print("Detection under three error models, one stream, one convention")
    print("=" * 68)
    for model in MODELS:
        k = f"u_0.00_{DELTA}"
        lam = float(d[f"lam_0.00_{DELTA}"][0])
        cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], FD.NX, FD.NY)
        sd = NOISE * float(np.abs(ex[cy < FD.BAND]).mean())

        sound = []
        for _ in range(N_REAL):
            r = FD.recover_band(prob, cx, cy,
                                *perturb(model, rng, cx, cy, ex, ey, gxy, sd),
                                area, lam, ARM)[0]
            sound.append(r if np.isfinite(r) else 0.0)
        sound = np.array(sound)
        thr = float(np.quantile(sound, 1.0 - PFA))
        rooted = float(np.isfinite(sound).mean())

        pods = {}
        for th in thetas:
            if th == 0.0:
                continue
            kk = f"u_{th:.2f}_{DELTA}"
            lamd = float(d[f"lam_{th:.2f}_{DELTA}"][0])
            cxd, cyd, exd, eyd, gd = element_strains(d["xy"], d[kk],
                                                     FD.NX, FD.NY)
            hits = 0
            for _ in range(N_REAL):
                r = FD.recover_band(prob, cxd, cyd,
                                    *perturb(model, rng, cxd, cyd, exd, eyd,
                                             gd, sd),
                                    area, lamd, ARM)[0]
                hits += bool(np.isfinite(r) and r > thr)
            pods[f"{th:.2f}"] = hits / N_REAL

        out[model] = {"threshold": thr, "pod": pods,
                      "sound_mean": float(sound.mean()),
                      "sound_sd": float(sound.std()),
                      "sound_rooted": rooted,
                      "sound": [float(v) for v in sound]}
        print(f"  {model:>13}: threshold {100*thr:5.1f} pp   "
              + "  ".join(f"P({k})={v:.2f}" for k, v in pods.items()))

    a = out["independent"]["pod"]["0.10"]
    g = out["gauge length"]["pod"]["0.10"]
    i = out["isotropic"]["pod"]["0.10"]
    print(f"\n  at ten per cent loss the representative case is {g:.2f}, "
          f"between {i:.2f} and {a:.2f}")
    out["_meta"] = {
        "convention": "95th percentile of the recovered value over all "
                      "sound-tie draws, a draw admitting no root counted as "
                      "zero; not the root-conditional mean of Table 2",
        "n_real": N_REAL, "pfa": PFA, "noise": NOISE, "ell_mm": ELL,
        "standing": {"independent": "lower bound, no instrument delivers it",
                     "gauge length": "representative for a bonded fiber",
                     "isotropic": "upper bound, correlates across the depth"}}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
