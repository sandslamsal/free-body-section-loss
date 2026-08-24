"""The probability-of-detection figure: the curve, and what sets it.

Panel a is the POD curve of pod_study.py: the probability that a
deteriorated tie is flagged, at a false-alarm rate fixed at 5 % on the
sound tie, under the two noise models. Panel b is the reason the two
curves differ: the sound-tie recovered-value distributions from which
each threshold is read, regenerated with the same machinery pod_study.py
uses. The correlated field parades as a structural signal, so its sound
distribution is wide and its threshold sits five times higher.

Run:  python make_pod_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figstyle as F                                                      # noqa: E402
import matplotlib.pyplot as plt                                           # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
FIG = HERE.parent / "figures"
POD = FIG / "pod.json"
# lower bound, the representative case, upper bound
MODELS = ("independent", "gauge length", "isotropic")
STYLE = {"independent": (F.SKY, (0, (1, 0)), "o"),
         "gauge length": (F.BLACK, (0, (1, 0)), "D"),
         "isotropic": (F.VERM, (0, (5, 2)), "s")}
NAME = {"independent": "independent (bound)",
        "gauge length": "gauge length, 260 mm",
        "isotropic": "isotropic 150 mm (bound)"}
CACHE = FIG / "pod_sound.npz"
DELTA, N_REAL, NOISE, PFA = "3.5", 200, 0.05, 0.05
F.apply()


# ----------------------------------------------------------------------
# the sound-tie Monte Carlo, exactly as pod_study.py runs it
# ----------------------------------------------------------------------
def sound_distributions(pod):
    """The sound-tie draws, taken straight from pod.json.

    An earlier version reconstructed these by replaying the random stream
    of pod_study.py, counting out the draws that study spent between its
    two sound loops so the thresholds would land in the same place. That
    only worked while the two scripts agreed about every intervening draw.
    detection_three_models.py now stores the distributions with the curves,
    so the figure reads what was actually used.
    """
    return {m: np.array(pod[m]["sound"]) for m in MODELS}


def main() -> None:
    pod = json.load(open(POD))
    sound = sound_distributions(pod)
    thr = {m: float(pod[m]["threshold"]) for m in MODELS}

    fig = plt.figure(figsize=(F.FIG_W, 2.95))
    gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.080, right=0.985,
                          top=0.855, bottom=0.160)

    # -- a: the POD curve at matched false-alarm rate --------------------
    a = fig.add_subplot(gs[0, 0])
    th_pct = {m: 100.0 * np.array([float(t) for t in pod[m]["pod"]])
              for m in MODELS}
    pd_ = {m: np.array(list(pod[m]["pod"].values())) for m in MODELS}
    a.axhline(0.90, color="0.45", lw=1.2, ls=(0, (4, 2)), zorder=2)
    a.text(44.5, 0.875, "target 0.90", fontsize=F.FS_ANNOT, color="0.35",
           va="top", ha="right")
    for m in MODELS:
        col, ls, mk = STYLE[m]
        a.plot(th_pct[m], pd_[m], color=col, lw=2.2, ls=ls, marker=mk, ms=5.4,
               mec="white", mew=1.0, zorder=4, label=NAME[m])
    xc = float(np.interp(0.90, pd_["isotropic"], th_pct["isotropic"]))
    a.plot([xc], [0.90], marker="o", ms=7, mfc="white", mew=1.7,
           color=F.VERM, zorder=6)
    a.set_xlabel("true section loss  (%)")
    a.set_ylabel("detection probability")
    a.set_xlim(5, 45)
    a.set_ylim(0.0, 1.05)
    a.set_xticks([10, 20, 30, 40])
    a.legend(fontsize=F.FS_SMALL, loc="lower right", handlelength=1.6,
             labelspacing=0.25, borderaxespad=0.6)
    F.clean(a, grid=True)

    # -- b: the sound-tie distributions the thresholds are read from -----
    b = fig.add_subplot(gs[0, 1])
    hi = max(float(sound[m].max()) for m in MODELS)
    bins = np.linspace(0.0, max(hi, 1.25 * thr["isotropic"]), 26)
    from matplotlib.lines import Line2D                                   # noqa: E402
    from matplotlib.patches import Patch                                  # noqa: E402
    handles = []
    for m in MODELS:
        col, _ls, _mk = STYLE[m]
        b.hist(sound[m], bins=bins, density=True, color=col, alpha=0.45,
               lw=0, zorder=2)
        b.axvline(thr[m], ymax=0.60, color=col, lw=1.8, ls=(0, (4, 2)),
                  zorder=5)
        handles.append(Patch(fc=col, alpha=0.45,
                             label=f"{NAME[m]}, {100*thr[m]:.1f} pp"))
    b.legend(handles=handles, fontsize=F.FS_SMALL, loc="upper right",
             handlelength=1.6, labelspacing=0.25, borderaxespad=0.6)
    b.set_xlabel(r"recovered section loss  $\hat\theta$, sound tie")
    b.set_ylabel("density")
    F.clean(b, grid=True)

    F.fig_panel(fig, a, "a", "probability of detection", y=0.92)
    F.fig_panel(fig, b, "b", "detection threshold on a sound tie", y=0.92)
    F.save(fig, FIG / "pod.png")
    plt.close(fig)
    print("  thresholds, points of section: "
          + ", ".join(f"{m} {100*thr[m]:.1f}" for m in MODELS))


if __name__ == "__main__":
    main()
