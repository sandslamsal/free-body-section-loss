"""The value-of-information figure: the decision diagram, and what arm error does to it.

Panel a is the decision the reading buys: the expected annual cost of
doing nothing, as a function of the reading, against the flat cost of
repairing now, one line per repair cost. The crossing on the C_r = 0.001
line is the only decision boundary that exists: the posterior failure
cost tops out at 0.008, so no reading ever reaches the 0.01 line, and at
the two larger repair costs the reading cannot change the decision at
all. Panel b prices that in decision currency against the arm tolerance
of this study's identification: an arm assumed short only suppresses
repair, which is the prior action anyway, so the value decays to zero
without crossing it; an arm assumed long triggers repairs the posterior
cannot justify and drives the value of the reading below zero at 32 mm.

Run:  python make_voi_figure.py
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

FIG = HERE.parent / "figures"
VOI = FIG / "voi.json"
F.apply()

CR_KEYS = ("1e-03", "1e-02", "1e-01")
CR_COL = {"1e-03": F.GREEN, "1e-02": F.SKY, "1e-01": F.VERM}
CR_LS = {"1e-03": (0, (5, 2)), "1e-02": (0, (6, 2, 1.4, 2)),
         "1e-01": (0, (1.4, 1.3))}
CR_NAME = {"1e-03": "0.001", "1e-02": "0.01", "1e-01": "0.1"}


def main() -> None:
    d = json.load(open(VOI))

    fig = plt.figure(figsize=(F.FIG_W, 2.95))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.085, right=0.985,
                          top=0.855, bottom=0.165)

    # -- a: expected annual cost of each action against the reading -------
    a = fig.add_subplot(gs[0, 0])
    hat = np.array(d["reading"]["grid"])
    epf = np.array(d["reading"]["Epf_posterior"])
    sl = (hat >= -0.10) & (hat <= 0.80)
    a.set_yscale("log")
    for key in CR_KEYS:
        y = d["decision"][key]["cost_repair"]
        a.axhline(y, color=CR_COL[key], ls=CR_LS[key],
                  lw=2.4 if key == "1e-02" else 1.8, zorder=2)
        a.text(-0.09, y / 1.15, f"repair, $C_r$ = {CR_NAME[key]}",
               fontsize=F.FS_TICK, color=CR_COL[key], va="top", ha="left",
               zorder=5)
    a.plot(hat[sl], epf[sl], color=F.BLACK, lw=2.4, zorder=4)
    a.annotate("do nothing", xy=(0.065, float(np.interp(0.065, hat, epf))),
               xytext=(-0.055, 1.8e-4), fontsize=F.FS_ANNOT, color="black",
               va="top", zorder=5,
               arrowprops=dict(arrowstyle="->", lw=0.9, color="0.35",
                               shrinkA=2, shrinkB=3))
    hs = d["decision"]["1e-03"]["hat_star"]
    yr = d["decision"]["1e-03"]["cost_repair"]
    p_rep = d["decision"]["1e-03"]["p_repair_unbiased"]
    a.plot([hs], [yr], marker="o", ms=7, mfc="white", mec=F.GREEN,
           mew=1.7, ls="none", zorder=6)
    a.annotate(f"repair pays beyond {hs:.2f}\n"
               f"({100.0 * p_rep:.1f} % of readings)",
               xy=(hs, yr), xytext=(0.55, 2.1e-4), fontsize=F.FS_ANNOT,
               color=F.GREEN, ha="center", va="top", zorder=5,
               arrowprops=dict(arrowstyle="->", lw=0.9, color=F.GREEN,
                               connectionstyle="arc3,rad=0.25",
                               shrinkA=2, shrinkB=6))
    a.annotate("no reading reaches the 0.01 line",
               xy=(0.76, float(np.interp(0.76, hat, epf))),
               xytext=(0.42, 2.4e-2), fontsize=F.FS_ANNOT, color="0.30",
               ha="center", va="top", zorder=5,
               arrowprops=dict(arrowstyle="->", lw=0.9, color="0.45",
                               connectionstyle="arc3,rad=-0.2",
                               shrinkA=2, shrinkB=4))
    a.set_xlim(-0.10, 0.80)
    a.set_ylim(4.5e-6, 0.45)
    a.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    a.set_xlabel(r"reading  $\hat\theta$")
    a.set_ylabel("expected annual cost  /  $C_f$")
    F.clean(a, grid=True)

    # -- b: VoI against arm tolerance, one curve per repair cost ----------
    b = fig.add_subplot(gs[0, 1])
    da = np.array(d["voi_vs_arm_mm"]["da_mm"])
    pfp = d["prior"]["pf_prior"]

    def pct(key, branch):
        return 100.0 * np.array(d["voi_vs_arm_mm"][key][branch]) / pfp

    b.plot(da, pct("1e-01", "arm_short"), color=CR_COL["1e-01"],
           ls=CR_LS["1e-01"], lw=2.4, zorder=2,
           label=CR_NAME["1e-01"] + " (zero)")
    b.plot(da, pct("1e-02", "arm_short"), color=CR_COL["1e-02"],
           ls=CR_LS["1e-02"], lw=1.8, zorder=3,
           label=CR_NAME["1e-02"] + " (zero)")
    b.plot(da, pct("1e-03", "arm_short"), color=CR_COL["1e-03"], ls="-",
           lw=2.4, marker="o", ms=5.5, mec="white", mew=0.9, zorder=4,
           label="0.001, arm short")
    b.plot(da, pct("1e-03", "arm_long"), color=CR_COL["1e-03"],
           ls=(0, (5, 2)), lw=2.0, marker="s", ms=5, mec="white", mew=0.9,
           zorder=4, label="0.001, arm long")
    xc = d["crossings_mm"]["1e-03"]["arm_long_bias_plus"]
    b.plot([xc], [0.0], marker="o", ms=7, mfc="white", mec=F.GREEN,
           mew=1.7, ls="none", zorder=6)
    b.annotate("worth less than not\n"
               f"measuring beyond {xc:.0f} mm",
               xy=(xc, -0.8), xytext=(74.0, -8.0), fontsize=F.FS_ANNOT,
               color=F.GREEN, ha="center", va="top", zorder=5,
               arrowprops=dict(arrowstyle="->", lw=0.9, color=F.GREEN,
                               connectionstyle="arc3,rad=0.22",
                               shrinkA=2, shrinkB=4))
    vl_end = 100.0 * d["voi_vs_arm_mm"]["1e-03"]["arm_long"][-1] / pfp
    b.text(49.0, -15.3, f"reaches {vl_end:.0f} % at 120 mm",
           fontsize=F.FS_SMALL, color=F.GREEN, va="bottom", ha="left",
           zorder=5)
    hnd, lab = b.get_legend_handles_labels()
    order = [2, 3, 1, 0]
    b.legend([hnd[i] for i in order], [lab[i] for i in order],
             fontsize=F.FS_SMALL, loc="upper center", ncol=2,
             title="repair cost $C_r$  ($C_f$ = 1)",
             handlelength=1.6, columnspacing=1.0, labelspacing=0.25,
             borderaxespad=0.35)
    b.set_xlim(-3.0, 123.0)
    b.set_ylim(-16.0, 19.0)
    b.set_xticks([0, 30, 60, 90, 120])
    b.set_yticks([-15, -10, -5, 0, 5, 10, 15])
    b.set_xlabel(r"arm error  $\Delta a$  (mm)")
    b.set_ylabel("value of information  (% of prior risk)")
    F.clean(b, grid=True)

    F.fig_panel(fig, a, "a", "expected cost of each action", y=0.92)
    F.fig_panel(fig, b, "b", "value of the reading vs arm error", y=0.92)
    probs = F.audit(fig)
    print(f"  audit: {len(probs)} complaint(s)")
    F.save(fig, FIG / "voi.png", check=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
