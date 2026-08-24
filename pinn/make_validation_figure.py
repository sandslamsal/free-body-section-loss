"""The validation figure: three measured campaigns, and what each settles.

Panel a is the headline parity: recovered against independently known
section loss for
every measured specimen the method has been put to, with the two
false-positive floors drawn about the 1:1 line so the reader can tell a
recovery that is inside the noise from one that is not.  Panel b opens
the campaign that carries the scatter, the weighed prisms, and shows the
order is recovered while the scale is not.  Panel c is the capacity
check on the milled dapped end, where the free body brackets the
measurement and the published truss idealization of the same joint does
not.  Panel d is the operating regime: the reinforcement signal the
identification lives on is absent at first yield and only saturates well
into the plastic range.

Every number here is measured or reported; nothing is refitted in this
script.

Run:  python make_validation_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import figstyle as F                                                      # noqa: E402
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.lines import Line2D                                       # noqa: E402
from matplotlib.patches import Patch, Rectangle                           # noqa: E402

FIG = HERE.parent / "figures"
F.apply()

# ---------------------------------------------------------------- data
# a: three independent measured campaigns, known -> recovered (%)
DAPPED = (34.9, 38.2)          # Desnerck et al. 2017, NS-LR, bars milled
PIERS = (33.3, 25.9)           # Bimschas 2010, VK1 against VK3

# b: Davis 2015 corroded prisms, weighed -> recovered (%)
PRISM = np.array([
    (0.0, 4.97), (0.0, -6.53), (0.0, 0.06), (0.0, 1.51),
    (0.9, -2.74), (0.6, 3.41), (0.5, 0.61), (6.1, 11.78),
    (7.1, 19.60), (3.4, 14.57), (12.3, 23.17), (13.0, 33.56),
])
PK, PR = PRISM[:, 0], PRISM[:, 1]
CTRL = PK == 0.0
SLOPE, INTERCEPT, R2, SPEARMAN = 2.310, 0.21, 0.876, 0.905

# the two measured false-positive floors (percentage points)
FLOOR_BATCH = 4.8              # batch scatter in yield strength
FLOOR_GAUGE = 1.9              # gauge noise on generated fields

# c: Desnerck et al. 2017 capacities (kN)
MEASURED = 261.9               # NS-LR, measured failure load
FB_LO, FB_HI = 248.5, 269.8    # free body, -5.1 % to +3.0 % of measured
STM = 187.0                    # source-paper strut and tie, 28.6 % low
REF = 402.3                    # NS-REF, sound reference specimen

# d: Bimschas 2010, elasticity of capacity to reinforcement ratio
DEF = np.array([10.5, 15.75, 21.0, 31.5, 42.0])
ELAS = np.array([0.111, 0.285, 0.434, 0.435, 0.312])
PEAK = 31.5                    # imposed deformation at peak load (mm)


def pm(v, nd=2):
    """Signed number with a real minus sign, for annotation text."""
    return f"{v:+.{nd}f}".replace("-", "\u2212")


# one color per campaign, carried through every panel it appears in
C_DAP, C_SWP, C_PRI = F.VERM, F.GREEN, F.SKY
GRAY = "0.45"


def panel_a(ax):
    """Recovered against independently known section loss.

    The dapped end is deliberately absent. The section loss that would be
    plotted for it, 34.9 per cent, is inferred from the measured capacity
    through this study's own chain rather than known, so it belongs in the
    capacity comparison of panel c and not on an axis of known values. What
    is known there is the milling and the capacity.
    """
    x0, x1, y0, y1 = -4.0, 42.0, -15.0, 45.0
    d = np.array([x0, x1])
    ax.fill_between(d, d - FLOOR_BATCH, d + FLOOR_BATCH, color="0.80",
                    lw=0, zorder=1)
    ax.fill_between(d, d - FLOOR_GAUGE, d + FLOOR_GAUGE, color="0.58",
                    lw=0, zorder=2)
    ax.plot(d, d, color=F.BLACK, lw=1.3, ls=(0, (5, 2)), zorder=3)
    ax.plot(PK, PR, ls="none", marker="o", ms=5.4, color=C_PRI,
            mec="white", mew=0.8, zorder=4)
    ax.plot([PIERS[0]], [PIERS[1]], ls="none", marker="^", ms=11.5,
            color=C_SWP, mec=F.BLACK, mew=1.0, zorder=6)

    handles = [
        # wrapped, because a one-line attribution makes the legend wide
        # enough to sit on the control cluster at the origin
        Line2D([], [], ls="none", marker="^", ms=8.5, color=C_SWP,
               mec=F.BLACK, mew=1.0,
               label="wall piers VK1, VK3,\nBimschas 2010"),
        Line2D([], [], ls="none", marker="o", ms=5.4, color=C_PRI,
               mec="white", mew=0.8, label="corroded prisms,\nDavis 2015"),
        Line2D([], [], color=F.BLACK, lw=1.3, ls=(0, (5, 2)), label="1:1"),
        # the axis already says "of section", and the source names have made
        # this legend wide enough to reach the data cluster near the origin
        Patch(fc="0.80", ec="none", label=f"±{FLOOR_BATCH:.1f} %, batch scatter"),
        Patch(fc="0.58", ec="none", label=f"±{FLOOR_GAUGE:.1f} %, gauge noise"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=F.FS_SMALL,
              handlelength=1.4, handletextpad=0.45, labelspacing=0.30,
              borderaxespad=0.5, borderpad=0.42)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([-10, 0, 10, 20, 30, 40])
    ax.set_xlabel("independently known section loss  (%)")
    ax.set_ylabel("recovered section loss  (%)")
    F.clean(ax)


def panel_b(ax):
    """The weighed prisms: order recovered, scale not."""
    x0, x1, y0, y1 = -1.5, 16.0, -26.0, 42.0
    d = np.array([x0, 14.6])
    ax.plot(d, d, color="0.55", lw=1.2, ls=(0, (5, 2)), zorder=2)
    ax.plot(d, SLOPE * d + INTERCEPT, color=F.BLACK, lw=1.8, zorder=3)
    ax.plot(PK[~CTRL], PR[~CTRL], ls="none", marker="o", ms=6.2,
            color=C_PRI, mec="white", mew=0.8, zorder=5)
    ax.plot(PK[CTRL], PR[CTRL], ls="none", marker="o", ms=6.2,
            mfc="white", mec=F.BLACK, mew=1.4, zorder=6)

    handles = [
        Line2D([], [], ls="none", marker="o", ms=6.2, color=C_PRI,
               mec="white", mew=0.8, label="corroded prisms, Davis 2015"),
        Line2D([], [], ls="none", marker="o", ms=6.2, mfc="white",
               mec=F.BLACK, mew=1.4,
               label=f"controls, {pm(PR[CTRL].min())} to "
                     f"{pm(PR[CTRL].max())}%"),
        Line2D([], [], color=F.BLACK, lw=1.8,
               label=f"fit, slope {SLOPE:.2f}, R² {R2:.3f}"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=F.FS_SMALL,
              handlelength=1.4, handletextpad=0.45, labelspacing=0.20,
              borderaxespad=0.5, borderpad=0.42)
    F.note(ax, 0.985, 0.545, "1:1", ha="right", va="top", color="0.45",
           fontsize=F.FS_SMALL)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlabel("weighed mass loss  (%)")
    ax.set_ylabel("recovered section loss  (%)")
    F.clean(ax)


def panel_c(ax):
    """Capacity of the milled dapped end, against two idealizations."""
    x1 = 492.0
    # Two of these bars are measurements and one is the source paper's own
    # model; only the free body is this study's. The labels say which, since
    # panel c carries no legend to say it in.
    rows = [
        (3, REF, "0.78", "NS-REF, sound", f"{REF:.1f}"),
        (2, MEASURED, F.BLACK, "NS-LR, measured", f"{MEASURED:.1f}"),
        (1, FB_LO, C_DAP, "free body, this study", f"{FB_LO:.1f}"),
        (0, STM, GRAY, "strut and tie", f"{STM:.1f}"),
    ]
    # the bracket is the fallback, so it is drawn as a band behind the
    # prediction rather than as an error bar on it
    ax.add_patch(Rectangle((FB_LO, -0.35), FB_HI - FB_LO, 2.90,
                           fc=C_DAP, alpha=0.16, ec="none", zorder=1))
    for y, v, col, name, val in rows:
        ax.barh(y, v, height=0.62, color=col, ec="none", zorder=3)
        ax.text(9.0, y, name, fontsize=F.FS_SMALL,
                color="0.20" if y == 3 else "white",
                va="center", ha="left", zorder=5)
        xt = (FB_HI if y == 1 else v) + 12.0
        ax.text(xt, y, val, fontsize=F.FS_SMALL, color="0.15",
                va="center", ha="left", zorder=5)
    ax.plot([FB_HI, FB_HI], [0.62, 1.38], color=C_DAP, lw=1.2, zorder=4)
    ax.plot([MEASURED, MEASURED], [-0.35, 2.55], color=F.BLACK, lw=1.3,
            ls=(0, (4, 2)), zorder=6)

    ax.set_xlim(0.0, x1)
    ax.set_ylim(-1.15, 3.60)
    ax.set_xticks([0, 200, 400])
    ax.set_yticks([])
    ax.set_xlabel("capacity  (kN)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_bounds(-0.45, 3.45)


def panel_d(ax):
    """Reinforcement sensitivity against imposed deformation."""
    x0, x1, y0, y1 = 7.0, 45.5, 0.0, 0.68
    ax.plot(DEF, ELAS, color=C_SWP, lw=2.2, marker="^", ms=7.0,
            mec="white", mew=1.0, zorder=4)
    ax.plot([PEAK], [ELAS[DEF == PEAK][0]], ls="none", marker="o", ms=13.0,
            mfc="none", mec=F.BLACK, mew=1.3, zorder=5)
    ax.text(PEAK, ELAS[3] + 0.030, "peak load", fontsize=F.FS_SMALL,
            color="0.15", ha="center", va="bottom", zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([10, 20, 30, 40])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("imposed deformation  (mm)")
    ax.set_ylabel("elasticity of capacity\nto reinforcement ratio")
    F.clean(ax, grid=True)


def main() -> None:
    fig = plt.figure(figsize=(F.FIG_W, 6.30))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.22, 0.88], hspace=0.36,
                          wspace=0.30, left=0.085, right=0.985, top=0.900,
                          bottom=0.065)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])
    panel_a(a)
    panel_b(b)
    panel_c(c)
    panel_d(d)

    fig.canvas.draw()
    y_top = a.get_position().y1 + 0.012
    y_bot = c.get_position().y1 + 0.012
    F.fig_panel(fig, a, "a", "recovered against known loss", y=y_top)
    F.fig_panel(fig, b, "b", "corroded prisms", y=y_top)
    F.fig_panel(fig, c, "c", "dapped end capacity",
                y=y_bot)
    F.fig_panel(fig, d, "d", "reinforcement sensitivity",
                y=y_bot)
    F.save(fig, FIG / "validation.png")
    plt.close(fig)

    print(f"  parity: dapped {DAPPED[0]}->{DAPPED[1]} "
          f"({DAPPED[1]-DAPPED[0]:+.1f} pp), piers {PIERS[0]}->{PIERS[1]} "
          f"({PIERS[1]-PIERS[0]:+.1f} pp), {PK.size} prisms")
    print(f"  controls span {PR[CTRL].min():+.2f} to {PR[CTRL].max():+.2f} pp")
    print(f"  free body lower branch {FB_LO} kN vs measured {MEASURED} kN, "
          f"{100*(1-FB_LO/MEASURED):.1f} % low; bracket to {FB_HI}; "
          f"truss {STM} kN is {100*(1-STM/MEASURED):.1f} % low")


if __name__ == "__main__":
    main()
