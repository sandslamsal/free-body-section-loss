# -*- coding: utf-8 -*-
"""Graphical abstract: the measurement, the statics, and the recovery.

Elsevier wants a single landscape panel that carries the argument at one
glance, so this is one canvas drawn in inches rather than a grid of small
subplots: the instrumented member on the left, the free body and the
identifying condition in the middle, and the parity of the recovery
against measured specimens on the right.

The canvas axes maps one data unit to one printed inch, so every
coordinate below is the position it prints at.  Type is set well above the
house floor because the panel is read at roughly 200 pixels tall in a
table of contents, where anything finer disappears.

Numbers in the parity panel are the measured campaigns of the validation
figure; nothing is refitted here.

Run:  python make_graphical_abstract.py
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
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,          # noqa: E402
                                Polygon, Rectangle)

FIG = HERE.parent / "figures"
F.apply()
# the abstract keeps the canvas it declares: Elsevier sizes the panel by
# its aspect, and a tight crop would hand them a different one
plt.rcParams["savefig.bbox"] = None

W, H = 7.5, 3.0                 # printed inches
SKY_DK = "#0072B8"              # the tie color, darkened for lines on the band
GRAY = "0.35"

# type sizes for this panel, all above the 6.5 pt floor
FS_HEAD = 11.5                  # zone headings
FS_EQ = 17.0                    # the identifying condition
FS_SYM = 11.0                   # R, T, C on the free body
FS_TAG = 8.5                    # small in-figure tags
FS_TAKE = 10.0                  # the one takeaway line
FS_AX = 9.5                     # parity axis labels
FS_TICK = 8.5
FS_LEG = 7.5

# ---------------------------------------------------------------- geometry
# zone 1: the member, drawn at 1.96 in for its 2000 mm span
BX0, BY0, BY1 = 0.28, 1.42, 2.40
SC = 1.96 / 2000.0              # in per mm, shared with the free body
BAND = BY0 + 0.98 * 0.150       # top of the tie band
XCUT_MM = 700.0                 # the cut station
GAUGE_MM = (200.0, 500.0, 800.0, 1100.0, 1400.0, 1700.0)

# zone 2: the retained portion, same scale and same height on the page
FX0 = 3.10
FX1 = FX0 + XCUT_MM * SC        # the cut face
XR = FX0 + 250.0 * SC           # the reaction, at the support center

XC1, XC2, XC3 = 1.26, 3.68, 6.40        # zone centers, for the headings
Y_HEAD = 2.78

# ------------------------------------------------------------------- data
# the measured campaigns, known -> recovered section loss (%)
DAPPED = (34.9, 38.2)           # Desnerck et al. 2017, NS-LR, bars milled
PIERS = (33.3, 25.9)            # Bimschas 2010, VK1 against VK3
PRISM = np.array([
    (0.0, 4.97), (0.0, -6.53), (0.0, 0.06), (0.0, 1.51),
    (0.9, -2.74), (0.6, 3.41), (0.5, 0.61), (6.1, 11.78),
    (7.1, 19.60), (3.4, 14.57), (12.3, 23.17), (13.0, 33.56),
])
C_DAP, C_SWP, C_PRI = F.VERM, F.GREEN, F.SKY


def bx(mm):
    """Page abscissa of a station on the member, in inches."""
    return BX0 + mm * SC


def fx(mm):
    """Page abscissa of a station on the free body, in inches."""
    return FX0 + mm * SC


def flow_arrow(ax, x0, x1, y):
    """The chunky arrow that separates two zones."""
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="simple,head_width=15,head_length=13,tail_width=6.0",
        mutation_scale=1.0, fc="0.74", ec="none", zorder=2))


def load_arrow(ax, x, y0, y1, color, lw=2.4, ms=15):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=7))


def side_arrow(ax, x0, x1, y, color, lw=2.2, ms=13):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=7))


def support(ax, xc, ytop, bold=False):
    """Bearing plate and triangle, drawn under a member edge."""
    ax.add_patch(Rectangle((xc - 0.098, ytop - 0.062), 0.196, 0.062,
                           fc="0.35", ec="none", zorder=5))
    lw = 1.2 if bold else 0.9
    ax.add_patch(Polygon([(xc, ytop - 0.062), (xc - 0.072, ytop - 0.205),
                          (xc + 0.072, ytop - 0.205)], closed=True,
                         fc="white", ec=F.BLACK, lw=lw, zorder=5))
    ax.plot([xc - 0.090, xc + 0.090], [ytop - 0.222] * 2, color=F.BLACK,
            lw=lw, zorder=5)


# ------------------------------------------------------------------ zone 1
def zone_measured(ax):
    """The member, its tie band, the gauges, the load and the cut."""
    xc = bx(XCUT_MM)
    ax.add_patch(Rectangle((BX0, BY0), 1.96, 0.98, fc="0.945", ec="none",
                           zorder=1))
    # the portion the free body keeps, tinted so the cut reads as a choice
    ax.add_patch(Rectangle((BX0, BY0), xc - BX0, 0.98, fc="0.875",
                           ec="none", zorder=1))
    ax.add_patch(Rectangle((BX0, BY0), 1.96, BAND - BY0, fc=F.SKY,
                           alpha=0.38, ec="none", zorder=2))
    ax.plot([BX0, BX0 + 1.96], [0.5 * (BY0 + BAND)] * 2, color=SKY_DK,
            lw=1.3, zorder=3)
    for mm in GAUGE_MM:
        ax.plot([bx(mm)] * 2, [BY0 + 0.018, BAND - 0.012], color=SKY_DK,
                lw=2.2, solid_capstyle="butt", zorder=4)
    ax.add_patch(Rectangle((BX0, BY0), 1.96, 0.98, fc="none", ec=F.BLACK,
                           lw=1.1, zorder=6))

    support(ax, bx(250.0), BY0, bold=True)
    support(ax, bx(1750.0), BY0)
    ax.add_patch(Rectangle((bx(900.0), BY1), 200.0 * SC, 0.055, fc="0.35",
                           ec="none", zorder=5))
    load_arrow(ax, bx(1000.0), 2.68, 2.47, F.VERM)

    ax.plot([xc, xc], [1.30, 2.52], color=F.VERM, lw=1.5,
            ls=(0, (4.5, 2.6)), zorder=8)
    ax.text(xc, 2.555, "cut", ha="center", va="bottom", fontsize=FS_TAG,
            color=F.VERM, zorder=9)

    # What the gauges read. A bonded fiber in cracked concrete returns a
    # sawtooth peaking at every crack, not the smooth envelope: that is the
    # whole of Section 7.7, so the abstract draws the sawtooth and the
    # gauge-length average the method actually consumes.
    t = np.linspace(0.0, 1.0, 400)
    env = 0.62 + 0.30 * np.sin(np.pi * t) ** 0.55          # smeared mean
    n_cr = 5.0                                             # cracks on the span
    tri = 2.0 * np.abs(((n_cr * t) % 1.0) - 0.5)           # 0 at crack, 1 mid
    # the tooth amplitude follows the tension: no tie force at the supports
    # means no crack-to-crack variation there either
    saw = env + 0.16 * ((env - 0.62) / 0.30) * (0.5 - tri)  # zero mean, so the
    #                                    dashed envelope really is the average
    ax.plot([BX0, BX0 + 1.96], [0.62] * 2, color="0.62", lw=0.9, zorder=2)
    # the shading is the quantity the method consumes; the spiky line is what
    # the fiber actually returns
    ax.fill_between(bx(2000.0 * t), 0.62, env, color=F.SKY, alpha=0.22,
                    lw=0, zorder=2)
    ax.plot(bx(2000.0 * t), saw, color=SKY_DK, lw=1.1, alpha=0.75, zorder=3)
    ax.plot(bx(2000.0 * t), env, color=SKY_DK, lw=2.2, zorder=4)
    ax.text(BX0, 0.545, r"$\varepsilon$ along the tie: raw, and averaged"
            "\n" r"over a crack spacing", ha="left",
            va="top", fontsize=FS_TAG, color="0.25", zorder=5)


# ------------------------------------------------------------------ zone 2
def zone_statics(ax):
    """The retained portion, the reaction and its arm, the cut tractions."""
    ax.add_patch(Rectangle((FX0, BY0), FX1 - FX0, 0.98, fc="0.875",
                           ec="none", zorder=1))
    ax.add_patch(Rectangle((FX0, BY0), FX1 - FX0, BAND - BY0, fc=F.SKY,
                           alpha=0.38, ec="none", zorder=2))
    ax.plot([FX0, FX1], [0.5 * (BY0 + BAND)] * 2, color=SKY_DK, lw=1.3,
            zorder=3)
    ax.add_patch(Rectangle((FX0, BY0), FX1 - FX0, 0.98, fc="none",
                           ec=F.BLACK, lw=1.1, zorder=6))
    ax.plot([FX1, FX1], [1.30, 2.52], color=F.VERM, lw=1.5,
            ls=(0, (4.5, 2.6)), zorder=8)

    # the tie force on the cut face, and the compression that closes the couple
    side_arrow(ax, FX1, FX1 + 0.36, 0.5 * (BY0 + BAND), SKY_DK, lw=2.6,
               ms=15)
    ax.text(FX1 + 0.40, 0.5 * (BY0 + BAND), "$T$", ha="left", va="center",
            fontsize=FS_SYM, color=SKY_DK, zorder=9)
    for y, ln in ((2.26, 0.25), (2.12, 0.20)):
        side_arrow(ax, FX1, FX1 - ln, y, GRAY, lw=1.8, ms=11)
    ax.text(FX1 + 0.05, 2.19, "$C$", ha="left", va="center",
            fontsize=FS_SYM, color=GRAY, zorder=9)

    # the reaction and the arm it acts on.  A free body carries the force,
    # not the support symbol, so only the bearing plate is kept here
    ax.add_patch(Rectangle((XR - 0.098, BY0 - 0.062), 0.196, 0.062,
                           fc="0.35", ec="none", zorder=5))
    load_arrow(ax, XR, 1.14, 1.350, F.VERM, lw=2.6, ms=15)
    ax.text(XR - 0.075, 1.24, "$R$", ha="right", va="center",
            fontsize=FS_SYM, color=F.VERM, zorder=9)
    ax.annotate("", xy=(FX1, 1.10), xytext=(XR, 1.10),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="0.45",
                                shrinkA=0, shrinkB=0))
    ax.text(0.5 * (XR + FX1), 1.125, "arm", ha="center", va="bottom",
            fontsize=FS_TAG, color="0.25", zorder=9)


def identifying_condition(ax):
    """The condition itself, centered on the zone and boxed."""
    y_eq, y_sub = 0.82, 0.56
    lhs = ax.text(3.57, y_eq, r"$T(\theta)$", ha="right", va="center",
                  fontsize=FS_EQ, color=SKY_DK, zorder=9)
    mid = ax.text(3.68, y_eq, "$=$", ha="center", va="center",
                  fontsize=FS_EQ, color=F.BLACK, zorder=9)
    rhs = ax.text(3.79, y_eq, r"$T_\mathrm{req}$", ha="left", va="center",
                  fontsize=FS_EQ, color=F.VERM, zorder=9)

    ax.figure.canvas.draw()
    cl = 0.5 * sum(extent(ax, lhs)[0::2])
    cr = 0.5 * sum(extent(ax, rhs)[0::2])
    sub_l = ax.text(cl, y_sub, "measured", ha="center", va="center",
                    fontsize=FS_TAG, color=SKY_DK, zorder=9)
    sub_r = ax.text(cr, y_sub, "statics", ha="center", va="center",
                    fontsize=FS_TAG, color=F.VERM, zorder=9)

    items = [lhs, mid, rhs, sub_l, sub_r]
    x0, y0, x1, y1 = union(ax, items)
    dx = XC2 - 0.5 * (x0 + x1)
    for t in items:
        t.set_x(t.get_position()[0] + dx)
    x0, y0, x1, y1 = union(ax, items)
    ax.add_patch(FancyBboxPatch(
        (x0 - 0.075, y0 - 0.070), (x1 - x0) + 0.150, (y1 - y0) + 0.140,
        boxstyle="round,pad=0,rounding_size=0.055", fc="#FFF3EE",
        ec=F.VERM, lw=1.0, alpha=0.95, zorder=5))


def extent(ax, t):
    """Data-coordinate (x0, y0, x1, y1) of a drawn Text."""
    bb = t.get_window_extent(renderer=F._renderer(ax.figure))
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return x0, y0, x1, y1


def union(ax, items):
    ax.figure.canvas.draw()
    b = np.array([extent(ax, t) for t in items])
    return b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()


# ------------------------------------------------------------------ zone 3
def zone_parity(fig):
    """Recovered against known section loss, on measured specimens."""
    ax = fig.add_axes([5.65 / W, 0.84 / H, 1.75 / W, 1.56 / H])
    # the abscissa runs past the last specimen so the legend has a corner
    # of its own: nothing here is allowed to sit on the 1:1 line
    lo, hi, xhi = -4.0, 44.0, 46.0
    ax.plot([lo, hi], [lo, hi], color=F.BLACK, lw=1.3, ls=(0, (5, 2)),
            zorder=2)
    ax.plot(PRISM[:, 0], PRISM[:, 1], ls="none", marker="o", ms=5.2,
            color=C_PRI, mec="white", mew=0.8, zorder=4)
    ax.plot([PIERS[0]], [PIERS[1]], ls="none", marker="^", ms=10.0,
            color=C_SWP, mec=F.BLACK, mew=0.9, zorder=5)
    handles = [
        Line2D([], [], ls="none", marker="^", ms=7.0, color=C_SWP,
               mec=F.BLACK, mew=0.9, label="wall piers VK1, VK3"),
        Line2D([], [], ls="none", marker="o", ms=5.2, color=C_PRI,
               mec="white", mew=0.8, label="corroded prisms"),
    ]
    # the 1:1 line is named where it runs clear of every specimen, which
    # keeps the legend one row shorter and off the line itself
    ax.text(25.0, 17.5, "1:1", ha="center", va="center", fontsize=FS_LEG,
            color=F.BLACK, zorder=3)
    # the dapped end belongs to a capacity comparison, not to these axes:
    # its section loss would be inferred rather than known
    ax.text(-2.0, 53.0, "dapped end enters as capacity,\npredicted 5 % low",
            ha="left", va="top", fontsize=FS_LEG, color=C_DAP, zorder=6)
    ax.legend(handles=handles, loc="lower right", fontsize=FS_LEG,
              handlelength=0.9, handletextpad=0.38, labelspacing=0.20,
              borderaxespad=0.30, borderpad=0.32, framealpha=0.94)
    ax.set_xlim(lo, xhi)
    ax.set_ylim(-12.0, 54.0)
    ax.set_xticks([0, 20, 40])
    ax.set_yticks([0, 20, 40])
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel("known  (%)", fontsize=FS_AX, labelpad=2.0)
    ax.set_ylabel("recovered  (%)", fontsize=FS_AX, labelpad=2.0)
    F.clean(ax)
    return ax


# --------------------------------------------------------------------- main
def main() -> None:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, W)
    ax.set_ylim(0.0, H)
    ax.axis("off")

    zone_measured(ax)
    zone_statics(ax)
    flow_arrow(ax, 2.44, 2.94, 1.91)
    flow_arrow(ax, 4.60, 5.10, 1.91)
    zone_parity(fig)
    identifying_condition(ax)

    for x, s in ((XC1, "measured strain"),
                 (XC2, "free body and statics"),
                 (XC3, "recovered section loss")):
        ax.text(x, Y_HEAD, s, ha="center", va="baseline", fontsize=FS_HEAD,
                fontweight="bold", color="0.15", zorder=9)

    ax.text(0.5 * W, 0.24,
            "a root, not an optimization, validated against published tests",
            ha="center", va="center", fontsize=FS_TAKE, color="0.15",
            zorder=9)

    F.save(fig, FIG / "graphical_abstract.png")
    plt.close(fig)
    print(f"  parity: dapped {DAPPED[0]}->{DAPPED[1]}, "
          f"piers {PIERS[0]}->{PIERS[1]}, {len(PRISM)} prisms")


if __name__ == "__main__":
    main()
