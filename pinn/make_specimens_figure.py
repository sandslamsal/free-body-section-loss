# -*- coding: utf-8 -*-
"""Every structure the study validates against, drawn to scale, one per panel.

The validation section reports five measured campaigns and the reader is
never shown any of them.  This figure is the orientation: each panel is an
elevation of one specimen carrying three things and no more, the geometry,
the reinforcement that governs, and a mark of where the measurement was
taken, with one line naming the quantity that was independently known.  The
dapped end is the richest case and takes two panels, the specimen and the
free body the identification cuts from it.

A fixed visual system runs across the panels: reinforcement is vermilion,
instrumentation is sky, an imposed or independently known deterioration is
amber, and a free-body cut is a heavy black dash.

Provenance, panel by panel.
  a, b  Desnerck, Lees and Morley 2016/2017/2018 open data.  Dimensions
        from halfjoint_geometry.py, milling from data/desnerck_tables.csv,
        bar forces from data/desnerck_fig16_barforces.csv, failure loads
        from halfjoint_identify.FAIL.
  c     Bimschas 2010, ETH diss. 18849, Tab. 5.1, 5.3 and 5.7 as carried in
        pinn/bimschas_theta.py; the section is stated verbatim in the
        thesis text ("Lv = 3.3 m, lw = 1.5 m, and bw = 0.35 m").
  d     Davis 2015 MASc thesis, prism block in the header of
        data/davis2015_tables.csv, recovery in data/davis2015_recovery.csv.
  e     Davis, Hoult and Scott 2017, every dimension from
        data/davis2017_geometry.json.

The instrumented wall of Fernandez, Berrocal and Rempling 2023 had a panel
here and no longer does: Fig. wall plots the same specimen from the same
open gauge coordinates, magnifies the 20 mm gap this panel could only
assert, and belongs to the section that uses it.

Nothing here is refitted and no dimension is invented.  Quantities the
sources print are drawn and labeled; quantities that are only digitised
are drawn but never labeled with a number.

Run:  /usr/local/bin/python3.12 make_specimens_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import figstyle as F                                                      # noqa: E402
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle                 # noqa: E402
from matplotlib.lines import Line2D                                       # noqa: E402
from halfjoint_geometry import (H, H_NIB, NODE, STIRRUPS, X_BEARING,      # noqa: E402
                                X_CORNER)
from halfjoint_identify import FAIL                                       # noqa: E402

DATA = HERE.parent / "data"
FIG = HERE.parent / "figures"
F.apply()

# ------------------------------------------------------------ house colors
C_CONC = "0.93"
C_STEEL = F.VERM          # reinforcement
C_MEAS = F.SKY            # instrumentation, and what it measured
C_KNOWN = F.ORANGE        # the independently known deterioration
C_DIM = "0.38"
C_TXT = "0.15"
C_NOTE = "0.32"
C_SRC = "0.50"
FS = F.FS_SMALL           # the working label size in these compact panels

# ------------------------------------------------- grid of compact panels
FIG_H = 7.60
#            0 = left      1 = right     2 = full width, for a lone panel
COL_X = (0.045, 0.505, 0.045)
COL_W = (0.450, 0.450, 0.910)
ROW_Y = (0.700, 0.412, 0.045)
ROW_H = (0.245, 0.245, 0.330)


# ------------------------------------------------------- a, b: dapped end
# nib steel, from the halfjoint_geometry docstring [src, 2018 Sec. 4.1]
N_UBAR, D_UBAR, N_DIAG, D_DIAG, D_STIR = 3, 12, 4, 12, 10
Y_UBAR = NODE["A"][1]      # 298, the horizontal nib tie          [dig]
Y_TOP = 50.0               # top chord the U-bars return to       [dig]
Y_BOT = NODE["F"][1]       # 661, bottom longitudinal bars        [dig]
X_LAP = NODE["G"][0]       # 649                                  [dig]
CRACK_DEG = 60.0           # [src, 2018 Fig. 14]
TAN = np.tan(np.radians(CRACK_DEG))
X_CRACK_TOP = X_CORNER + H_NIB / TAN
DIAG_TOP = 90.0            # the inclined bars are at 45 deg: HDiagn = VDiagn
X_DIAG0 = X_LAP - (Y_BOT - DIAG_TOP)
XBREAK_A, XBREAK_B = 1200.0, 800.0

# ----------------------------------------------------------- c: wall pier
# Bimschas 2010 Tab. 5.1, 5.3 and 5.7, as carried in bimschas_theta.py
L_V, L_W, B_W = 3300.0, 1500.0, 350.0
N_BASE = 1370.0            # kN, constant axial compression
D_BAR_W = 14.0
COVER_W = 750.0 - 717.0    # 33 mm to the bar center, from Y_END in that file
RHO = {"VK1": 0.82, "VK3": 1.23}
LAYOUT = {
    "VK1": [(717.0, 4), (-717.0, 4)]
           + [(s * y, 2) for y in (585., 455., 325., 195., 65.) for s in (1, -1)],
    "VK3": [(717.0, 4), (-717.0, 4)]
           + [(s * y, 2) for y in (640., 560., 480., 400., 320., 240., 160., 80.)
              for s in (1, -1)] + [(0.0, 2)],
}
BUNDLE_DX = 62.0           # drawing offset for the second bar of an end group

# -------------------------------------------------------- d: tension prism
# Davis 2015, prism block in the header of data/davis2015_tables.csv
PR_L, PR_H = 900.0, 100.0
PR_STUB = 100.0            # bar protruding each end
PR_FIBER = 800.0           # fiber bonded over the central length
PR_AVG = 700.0             # strain averaged over the middle length
PR_AS = 200.0              # 15M bar, nominal area

# --------------------------------------------------------- e: corroded beam
GEO17 = json.loads((DATA / "davis2017_geometry.json").read_text())
G17 = GEO17["geometry_mm"]
R17 = GEO17["reinforcement"]

# ---------------------------------------------------- f: instrumented wall


def barforces():
    raw = (DATA / "desnerck_fig16_barforces.csv").read_text().strip().split("\n")
    cols = raw[0].split(",")[1:]
    a = np.array([[np.nan if x == "" else float(x) for x in ln.split(",")[1:]]
                  for ln in raw[1:] if ln.startswith("NS-REF")])
    i = int(np.nanargmax(a[:, 0]))
    pk = {c: float(a[i, k]) for k, c in enumerate(cols)}
    pk["datum"] = float(np.nanmin(a[:, 0]))
    return pk


def table():
    out = {}
    for ln in (DATA / "desnerck_tables.csv").read_text().strip().split("\n")[1:]:
        q, _s, v, _u, _src = ln.split(",")
        out[q] = v
    return out


TAB = table()
MILL_LEN = float(TAB["milled_zone_length"])          # 100 mm  [src]
MILL_FRAC = float(TAB["milled_area_fraction"])       # 0.50    [src]
X_MILL0 = STIRRUPS[0] - 0.5 * MILL_LEN
X_MILL1 = STIRRUPS[0] + 0.5 * MILL_LEN
PK = barforces()
R_REQ = FAIL["NS-REF"] - PK["datum"]
S_MEAS = PK["VStack"]
V_CONC = R_REQ - S_MEAS
SHARE = 100.0 * S_MEAS / R_REQ


def crack(x):
    return -(H_NIB - (np.asarray(x, float) - X_CORNER) * TAN)


def cross_diag():
    x = (H_NIB + X_CORNER * TAN + X_DIAG0 - DIAG_TOP) / (1.0 + TAN)
    return x, -(DIAG_TOP + (x - X_DIAG0))


# ------------------------------------------------------------- draughting
def place(fig, ax, col, row, xlo, ylo, yhi):
    """Aspect-equal panel filling its grid cell exactly."""
    fw, fh = fig.get_size_inches()
    span = (yhi - ylo) * COL_W[col] * fw / (ROW_H[row] * fh)
    ax.set_xlim(xlo, xlo + span)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_position((COL_X[col], ROW_Y[row], COL_W[col], ROW_H[row]))
    return xlo + span


def zig(x, y0, y1, n=11, amp=None):
    """Vertical break line, so a truncated member reads as truncated."""
    amp = 0.014 * abs(y1 - y0) if amp is None else amp
    t = np.linspace(0.0, 1.0, n)
    return x + amp * np.where(np.arange(n) % 2 == 0, -1.0, 1.0), y0 + (y1 - y0) * t


def hzig(y, x0, x1, n=17, amp=None):
    """Horizontal break line."""
    amp = 0.020 * abs(x1 - x0) if amp is None else amp
    t = np.linspace(0.0, 1.0, n)
    return x0 + (x1 - x0) * t, y + amp * np.where(np.arange(n) % 2 == 0, -1.0, 1.0)


def dim(ax, p0, p1, text, off, rot=0, col=C_DIM, fs=FS, ha="center",
        va="center"):
    ax.annotate("", xy=p1, xytext=p0, annotation_clip=False,
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=col,
                                shrinkA=0, shrinkB=0, mutation_scale=5.5))
    mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
    if rot:
        mx += off
    else:
        my += off
    ax.text(mx, my, text, fontsize=fs, color=col, ha=ha, va=va, rotation=rot,
            rotation_mode="anchor", clip_on=False,
            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))


def wit(ax, x0, y0, x1, y1, col="0.6"):
    ax.plot([x0, x1], [y0, y1], color=col, lw=0.5, clip_on=False, zorder=2)


def lead(ax, xy, xytext, col="0.45", rad=0.0):
    ax.annotate("", xy=xy, xytext=xytext, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.65, color=col,
                                shrinkA=1.2, shrinkB=1.2,
                                connectionstyle=f"arc3,rad={rad}"))


def txt(ax, x, y, s, col=C_TXT, fs=FS, ha="left", va="center", box=True, **kw):
    bb = dict(fc="white", ec="none", pad=1.2, alpha=0.88) if box else None
    kw.setdefault("zorder", 12)          # labels mask the drawing, never under it
    return ax.text(x, y, s, fontsize=fs, color=col, ha=ha, va=va, bbox=bb,
                   clip_on=False, linespacing=1.28, **kw)


def known(ax, x, y, s, dx, col="0.20"):
    """The line every panel carries: the independently known quantity."""
    ax.plot([x], [y], marker="s", ms=3.4, color=C_KNOWN, clip_on=False,
            zorder=5)
    txt(ax, x + dx, y, s, col=col, box=False)


def source(ax, x, y, s):
    txt(ax, x, y, s, col=C_SRC, ha="right", box=False)


def ubar_path():
    """A U-bar in elevation: the nib tie, returned round the end face."""
    r, xb, xv = 30.0, 68.0, 38.0
    t1 = np.radians(np.linspace(90.0, 180.0, 20))
    t2 = np.radians(np.linspace(180.0, 270.0, 20))
    x = np.concatenate(([X_LAP, xb], xb + r * np.cos(t1), [xv, xv],
                        xb + r * np.cos(t2), [xb, X_LAP]))
    d = np.concatenate(([Y_TOP, Y_TOP], Y_TOP + r - r * np.sin(t1),
                        [Y_TOP + r, Y_UBAR - r],
                        Y_UBAR - r - r * np.sin(t2), [Y_UBAR, Y_UBAR]))
    return x, -d


# ------------------------------------------------------------------ panel a
def panel_a(ax, xhi):
    xz, yz = zig(XBREAK_A, 0.0, -H)
    ax.add_patch(Polygon([(0.0, 0.0)] + list(zip(xz, yz)) +
                         [(X_CORNER, -H), (X_CORNER, -H_NIB), (0.0, -H_NIB)],
                         closed=True, fc=C_CONC, ec="black", lw=1.0, zorder=1))

    mill = [(X_MILL0, 0.0), (X_MILL1, 0.0), (X_MILL1, -H), (X_CORNER, -H),
            (X_CORNER, -H_NIB), (X_MILL0, -H_NIB)]
    ax.add_patch(Polygon(mill, closed=True, fc=C_KNOWN, alpha=0.42, ec="none",
                         zorder=2))
    ax.add_patch(Polygon(mill, closed=True, fill=False, ec="0.25", lw=0.8,
                         ls=(0, (3.5, 2)), zorder=9))

    ax.add_patch(Rectangle((X_BEARING - 70, -H_NIB - 30), 140, 30, fc="0.40",
                           ec="none", zorder=5))
    ax.add_patch(Circle((X_BEARING, -H_NIB - 75), 45, fc="0.80", ec="0.35",
                        lw=0.8, zorder=5))
    ax.add_patch(Rectangle((X_BEARING - 70, -H_NIB - 150), 140, 30, fc="0.40",
                           ec="none", zorder=5))
    for xh in np.linspace(X_BEARING - 74, X_BEARING + 60, 6):
        ax.plot([xh, xh + 18], [-H_NIB - 150, -H_NIB - 182], color="0.5",
                lw=0.7, zorder=5)

    for x in STIRRUPS:
        if x < XBREAK_A - 40:
            ax.plot([x, x], [-Y_TOP, -Y_BOT], color=C_STEEL, lw=1.1, zorder=6)
    ax.plot([X_CORNER + 28, XBREAK_A - 18], [-Y_BOT] * 2, color=C_STEEL,
            lw=1.8, zorder=6)
    ax.plot([X_LAP, XBREAK_A - 18], [-Y_TOP] * 2, color=C_STEEL, lw=1.8,
            zorder=6)
    for dx in (0.0, 46.0):
        ax.plot([X_DIAG0 + dx, X_LAP + dx], [-DIAG_TOP, -Y_BOT], color=C_STEEL,
                lw=1.7, zorder=7)
    ux, uy = ubar_path()
    ax.plot(ux, uy, color=C_STEEL, lw=1.9, zorder=8)
    ax.add_patch(Circle((X_CORNER, -H_NIB), 20, fill=False, ec="black", lw=1.0,
                        zorder=10))

    # the gauged bars: the open data reports these four forces at the nib
    for gx, gy in ((470.0, -Y_UBAR), (430.0, -(DIAG_TOP + 430.0 - X_DIAG0)),
                   (STIRRUPS[0], -180.0), (STIRRUPS[1], -180.0)):
        ax.plot([gx], [gy], marker="s", ms=3.2, color=C_MEAS, mec="white",
                mew=0.6, zorder=11)
    txt(ax, 560, -115, "bar gauges", col=C_MEAS)
    lead(ax, (STIRRUPS[1] + 8, -178), (552, -118), col=C_MEAS, rad=0.1)

    txt(ax, 452, -255, "U-bars", col=C_STEEL)
    txt(ax, 420, -585, "diagonals", col=C_STEEL)
    txt(ax, 880, -300, "stirrups", col=C_STEEL)
    txt(ax, 250, -775, "re-entrant corner", col="0.15")
    lead(ax, (X_CORNER + 6, -H_NIB - 10), (250, -762), rad=-0.25)
    txt(ax, -228, -450, "bearing", col=C_NOTE)
    lead(ax, (28, -400), (-30, -450))
    txt(ax, 430, 92, "milled zone", col="0.20")
    lead(ax, (0.5 * (X_MILL0 + X_MILL1), 18), (422, 88), rad=0.1)

    dim(ax, (-78, 0.0), (-78, -H_NIB), f"{H_NIB:.0f}", off=-16, rot=90,
        va="bottom")
    dim(ax, (1268, 0.0), (1268, -H), f"{H:.0f}", off=16, rot=90, va="top")
    wit(ax, -92, 0.0, 0.0, 0.0)
    wit(ax, -92, -H_NIB, 0.0, -H_NIB)
    wit(ax, 1254, 0.0, XBREAK_A, 0.0)
    wit(ax, 1254, -H, XBREAK_A, -H)
    dim(ax, (0.0, -560), (X_BEARING, -560), f"{X_BEARING:.0f}", off=-42,
        va="top")
    wit(ax, 0.0, -H_NIB, 0.0, -574)
    wit(ax, X_BEARING, -H_NIB - 182, X_BEARING, -574)
    known(ax, -222, -900, f"nib bars milled to {100 * MILL_FRAC:.0f} % of "
          f"area over {MILL_LEN:.0f} mm", 62)


# ------------------------------------------------------------------ panel b
def panel_b(ax, xhi):
    xz, yz = zig(XBREAK_B, 0.0, -H)
    ax.add_patch(Polygon([(float(X_CRACK_TOP), 0.0)] + list(zip(xz, yz)) +
                         [(X_CORNER, -H), (X_CORNER, -H_NIB)], closed=True,
                         fc="0.975", ec="black", lw=1.3, ls=(0, (4.5, 2.0)),
                         zorder=1))
    ax.add_patch(Polygon([(0.0, 0.0), (float(X_CRACK_TOP), 0.0),
                          (X_CORNER, -H_NIB), (0.0, -H_NIB)], closed=True,
                         fc=C_CONC, ec="black", lw=1.1, zorder=2))

    for x in STIRRUPS[:2]:
        ax.plot([x, x], [-Y_TOP, -Y_BOT], color=C_STEEL, lw=1.1, zorder=4)
    ax.plot([X_DIAG0, X_LAP], [-DIAG_TOP, -Y_BOT], color=C_STEEL, lw=1.7,
            zorder=4)
    ux, uy = ubar_path()
    ax.plot(ux, uy, color=C_STEEL, lw=1.8, zorder=4)
    ax.plot([X_CORNER, X_CRACK_TOP], [-H_NIB, 0.0], color="black", lw=1.7,
            ls=(0, (4.5, 2.0)), zorder=8)
    txt(ax, 500, 60, f"crack, {CRACK_DEG:.0f}°", col="0.15", box=False)
    lead(ax, (392.0, -100.0), (492, 56), rad=-0.12)

    xd, yd = cross_diag()
    xu = float(X_CORNER + (H_NIB - Y_UBAR) / TAN)
    for x, y in ((STIRRUPS[1], float(crack(STIRRUPS[1]))),
                 (STIRRUPS[0], float(crack(STIRRUPS[0]))),
                 (xd, yd)):
        ax.plot([x], [y], marker="o", ms=4.6, mfc="white", mec=C_MEAS, mew=1.4,
                zorder=10)
    txt(ax, 545, -120, "each ring is a bar\ncrossing the crack", col=C_MEAS)
    lead(ax, (STIRRUPS[1] + 6, float(crack(STIRRUPS[1]))), (538, -120),
         col=C_MEAS, rad=-0.12)
    ax.plot([xu], [-Y_UBAR], marker="o", ms=4.6, mfc="white", mec="0.5",
            mew=1.4, zorder=10)
    txt(ax, 545, -400, "U-bars carry no shear", col="0.35")
    lead(ax, (xu, -Y_UBAR), (538, -400), col="0.55", rad=-0.12)

    ax.annotate("", xy=(X_BEARING, -H_NIB - 16),
                xytext=(X_BEARING, -H_NIB - 150),
                arrowprops=dict(arrowstyle="-|>", lw=1.7, color="black",
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    txt(ax, X_BEARING + 26, -H_NIB - 75, "R", box=False)

    # required against measured, at NS-REF failure, to one force scale
    s = 1950.0 / R_REQ
    y0, hgt = -900.0, 92.0
    x = 0.0
    for v, nm, al in ((PK["VDiagn"], "VDiagn", 0.95), (PK["VSt1"], "VSt1", 0.72),
                      (PK["VSt2"], "VSt2", 0.52)):
        ax.add_patch(Rectangle((x, y0), v * s, hgt, fc=C_MEAS, alpha=al,
                               ec="white", lw=0.7, zorder=3))
        txt(ax, x + 0.5 * v * s, y0 + 0.5 * hgt, f"{nm} {v:.1f}",
            col="white" if al > 0.6 else C_TXT, ha="center", box=False)
        x += v * s
    ax.add_patch(Rectangle((x, y0), V_CONC * s, hgt, fc="0.88", ec="0.45",
                           lw=0.7, hatch="////", zorder=3))
    ax.plot([R_REQ * s] * 2, [y0 - 26, y0 + hgt + 26], color="black", lw=1.2,
            zorder=6)
    txt(ax, R_REQ * s, y0 + hgt + 34, f"R required, {R_REQ:.0f} kN",
        ha="right", va="bottom", box=False)
    txt(ax, 0, y0 - 56, f"bars supply {S_MEAS:.0f} kN, {SHARE:.0f} % of R",
        va="top", box=False)

    ax.text(0, -1120, r"$(1-\theta)\,S = R$", fontsize=F.FS_TITLE,
            color="black", ha="left", va="center", clip_on=False)


# ------------------------------------------------------------------ panel c
def panel_c(ax, xhi):
    y_brk, y_stub = 1400.0, 1700.0
    y_top = 2000.0
    ax.add_patch(Rectangle((-260, -300), L_W + 520, 300, fc="0.82", ec="0.45",
                           lw=0.8, zorder=1))
    for xh in np.linspace(-240, L_W + 170, 8):
        ax.plot([xh, xh + 110], [-300, -450], color="0.55", lw=0.7, zorder=1)
    bx, by = hzig(y_brk, 0.0, L_W)
    ax.add_patch(Polygon([(0.0, 0.0), (L_W, 0.0)] + list(zip(bx[::-1], by[::-1])),
                         closed=True, fc=C_CONC, ec="black", lw=1.0, zorder=2))
    sx, sy = hzig(y_stub, 0.0, L_W)
    ax.add_patch(Polygon(list(zip(sx, sy)) + [(L_W, y_top), (0.0, y_top)],
                         closed=True, fc=C_CONC, ec="black", lw=1.0, zorder=2))
    ax.plot([0, L_W], [0, 0], color="black", lw=1.4, ls=(0, (4.5, 2.0)),
            zorder=6)

    ax.annotate("", xy=(0, 1870), xytext=(-1100, 1870),
                arrowprops=dict(arrowstyle="-|>", lw=1.7, color="black",
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    txt(ax, -1090, 2110, "F", box=False)
    ax.annotate("", xy=(0.5 * L_W, y_top + 30), xytext=(0.5 * L_W, y_top + 420),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="0.35",
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    txt(ax, 880, 2240, f"N = {N_BASE:.0f} kN", col=C_NOTE, box=False)
    ax.annotate("", xy=(L_W + 560, 1870), xytext=(L_W + 80, 1870),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_MEAS,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    txt(ax, 2160, 1870, "top displacement", col=C_MEAS, box=False)
    # clear of the footing block: the strip right of it and below the
    # section insets is the only empty ground in this panel
    txt(ax, 1900, -250, "cut at the base", col="0.15", box=False)
    lead(ax, (L_W - 120, -12), (1880, -250), rad=0.16)

    # the effective height, dimensioned across the break
    xd = -700.0
    ax.annotate("", xy=(xd, y_brk), xytext=(xd, 0.0),
                arrowprops=dict(arrowstyle="<|-", lw=0.7, color=C_DIM,
                                shrinkA=0, shrinkB=0, mutation_scale=5.5))
    ax.annotate("", xy=(xd, y_top), xytext=(xd, y_stub),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C_DIM,
                                shrinkA=0, shrinkB=0, mutation_scale=5.5))
    for yb in (y_brk + 60, y_stub - 60):
        ax.plot([xd - 55, xd + 55], [yb - 55, yb + 55], color=C_DIM, lw=0.7)
    ax.text(xd - 60, 700.0, f"{L_V:.0f}", fontsize=FS, color=C_DIM,
            ha="center", va="bottom", rotation=90, rotation_mode="anchor")
    wit(ax, xd - 90, 0.0, 0.0, 0.0)
    wit(ax, xd - 90, y_top, 0.0, y_top)

    for nm, ybase, ylab in (("VK1", 950.0, 1350.0), ("VK3", 300.0, 670.0)):
        x0 = 2300.0
        ax.add_patch(Rectangle((x0, ybase), L_W, B_W, fc=C_CONC, ec="black",
                               lw=0.9, zorder=2))
        for pos, n in LAYOUT[nm]:
            for k in range(n // 2):
                px = x0 + 0.5 * L_W + pos - (0.0 if k == 0 else
                                             np.sign(pos or 1.0) * BUNDLE_DX)
                for py in (ybase + COVER_W, ybase + B_W - COVER_W):
                    ax.plot([px], [py], marker="o", ms=2.0, color=C_STEEL,
                            mec="none", zorder=5)
        txt(ax, x0, ylab, f"{nm}, {sum(n for _p, n in LAYOUT[nm])} bars",
            col="0.20", va="bottom", box=False)
    txt(ax, 2300.0, 60.0, f"{L_W:.0f} × {B_W:.0f} mm", col=C_DIM, box=False)
    known(ax, -1180, -830, f"28 bars at {RHO['VK1']:.2f} % against 42 "
          f"at {RHO['VK3']:.2f} %,\na 33.3 % uniform section loss", 175)


# ------------------------------------------------------------------ panel d
def panel_d(ax, xhi):
    ax.add_patch(Rectangle((0, -0.5 * PR_H), PR_L, PR_H, fc=C_CONC, ec="black",
                           lw=1.0, zorder=2))
    ax.add_patch(Rectangle((0.5 * (PR_L - PR_AVG), -0.5 * PR_H), PR_AVG, PR_H,
                           fc=C_MEAS, alpha=0.17, ec="none", zorder=3))
    ax.plot([-PR_STUB, PR_L + PR_STUB], [0, 0], color=C_STEEL, lw=2.2,
            zorder=5)
    ax.plot([0.5 * (PR_L - PR_FIBER), 0.5 * (PR_L + PR_FIBER)], [26, 26],
            color=C_MEAS, lw=1.8, zorder=6)
    for x0, x1 in ((-PR_STUB - 30, -PR_STUB - 170),
                   (PR_L + PR_STUB + 30, PR_L + PR_STUB + 170)):
        ax.annotate("", xy=(x1, 0), xytext=(x0, 0), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", lw=1.7, color="black",
                                    shrinkA=0, shrinkB=0, mutation_scale=9))
    txt(ax, -PR_STUB - 100, 68, "N", ha="center", box=False)
    txt(ax, PR_L + PR_STUB + 100, 68, "N", ha="center", box=False)

    txt(ax, 0.5 * PR_L, 115, f"fiber bonded over {PR_FIBER:.0f} mm",
        col=C_MEAS, ha="center", box=False)
    txt(ax, 0.5 * PR_L, -118, f"strain averaged over {PR_AVG:.0f} mm",
        col=C_MEAS, ha="center", va="top", box=False)

    dim(ax, (0, -238), (PR_L, -238), f"{PR_L:.0f}", off=-30, va="top")
    wit(ax, 0, -0.5 * PR_H, 0, -252)
    wit(ax, PR_L, -0.5 * PR_H, PR_L, -252)

    txt(ax, 0, 250, "the free body carries no moment:\n"
        "the identifying condition is exact", col=C_NOTE, box=False)
    txt(ax, 0, -440, f"single 15M bar, {PR_AS:.0f} mm²\n"
        f"{PR_H:.0f} × {PR_H:.0f} mm section", col=C_STEEL, box=False)
    known(ax, -360, -580, "weighed mass loss of the extracted bar", 46)


# ------------------------------------------------------------------ panel e
def panel_e(ax, xhi):
    ln, ht = float(G17["overall_length"]), float(G17["height"])
    span, sh = float(G17["clear_span"]), float(G17["shear_span"])
    x_sup = float(G17["support_center_from_beam_end"])
    plate, thk = float(G17["load_plate_width"]), float(G17["plate_thickness"])
    d_eff, d_top = float(G17["effective_depth_d"]), float(G17["top_bar_depth"])
    xm = 0.5 * ln

    ax.add_patch(Rectangle((0, 0), ln, ht, fc=C_CONC, ec="black", lw=1.0,
                           zorder=2))
    for xc in (x_sup, ln - x_sup):
        ax.add_patch(Rectangle((xc - 0.5 * plate, -thk), plate, thk, fc="0.40",
                               ec="none", zorder=3))
        ax.add_patch(Polygon([(xc, -thk), (xc - 95, -thk - 150),
                              (xc + 95, -thk - 150)], closed=True, fc="0.82",
                             ec="0.4", lw=0.8, zorder=3))
    ax.add_patch(Rectangle((xm - 0.5 * plate, ht), plate, thk, fc="0.40",
                           ec="none", zorder=3))
    ax.annotate("", xy=(xm, ht + thk + 20), xytext=(xm, ht + 330),
                arrowprops=dict(arrowstyle="-|>", lw=1.7, color="black",
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    txt(ax, xm + 70, ht + 230, "P", box=False)

    n_st = int(R17["stirrups"]["n"])
    s_st = float(R17["stirrups"]["spacing_mm"])
    x0 = 0.5 * (ln - (n_st - 1) * s_st)
    for k in range(n_st):
        ax.plot([x0 + k * s_st] * 2, [ht - d_top, ht - d_eff + 30],
                color=C_STEEL, lw=0.9, zorder=5)
    ax.plot([34, ln - 34], [ht - d_eff] * 2, color=C_STEEL, lw=2.1, zorder=6)
    ax.plot([34, ln - 34], [ht - d_top] * 2, color=C_STEEL, lw=1.4, zorder=6)
    ax.plot([x_sup, ln - x_sup], [ht - d_eff + 22] * 2, color=C_MEAS, lw=1.5,
            zorder=7)
    txt(ax, xm, ht - d_eff - 110, "fiber bonded to the bars", col=C_MEAS,
        ha="center", va="top")
    txt(ax, ln + 70, ht - d_eff + 60,
        f"2 bars,\n{R17['bottom']['designation']}", col=C_STEEL, va="center",
        box=False)

    dim(ax, (x_sup, -330), (ln - x_sup, -330), f"{span:.0f}", off=-72,
        va="top")
    dim(ax, (x_sup, -580), (xm, -580), f"{sh:.0f}", off=-58, va="top")
    dim(ax, (-140, 0), (-140, ht), f"{ht:.0f}", off=-30, rot=90, va="bottom")
    wit(ax, x_sup, -thk - 150, x_sup, -620)
    wit(ax, ln - x_sup, -thk - 150, ln - x_sup, -350)
    wit(ax, xm, ht + thk, xm, -620)
    wit(ax, -160, 0, 0, 0)
    wit(ax, -160, ht, 0, ht)
    known(ax, -300, -840, "weighed mass loss, with the bond degraded", 90)
    txt(ax, 1620, -840, "negative control: the recovery is displaced",
        col=C_STEEL, box=False)


# ---------------------------------------------------------------- assembly
SPEC = [
    ("a", 0, 0, -235.0, -975.0, 190.0, panel_a,
     "dapped end, Desnerck et al. 2017"),
    ("b", 1, 0, -185.0, -1190.0, 190.0, panel_b,
     "free body of the nib"),
    ("c", 0, 1, -1300.0, -1150.0, 2550.0, panel_c,
     "wall pier, Bimschas 2010"),
    ("d", 1, 1, -420.0, -620.0, 460.0, panel_d,
     "tension prism, Davis 2015"),
    ("e", 2, 2, -638.0, -920.0, 660.0, panel_e,
     "corroded beam, Davis et al. 2017"),
]


def plate_key(fig):
    """One key for the whole plate, so no panel spends space on the code."""
    h = [Line2D([], [], color=C_STEEL, lw=2.4, label="reinforcement"),
         Line2D([], [], color=C_MEAS, lw=2.4, label="instrumentation"),
         Line2D([], [], ls="none", marker="s", ms=4.6, color=C_KNOWN,
                label="independently known")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.002),
               ncol=3, frameon=False, fontsize=FS, handlelength=1.7,
               handletextpad=0.5, columnspacing=2.6)


def main() -> None:
    fig = plt.figure(figsize=(F.FIG_W, FIG_H))
    axes = {}
    for key, col, row, xlo, ylo, yhi, fn, _t in SPEC:
        ax = fig.add_axes((COL_X[col], ROW_Y[row], COL_W[col], ROW_H[row]))
        axes[key] = ax
        fn(ax, place(fig, ax, col, row, xlo, ylo, yhi))

    for key, _c, row, _xl, _yl, _yh, _fn, title in SPEC:
        F.fig_panel(fig, axes[key], key, title,
                    y=ROW_Y[row] + ROW_H[row] + 0.010)

    plate_key(fig)
    F.save(fig, FIG / "specimens.png")
    plt.close(fig)

    print(f"  dapped end: crack meets the top fiber at x = {X_CRACK_TOP:.1f} "
          f"mm; milled zone {X_MILL0:.0f} to {X_MILL1:.0f} mm")
    print(f"  NS-REF at failure: R {R_REQ:.1f} kN required, measured "
          f"VDiagn {PK['VDiagn']:.1f} + VSt1 {PK['VSt1']:.1f} + VSt2 "
          f"{PK['VSt2']:.1f} = {S_MEAS:.1f} kN ({SHARE:.1f} % of R), "
          f"concrete residual {V_CONC:.1f} kN")
    for nm in ("VK1", "VK3"):
        print(f"  {nm}: {sum(n for _p, n in LAYOUT[nm])} bars of "
              f"{D_BAR_W:.0f} mm, rho_sl {RHO[nm]:.2f} %")
    print(f"  prism {PR_L:.0f} x {PR_H:.0f} mm, fiber {PR_FIBER:.0f} mm, "
          f"averaged {PR_AVG:.0f} mm; beam span {G17['clear_span']} mm, "
          f"a/d {G17['shear_span_to_depth_ratio']}")


if __name__ == "__main__":
    main()
