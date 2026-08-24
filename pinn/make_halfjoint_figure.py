# -*- coding: utf-8 -*-
"""The tested half-joint: the specimen, and the free body cut from it.

The validation section identifies a section loss on the Cambridge
half-joints of Desnerck, Lees and Morley, and that is the strongest
measured result in the study, yet the reader is never shown the specimen.
Panel a draws it: the full-depth beam stepping down to the nib, the
bearing, the re-entrant corner, the three nib reinforcement groups the
text names, and the 100 mm zone in which the nib bars of NS-LR were milled
to half their area.  Panel b draws the free body the identification
actually uses, states the identifying condition on it, and compares the
tie force that free body requires against the tie force the bar gauges
measured on the reference specimen at failure.

Provenance.  Every dimension comes from halfjoint_geometry.py, every
failure load from halfjoint_identify.FAIL, every bar force from
data/desnerck_fig16_barforces.csv, and the milling from
data/desnerck_tables.csv.  Nothing is refitted here and no dimension is
invented: quantities the sources print are tagged [src] in
halfjoint_geometry.py, quantities read off a drawing are tagged [dig] and
are drawn but never labeled with a number.

Run:  /usr/local/bin/python3.12 make_halfjoint_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import figstyle as F                                                      # noqa: E402
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle                 # noqa: E402
from halfjoint_geometry import (H, H_NIB, L_HALF, NODE, STIRRUPS,         # noqa: E402
                                X_BEARING, X_CORNER)
from halfjoint_identify import FAIL                                       # noqa: E402

DATA = HERE.parent / "data"
FIG = HERE.parent / "figures"
F.apply()

# --------------------------------------------------------------- sources
# the nib steel, from the halfjoint_geometry docstring [src, 2018 Sec. 4.1]
N_UBAR, D_UBAR = 3, 12
N_DIAG, D_DIAG = 4, 12
D_STIRRUP = 10

# levels read off the digitised strut-and-tie nodes [dig]
Y_UBAR = NODE["A"][1]          # 298, the horizontal nib tie
Y_TOP = 50.0                   # top chord, the level the U-bars return to
Y_BOT = NODE["F"][1]           # 661, bottom longitudinal bars
X_LAP = NODE["G"][0]           # 649, where the diagonal meets the bottom bars
CRACK_DEG = 60.0               # [src, 2018 Fig. 14]

# the drawing colors, one per named reinforcement group
C_UBAR, C_DIAG, C_STIR = F.SKY, F.GREEN, F.VERM
C_OTHER = "0.62"
C_CONC = "0.93"
C_MILL = F.ORANGE

XMAX_A = 1250.0                # panel a, where the beam is broken off
XMAX_B = 800.0                 # panel b, where the remainder is broken off


def table() -> dict:
    out = {}
    for line in (DATA / "desnerck_tables.csv").read_text().strip().split("\n")[1:]:
        q, _spec, val, _unit, _src = line.split(",")
        out[q] = val
    return out


def peak_bar_forces() -> dict:
    """NS-REF bar forces at the peak of the open-data record (2016 Fig. 16)."""
    raw = (DATA / "desnerck_fig16_barforces.csv").read_text().strip().split("\n")
    cols = raw[0].split(",")[1:]
    a = np.array([[np.nan if x == "" else float(x) for x in ln.split(",")[1:]]
                  for ln in raw[1:] if ln.startswith("NS-REF")])
    i = int(np.nanargmax(a[:, 0]))
    pk = {c: float(a[i, k]) for k, c in enumerate(cols)}
    pk["datum"] = float(np.nanmin(a[:, 0]))
    return pk


TAB = table()
MILL_LEN = float(TAB["milled_zone_length"])        # 100 mm  [src]
MILL_FRAC = float(TAB["milled_area_fraction"])     # 0.50    [src]
PK = peak_bar_forces()
R_REQ = FAIL["NS-REF"] - PK["datum"]               # 393.2 kN
S_MEAS = PK["VStack"]                              # 346.7 kN
V_CONC = R_REQ - S_MEAS                            # 46.5 kN
SHARE = 100.0 * S_MEAS / R_REQ                     # 88 %

TAN = np.tan(np.radians(CRACK_DEG))
X_CRACK_TOP = X_CORNER + H_NIB / TAN               # crack meets the top fiber
X_MILL0 = STIRRUPS[0] - 0.5 * MILL_LEN             # 100 mm zone about stirrup 1
X_MILL1 = STIRRUPS[0] + 0.5 * MILL_LEN
# the inclined bars are at 45 deg: the open data reports HDiagn = VDiagn
DIAG_TOP = 90.0                                    # depth at which they start
X_DIAG0 = X_LAP - (Y_BOT - DIAG_TOP)


def crack(x):
    """Depth of the corner crack at `x`, as a plotted (negative) ordinate."""
    return -(H_NIB - (np.asarray(x, float) - X_CORNER) * TAN)


def diag(x):
    """Plotted ordinate of the inclined bars."""
    return -(DIAG_TOP + (np.asarray(x, float) - X_DIAG0))


def cross_diag_crack():
    x = (H_NIB + X_CORNER * TAN + X_DIAG0 - DIAG_TOP) / (1.0 + TAN)
    return x, float(diag(x))


# ------------------------------------------------------------- draughting
def zig(x, y0, y1, n=13, amp=11.0):
    """A break line, so a truncated member does not read as a real end."""
    t = np.linspace(0.0, 1.0, n)
    return x + amp * np.where(np.arange(n) % 2 == 0, -1.0, 1.0), y0 + (y1 - y0) * t


def dim(ax, p0, p1, text, off=0.0, fs=F.FS_SMALL, col="0.35", rot=0,
        ha="center", va="top", pad=1.0):
    """Dimension line with bar terminators and its own label."""
    ax.annotate("", xy=p1, xytext=p0, annotation_clip=False,
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=col,
                                shrinkA=0, shrinkB=0, mutation_scale=6.5))
    mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
    if rot:
        mx += off
    else:
        my += off
    ax.text(mx, my, text, fontsize=fs, color=col, ha=ha, va=va,
            rotation=rot, rotation_mode="anchor", clip_on=False,
            bbox=dict(fc="white", ec="none", pad=pad, alpha=0.9))


def witness(ax, x, y0, y1, col="0.55"):
    ax.plot([x, x], [y0, y1], color=col, lw=0.5, clip_on=False, zorder=2)


def lead(ax, xy, xytext, col="0.45", rad=0.0):
    ax.annotate("", xy=xy, xytext=xytext, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=col,
                                shrinkA=1.5, shrinkB=1.5,
                                connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, col="0.15", fs=F.FS_ANNOT, ha="left", va="center",
          box=True, **kw):
    bb = dict(fc="white", ec="none", pad=1.6, alpha=0.88) if box else None
    return ax.text(x, y, text, fontsize=fs, color=col, ha=ha, va=va,
                   clip_on=False, bbox=bb, linespacing=1.32, **kw)


def ubar_path():
    """A U-bar in elevation: the nib tie, returned round the end face."""
    r, xb, xv = 30.0, 68.0, 38.0
    t1 = np.radians(np.linspace(90.0, 180.0, 24))
    t2 = np.radians(np.linspace(180.0, 270.0, 24))
    x = np.concatenate(([X_LAP, xb], xb + r * np.cos(t1), [xv, xv],
                        xb + r * np.cos(t2), [xb, X_LAP]))
    d = np.concatenate(([Y_TOP, Y_TOP], Y_TOP + r - r * np.sin(t1),
                        [Y_TOP + r, Y_UBAR - r],
                        Y_UBAR - r - r * np.sin(t2), [Y_UBAR, Y_UBAR]))
    return x, -d


def stirrup(ax, x, col=C_STIR, lw=1.5, z=6, alpha=1.0):
    ax.plot([x, x], [-Y_TOP, -Y_BOT], color=col, lw=lw, zorder=z,
            solid_capstyle="butt", alpha=alpha)


# ------------------------------------------------------------------ panel a
def panel_a(ax):
    ax.set_axis_off()

    xz, yz = zig(XMAX_A, 0.0, -H)
    outline = ([(0.0, 0.0)] + list(zip(xz, yz)) +
               [(X_CORNER, -H), (X_CORNER, -H_NIB), (0.0, -H_NIB)])
    ax.add_patch(Polygon(outline, closed=True, fc=C_CONC, ec="black",
                         lw=1.1, zorder=1, joinstyle="miter"))

    # the 100 mm zone milled on NS-LR, clipped to the section that exists
    mill = [(X_MILL0, 0.0), (X_MILL1, 0.0), (X_MILL1, -H), (X_CORNER, -H),
            (X_CORNER, -H_NIB), (X_MILL0, -H_NIB)]
    ax.add_patch(Polygon(mill, closed=True, fc=C_MILL, alpha=0.30, ec="none",
                         zorder=2))
    ax.add_patch(Polygon(mill, closed=True, fill=False, ec="0.25", lw=0.9,
                         ls=(0, (4, 2)), zorder=9))

    # bearing: roller on plates, under the nib soffit
    ax.add_patch(Rectangle((X_BEARING - 70, -H_NIB - 30), 140, 30, fc="0.40",
                           ec="none", zorder=5))
    ax.add_patch(Circle((X_BEARING, -H_NIB - 75), 45, fc="0.78", ec="0.30",
                        lw=0.9, zorder=5))
    ax.add_patch(Rectangle((X_BEARING - 70, -H_NIB - 150), 140, 30, fc="0.40",
                           ec="none", zorder=5))
    ax.plot([X_BEARING - 78, X_BEARING + 78], [-H_NIB - 150] * 2, color="0.30",
            lw=1.0, zorder=5)
    for xh in np.linspace(X_BEARING - 74, X_BEARING + 62, 7):
        ax.plot([xh, xh + 16], [-H_NIB - 150, -H_NIB - 178], color="0.45",
                lw=0.8, zorder=5)

    # reinforcement
    for x in STIRRUPS:
        if x < XMAX_A - 40:
            stirrup(ax, x)
    ax.plot([X_CORNER + 28, XMAX_A - 20], [-Y_BOT] * 2, color=C_OTHER, lw=2.0,
            zorder=4, solid_capstyle="butt")
    ax.plot([X_LAP, XMAX_A - 20], [-Y_TOP] * 2, color=C_OTHER, lw=2.0,
            zorder=4, solid_capstyle="butt")
    for dx in (0.0, 46.0):
        ax.plot([X_DIAG0 + dx, X_LAP + dx], [-DIAG_TOP, -Y_BOT], color=C_DIAG,
                lw=1.9, zorder=7, solid_capstyle="round")
    ux, uy = ubar_path()
    ax.plot(ux, uy, color=C_UBAR, lw=2.1, zorder=8, solid_joinstyle="round")

    # the re-entrant corner
    ax.add_patch(Circle((X_CORNER, -H_NIB), 21, fill=False, ec="black", lw=1.1,
                        zorder=10))
    label(ax, 372, -430, "re-entrant corner", ha="left")
    lead(ax, (X_CORNER + 20, -H_NIB - 14), (368, -430), rad=-0.15)
    label(ax, X_BEARING, -H_NIB - 210, "bearing", ha="center", va="top",
          col="0.30", fs=F.FS_SMALL, box=False)

    # dimensions: depth, nib depth, and the spacing chain along the soffit
    witness(ax, 0.0, -H_NIB, -H_NIB - 250)
    dim(ax, (-190, 0.0), (-190, -H), f"{H:.0f}", off=-14, rot=90, va="bottom")
    dim(ax, (-70, 0.0), (-70, -H_NIB), f"{H_NIB:.0f}", off=-14, rot=90,
        va="bottom")
    ax.plot([-205, X_CORNER], [0, 0], color="0.55", lw=0.5, zorder=2)
    ax.plot([-205, 0], [-H, -H], color="0.55", lw=0.5, zorder=2)
    ax.plot([-85, 0], [-H_NIB, -H_NIB], color="0.55", lw=0.5, zorder=2)

    chain = [0.0, X_BEARING] + [s for s in STIRRUPS if s < XMAX_A - 40]
    yc = -H - 95
    for x in chain:
        witness(ax, x, -H - 12 if x > X_CORNER else -H_NIB - 190, yc + 16)
    for x0, x1 in zip(chain[:-1], chain[1:]):
        dim(ax, (x0, yc), (x1, yc), f"{x1 - x0:.0f}", off=-26, va="top")
    ax.annotate("", xy=(chain[-1] + 150, yc), xytext=(chain[-1], yc),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35",
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    label(ax, chain[-1] + 168, yc, f"spacing chain totals {L_HALF:.0f} mm\n"
          "at mid-span", col="0.35", fs=F.FS_SMALL, box=False)

    # the imposed section loss, above the specimen
    label(ax, X_MILL1 + 210, 96,
          f"imposed section loss, specimen NS-LR:  the U-bars, the inclined "
          f"diagonals and the first\nstirrup were milled to "
          f"{100 * MILL_FRAC:.0f} % of bar area over a {MILL_LEN:.0f} mm "
          f"zone, drawn shaded",
          ha="left", va="center", box=False)
    lead(ax, (0.5 * (X_MILL0 + X_MILL1), 24), (X_MILL1 + 196, 96), rad=0.12)

    # the reinforcement groups, named in a column clear of the section
    xl = XMAX_A + 105
    groups = [
        (-150.0, C_UBAR, f"U-bars, {N_UBAR} bars of {D_UBAR} mm",
         (600.0, float(-Y_UBAR))),
        (-330.0, C_DIAG,
         f"inclined diagonals,\n{N_DIAG} bars of {D_DIAG} mm at 45°",
         (520.0, float(diag(520.0)))),
        (-540.0, C_STIR, f"vertical stirrups,\ntwo-legged, {D_STIRRUP} mm",
         (STIRRUPS[5], -430.0)),
        (-700.0, C_OTHER, "longitudinal bars", (1120.0, -Y_BOT)),
    ]
    for y, col, txt, tgt in groups:
        ax.plot([xl, xl + 52], [y, y], color=col, lw=2.1, clip_on=False,
                solid_capstyle="butt")
        label(ax, xl + 70, y, txt, ha="left", box=False)
        lead(ax, tgt, (xl - 12, y), rad=0.08)


# ------------------------------------------------------------------ panel b
def panel_b(ax):
    ax.set_axis_off()

    # the discarded remainder, in outline, and the free body itself
    xz, yz = zig(XMAX_B, 0.0, -H)
    rest = ([(float(X_CRACK_TOP), 0.0)] + list(zip(xz, yz)) +
            [(X_CORNER, -H), (X_CORNER, -H_NIB)])
    ax.add_patch(Polygon(rest, closed=True, fc="0.975", ec="0.72", lw=0.9,
                         ls=(0, (4, 2.5)), zorder=1))
    body = [(0.0, 0.0), (float(X_CRACK_TOP), 0.0), (X_CORNER, -H_NIB),
            (0.0, -H_NIB)]
    ax.add_patch(Polygon(body, closed=True, fc=C_CONC, ec="black", lw=1.2,
                         zorder=2))

    # the members the cut crosses, and the crack that cuts them
    for x in STIRRUPS[:2]:
        stirrup(ax, x, z=4)
    ax.plot([X_DIAG0, X_LAP], [-DIAG_TOP, -Y_BOT], color=C_DIAG, lw=1.9,
            zorder=4, solid_capstyle="round")
    ux, uy = ubar_path()
    ax.plot(ux, uy, color=C_UBAR, lw=2.0, zorder=4)
    ax.plot([X_CORNER, X_CRACK_TOP], [-H_NIB, 0.0], color="black", lw=1.9,
            ls=(0, (5, 2.2)), zorder=8)

    xd, yd = cross_diag_crack()
    marks = [
        (xd, yd, C_DIAG, f"VDiagn  {PK['VDiagn']:.1f} kN", -300.0),
        (STIRRUPS[0], float(crack(STIRRUPS[0])), C_STIR,
         f"VSt1  {PK['VSt1']:.1f} kN", -180.0),
        (STIRRUPS[1], float(crack(STIRRUPS[1])), C_STIR,
         f"VSt2  {PK['VSt2']:.1f} kN", -62.0),
    ]
    for x, y, col, txt, ylab in marks:
        ax.plot([x], [y], marker="o", ms=5.2, mfc="white", mec=col, mew=1.6,
                zorder=10, clip_on=False)
        label(ax, 505, ylab, txt, col=col, ha="left")
        lead(ax, (x, y), (498, ylab), col=col, rad=-0.12)
    ax.plot([float(X_CORNER + (H_NIB - Y_UBAR) / TAN)], [-Y_UBAR], marker="o",
            ms=5.2, mfc="white", mec=C_UBAR, mew=1.6, zorder=10)
    label(ax, 505, -430.0, "U-bars cross the cut\nhorizontally, adding no\n"
          "vertical force", col=C_UBAR, ha="left")
    lead(ax, (float(X_CORNER + (H_NIB - Y_UBAR) / TAN), -Y_UBAR), (498, -430.0),
         col=C_UBAR, rad=0.12)
    label(ax, 0.5 * (X_CORNER + X_CRACK_TOP) - 6, -104,
          f"corner crack, {CRACK_DEG:.0f}°", ha="right", va="bottom",
          rotation=-60, rotation_mode="anchor", fs=F.FS_SMALL, col="0.15")

    # the reaction, its position, and the arm the vertical balance omits
    ax.add_patch(Rectangle((X_BEARING - 70, -H_NIB - 30), 140, 30, fc="0.40",
                           ec="none", zorder=5))
    ax.annotate("", xy=(X_BEARING, -H_NIB - 38), xytext=(X_BEARING, -H_NIB - 190),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="black",
                                shrinkA=0, shrinkB=0, mutation_scale=11))
    label(ax, X_BEARING + 22, -H_NIB - 168, f"R = {R_REQ:.1f} kN", ha="left",
          fs=F.FS_ANNOT)
    witness(ax, X_BEARING, -H_NIB - 190, -H_NIB - 268)
    witness(ax, X_CORNER, -H_NIB - 10, -H_NIB - 268)
    dim(ax, (X_BEARING, -H_NIB - 250), (X_CORNER, -H_NIB - 250), "arm of R",
        off=-30, va="top", ha="center")
    label(ax, 0.0, -H_NIB - 330,
          "vertical balance of the nib: the arm, the crack angle\n"
          "and plane sections do not enter it", col="0.30", fs=F.FS_SMALL,
          ha="left", va="top", box=False)

    # the identifying condition, at the head of the panel
    label(ax, 0.0, 176, "identifying condition on this free body",
          col="0.15", ha="left", va="center", box=False)
    ax.text(430, 176, r"$(1-\theta)\,S = T_{\mathrm{req}}$", fontsize=F.FS_TITLE,
            color="black", ha="left", va="center", clip_on=False)
    label(ax, 900, 176, "with  S = VDiagn + VSt1 + VSt2   measured on the bars,"
          "   and   R  required by statics", ha="left", va="center", box=False)

    # required against measured, drawn to a common force scale
    s, base, wd = 1.35, -760.0, 230.0
    x1, x2 = 1150.0, 1560.0
    ax.add_patch(Rectangle((x1 - wd / 2, base), wd, R_REQ * s, fc="0.86",
                           ec="0.35", lw=0.9, zorder=3))
    stack = [(PK["VDiagn"], C_DIAG, 0.85), (PK["VSt1"], C_STIR, 0.85),
             (PK["VSt2"], C_STIR, 0.45)]
    y = base
    for v, col, al in stack:
        ax.add_patch(Rectangle((x2 - wd / 2, y), wd, v * s, fc=col, alpha=al,
                               ec="white", lw=0.8, zorder=3))
        y += v * s
    ax.add_patch(Rectangle((x2 - wd / 2, y), wd, V_CONC * s, fc="0.86",
                           ec="0.35", lw=0.9, hatch="////", zorder=3))
    ax.plot([x1 - wd / 2 - 30, x2 + wd / 2], [base + S_MEAS * s] * 2,
            color="black", lw=1.0, ls=(0, (4, 2)), zorder=6)

    for v, col, txt in ((PK["VDiagn"], C_DIAG, "VDiagn"),
                        (PK["VSt1"], C_STIR, "VSt1"),
                        (PK["VSt2"], C_STIR, "VSt2")):
        pass
    y = base
    for (v, col, al), nm in zip(stack, ("VDiagn", "VSt1", "VSt2")):
        label(ax, x2 + wd / 2 + 34, y + 0.5 * v * s, f"{nm}  {v:.1f} kN",
              col=col if al > 0.6 else col, ha="left", box=False,
              fs=F.FS_SMALL)
        y += v * s
    label(ax, x2 + wd / 2 + 34, y + 0.5 * V_CONC * s,
          f"concrete residual\n{V_CONC:.1f} kN", col="0.30", ha="left",
          box=False, fs=F.FS_SMALL)
    label(ax, x1, base + R_REQ * s - 34, f"{R_REQ:.1f} kN", ha="center",
          va="bottom", box=False, fs=F.FS_SMALL)
    label(ax, x1, base + S_MEAS * s - 30, f"{SHARE:.0f} % of R", ha="center",
          va="bottom", col="0.15", fs=F.FS_SMALL)
    label(ax, x1, base + 40, "free body\nrequires", ha="center", va="top",
          box=False, fs=F.FS_SMALL, col="0.25")
    label(ax, x2, base + 40, "bar gauges\nmeasure", ha="center", va="top",
          box=False, fs=F.FS_SMALL, col="0.25")
    label(ax, 0.5 * (x1 + x2), base - 235,
          f"specimen NS-REF at failure: the measured bar forces carry "
          f"{SHARE:.0f} % of the demand,\nthe {100 - SHARE:.0f} % residual "
          f"being the concrete share the free body leaves out",
          ha="center", va="top", box=False)


# ---------------------------------------------------------------- assembly
def place(fig, ax, y0, y1, xlo, ylo, yhi, x0=0.052, w=0.905):
    """Aspect-equal schematic placed to a chosen box, limits made to match."""
    fw, fh = fig.get_size_inches()
    span = (yhi - ylo) * w * fw / ((y1 - y0) * fh)
    ax.set_xlim(xlo, xlo + span)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    return F.fit_schematic(fig, ax, y0, y1, x0=x0)


def main() -> None:
    fig = plt.figure(figsize=(F.FIG_W, 7.15))
    a = fig.add_axes((0.05, 0.55, 0.9, 0.40))
    b = fig.add_axes((0.05, 0.06, 0.9, 0.40))
    wa = place(fig, a, 0.545, 0.930, -340.0, -930.0, 190.0)
    wb = place(fig, b, 0.045, 0.430, -140.0, -1090.0, 215.0)
    panel_a(a)
    panel_b(b)
    F.fig_panel(fig, a, "a", "specimen and nib reinforcement", y=0.955)
    F.fig_panel(fig, b, "b", "free body and measured forces", y=0.455)
    F.save(fig, FIG / "halfjoint.png")
    plt.close(fig)

    print(f"  panel widths {wa:.3f}, {wb:.3f} of the figure")
    print(f"  crack from the corner at {CRACK_DEG:.0f} deg meets the top "
          f"fiber at x = {X_CRACK_TOP:.1f} mm")
    print(f"  milled zone {X_MILL0:.0f} to {X_MILL1:.0f} mm holds the crack "
          f"crossings of the U-bars ({X_CORNER + (H_NIB - Y_UBAR) / TAN:.0f}), "
          f"the diagonals ({cross_diag_crack()[0]:.0f}) and stirrup 1 "
          f"({STIRRUPS[0]:.0f}); stirrup 2 at {STIRRUPS[1]:.0f} is outside")
    print(f"  NS-REF at failure: R {R_REQ:.1f} kN required, measured "
          f"VDiagn {PK['VDiagn']:.1f} + VSt1 {PK['VSt1']:.1f} + VSt2 "
          f"{PK['VSt2']:.1f} = {S_MEAS:.1f} kN, {SHARE:.1f} % of R; "
          f"concrete residual {V_CONC:.1f} kN")


if __name__ == "__main__":
    main()
