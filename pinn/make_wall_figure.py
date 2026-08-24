"""The measured-structure figure: the gap, the data, and the response.

Section 9 makes a prediction from a sensor plan and then checks it against
the measurement. The figure carries the same three steps: where the
instrument is relative to the parameter it is meant to see, what it
actually recorded, and how the two candidate observables respond to the
parameter once the recording is used.

Run:  python make_wall_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))
import figstyle as F                                                       # noqa: E402
import matplotlib.pyplot as plt                                            # noqa: E402
from csfm_constitutive import CsfmMaterial, membrane                       # noqa: E402

DATA = HERE.parent / "data" / "wall_strain_gauges.csv"
FIG = HERE.parent / "figures"
L, H, T_THK = 800.0, 500.0, 100.0
BAND = 50.0
AS_TIE = 2.0 * np.pi * 6.0 ** 2
RHO_TIE = AS_TIE / (BAND * T_THK)
RHO_GRID = 2.0 * np.pi * 3.0 ** 2 / (100.0 * T_THK)
MAT = CsfmMaterial(fc=32.0, fy=500.0)
X_CUT, STRIP = 240.0, 40.0
F.apply()


def sigma_x(y, ex, ey, gxy, rho_x):
    t = lambda v: torch.tensor(np.asarray(v, float)).unsqueeze(-1)   # noqa: E731
    st = membrane(t(ex), t(ey), t(gxy), t(rho_x),
                  torch.full_like(t(ex), RHO_GRID), MAT, soften=True)
    return st["sigma_x"].squeeze().numpy()


def main() -> None:
    a = np.loadtxt(DATA, delimiter=",", skiprows=1)
    x, y, ex, ey, gxy = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]

    fig = plt.figure(figsize=(F.FIG_W, 7.1))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.00, 0.30, 0.95],
                          hspace=0.55, wspace=0.52, left=0.085,
                          right=0.975, top=0.895, bottom=0.070)

    # -- a: instrument and reinforcement, on the member ------------------
    ax = fig.add_subplot(gs[0, :])
    ax.add_patch(plt.Rectangle((0, 0), L, H, fc='0.965', ec='0.3', lw=1.2,
                               zorder=1))
    ax.add_patch(plt.Rectangle((0, 0), L, BAND, fc=F.VERM, alpha=0.26,
                               ec='none', zorder=2))
    for gy in np.arange(BAND + 71, H, 100.0):
        ax.plot([12, L - 12], [gy, gy], color='0.62', lw=1.0,
                ls=(0, (6, 3)), zorder=3)
    for gx in np.arange(70, L, 100.0):
        ax.plot([gx, gx], [12, H - 12], color='0.62', lw=1.0,
                ls=(0, (6, 3)), zorder=3)
    for by in (21.0, 33.0):
        ax.plot([10, L - 10], [by, by], color=F.VERM, lw=2.4, zorder=4)
    ax.scatter(x, y, s=4.5, color=F.SKY, alpha=0.9, lw=0, zorder=5,
               label=f'{x.size} gauges')
    ax.plot([], [], color=F.VERM, lw=2.4, label='tie, 2 bars 12 mm')
    ax.plot([], [], color='0.62', lw=1.0, ls=(0, (6, 3)),
            label='grid, 6 mm/100')
    for xs in (62.5, 737.5):
        ax.add_patch(plt.Rectangle((xs - 25, -30), 50, 30, fc='0.35',
                                   ec='none', zorder=6, clip_on=False))
    ax.add_patch(plt.Rectangle((370, H), 60, 30, fc='0.35', ec='none',
                               zorder=6, clip_on=False))
    ax.axvline(X_CUT, color=F.BLACK, lw=1.2, ls=(0, (5, 2)), zorder=7)
    ax.text(X_CUT + 10, H - 36, 'cut', fontsize=F.FS_ANNOT, va='top')
    ax.add_patch(plt.Rectangle((0, 0), L, 100, fill=False, ec='0.45',
                               lw=1.0, ls=(0, (4, 2)), zorder=8))
    ax.set_xlim(-25, L + 25); ax.set_ylim(-45, H + 40)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x$  (mm)'); ax.set_ylabel('$y$  (mm)')
    ax.legend(fontsize=F.FS_SMALL, loc='lower center',
              bbox_to_anchor=(0.5, 1.055), handlelength=1.4,
              markerscale=2.4, ncol=3, columnspacing=1.4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # -- b: the boxed strip, magnified clear of the member ---------------
    b = fig.add_subplot(gs[1, :])
    b.add_patch(plt.Rectangle((0, 0), L, BAND, fc=F.VERM, alpha=0.26,
                              ec='none'))
    for by in (21.0, 33.0):
        b.plot([0, L], [by, by], color=F.VERM, lw=2.6)
    low = y <= 95
    b.scatter(x[low], y[low], s=16, color=F.SKY, lw=0, zorder=4)
    # a 20 mm gap on a 100 mm axis is too short for an arrowhead to read,
    # so the two bounds are carried out to the margin and labeled between
    for yy, col in ((BAND, F.VERM), (y.min(), F.SKY)):
        b.plot([0, L + 200], [yy, yy], color=col, lw=0.9,
               ls=(0, (4, 2.5)), zorder=2, clip_on=False)
    b.axhspan(BAND, y.min(), xmin=0.80, color='0.85', lw=0, zorder=1)
    b.text(L + 100, 0.5 * (BAND + y.min()), f'{y.min()-BAND:.0f} mm',
           fontsize=F.FS_SMALL, va='center', ha='center', zorder=5,
           bbox=dict(fc='white', ec='none', pad=1.2))
    b.text(L + 30, 22, 'tie bars', fontsize=F.FS_SMALL, color=F.VERM,
           va='center', ha='left')
    b.text(L + 30, 88, 'lowest gauges', fontsize=F.FS_SMALL, color=F.SKY,
           va='center', ha='left')
    b.set_xlim(-25, L + 240); b.set_ylim(-6, 100)
    b.set_yticks([0, 50, 100])
    b.set_xlabel('$x$  (mm)'); b.set_ylabel('$y$  (mm)')
    F.clean(b)

    # -- c: the measurement, and where it stops -------------------------
    c = fig.add_subplot(gs[2, 0])
    lim = float(np.abs(ex * 1e3).max())
    sc = c.scatter(x, y, c=ex * 1e3, s=11, cmap='RdBu_r', lw=0,
                   vmin=-lim, vmax=lim, zorder=3)
    c.axhspan(0, BAND, color=F.VERM, alpha=0.22, lw=0, zorder=1)
    c.axhline(y.min(), color='0.3', lw=1.1, ls=(0, (4, 2)), zorder=4)
    c.text(L / 2, BAND / 2, 'tie, unsampled', fontsize=F.FS_SMALL,
           color=F.VERM, ha='center', va='center', zorder=6,
           bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.6))
    cb = fig.colorbar(sc, ax=c, pad=0.03, fraction=0.05)
    cb.ax.set_title('millistrain', fontsize=F.FS_SMALL, pad=5)
    cb.ax.tick_params(labelsize=F.FS_ANNOT)
    c.set_xlabel('$x$  (mm)'); c.set_ylabel('$y$  (mm)')
    c.set_xlim(0, L); c.set_ylim(0, H)
    F.clean(c)

    # -- d: response of each observable, against a case that works ------
    d = fig.add_subplot(gs[2, 1])
    sel = np.abs(x - X_CUT) < STRIP
    # The observable is the resultant the text names, so the stresses are
    # weighted by the tributary area of their gauge and summed with sign.
    # Summing magnitudes instead would report a norm, not a resultant, and
    # would overstate the swing on a cut that is largely in compression.
    o = np.argsort(y[sel])
    ys = y[sel][o]
    exs, eys, gs = ex[sel][o], ey[sel][o], gxy[sel][o]
    edges = np.concatenate(([0.0], 0.5 * (ys[1:] + ys[:-1]), [H]))
    dA = np.diff(edges) * T_THK
    grid = np.linspace(0.0, 0.70, 71)
    tie, gr = [], []
    for q in grid:
        rho = np.where(ys < BAND, RHO_TIE * (1.0 - q), RHO_GRID)
        tie.append(abs(float((sigma_x(ys, exs, eys, gs, rho) * dA).sum())))
        gr.append(abs(float((sigma_x(ys, exs, eys, gs,
                                     np.full(ys.size,
                                             RHO_GRID * (1 - q))) * dA).sum())))
    tie, gr = np.array(tie), np.array(gr)
    bench = 1.0 - 1.01 * grid / 0.70
    d.fill_between(grid, bench, 1.0, color=F.BLACK, alpha=0.08, lw=0,
                   label='signal when it works')
    d.plot(grid, bench, color=F.BLACK, lw=2.4, label='benchmark, for scale')
    d.plot(grid, gr / gr[0], color='0.35', lw=2.6, ls=(0, (5, 2)),
           label='grid: overlap, no share')
    d.plot(grid, tie / tie[0], color=F.VERM, lw=2.8, label='tie: no overlap')
    d.annotate(f'{100*abs(tie[-1]/tie[0]-1):.2f} % and '
               f'{100*abs(gr[-1]/gr[0]-1):.2f} %\nover the range',
               xy=(0.60, 1.0), xytext=(0.44, 0.62), fontsize=F.FS_ANNOT,
               color='0.3', ha='center',
               arrowprops=dict(arrowstyle='->', lw=0.9, color='0.5',
                               connectionstyle='arc3,rad=0.3',
                               shrinkA=2, shrinkB=5))
    d.set_xlabel(r'section loss  $\theta$')
    d.set_ylabel('observable, normalized')
    d.set_ylim(-0.06, 1.16)
    d.legend(fontsize=F.FS_SMALL, loc='lower left', handlelength=1.4,
             labelspacing=0.22, borderaxespad=0.5, framealpha=1.0).set_zorder(10)
    F.clean(d, grid=True)

    F.fig_panel(fig, ax, 'a', 'instrument and reinforcement', y=0.952)
    F.fig_panel(fig, b, 'b', 'the boxed strip, magnified', y=0.556)
    F.fig_panel(fig, c, 'c', 'measured horizontal strain', y=0.356)
    F.fig_panel(fig, d, 'd', 'response to the parameter', y=0.356)
    F.save(fig, FIG / 'wall.png')
    plt.close(fig)
    print(f"  tie varies {100*(tie.max()-tie.min())/tie.mean():.3f} %, "
          f"grid {100*(gr.max()-gr.min())/gr.mean():.2f} %")


if __name__ == '__main__':
    main()
