"""Figures for the manuscript, drawn from the cached reference fields.

Four, in the order a reader needs them: the field the identification is
posed on and why it admits no lever arm; the observable comparison that
decides whether the parameter is recoverable at all; the reaction
distribution that decides whether its moment arm is right; and the recovery
itself, with the function it is a root of.

Every quantity comes from figures/figdata.npz, written by figdata.py, so a
number in a panel can be traced to the field it was computed from.

Run:  python figdata.py && python make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from recover_utils import bracket_root

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))
import figstyle as F                                                       # noqa: E402
import matplotlib.pyplot as plt                                            # noqa: E402

F.apply()
FIG = HERE.parent / "figures"
D = np.load(FIG / "figdata.npz")

L, H, BAND = 2000.0, 1000.0, 150.0
SER = [F.BLACK, F.VERM, F.SKY, F.GREEN, F.PURPLE]   # discrete series
X_CUT = 700.0


def _beam(ax, band=True):
    """Outline of the member, so a field panel reads as a structure."""
    ax.add_patch(plt.Rectangle((0, 0), L, H, fill=False, ec='0.25', lw=1.0,
                               zorder=5))
    if band:
        ax.axhline(BAND, color='0.25', lw=0.7, ls=(0, (4, 2)), zorder=5)
    for xs in (250.0, 1750.0):
        ax.add_patch(plt.Rectangle((xs - 100, -46), 200, 46, fc='0.35',
                                   ec='none', zorder=6, clip_on=False))
    ax.add_patch(plt.Rectangle((900, H), 200, 46, fc='0.35', ec='none',
                               zorder=6, clip_on=False))
    ax.set_xlim(-40, L + 40)
    ax.set_ylim(-70, H + 70)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_yticks([0, 500, 1000])


# ======================================================================
# 1. the field, and why it admits no lever arm
# ======================================================================
def fig_field():
    cx, cy = D['cx'], D['cy']
    s3, ang = D['s3'], D['ang']
    fig = plt.figure(figsize=(F.FIG_W, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.50,
                          wspace=0.26, left=0.085, right=0.975,
                          top=0.945, bottom=0.070)

    # -- a: compression field and its trajectories ---------------------
    a = fig.add_subplot(gs[0, :])
    mag = np.clip(-s3, 0.0, None)
    im = a.pcolormesh(cx, cy, mag, cmap='BuPu', shading='gouraud',
                      vmin=0.0, vmax=float(np.percentile(mag, 99)))
    st = 2
    u, v = np.cos(ang[::st, ::st]), np.sin(ang[::st, ::st])
    w = mag[::st, ::st] / max(mag.max(), 1e-9)
    a.quiver(cx[::st, ::st], cy[::st, ::st], u * w, v * w,
             headwidth=0, headlength=0, headaxislength=0, pivot='mid',
             scale=13, width=0.0035, color='0.12', alpha=0.85)
    _beam(a)
    # an aspect-equal axes shrinks at draw time, so the bar is placed
    # against where the beam actually ends rather than against the slot
    fig.canvas.draw()
    pos = a.get_position()
    cax = fig.add_axes([pos.x1 + 0.012, pos.y0 + 0.06 * pos.height,
                        0.016, 0.88 * pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('compression  (MPa)', fontsize=F.FS_ANNOT)
    cb.ax.tick_params(labelsize=F.FS_ANNOT)
    a.set_ylabel('$y$  (mm)')
    a.set_xlabel('$x$  (mm)')
    a.set_anchor('W')
    a.annotate('the strut spreads\ninto a fan',
               xy=(790, 700), xytext=(1000, 560), fontsize=F.FS_ANNOT,
               color='0.15', ha='center', va='center', bbox=dict(fc='white', ec='none', alpha=0.78, pad=1.5), zorder=7,
               arrowprops=dict(arrowstyle='->', lw=0.9, color='0.35',
                               connectionstyle='arc3,rad=0.25',
                               shrinkA=2, shrinkB=4))
    a.annotate('and the tie anchors it', xy=(360, 75), xytext=(720, 210),
               fontsize=F.FS_ANNOT, color='0.15', ha='center', bbox=dict(fc='white', ec='none', alpha=0.78, pad=1.5), zorder=7,
               arrowprops=dict(arrowstyle='->', lw=0.9, color='0.35',
                               connectionstyle='arc3,rad=0.25',
                               shrinkA=2, shrinkB=4))
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)

    # -- b: tie force against the moment it is supposed to follow ------
    b = fig.add_subplot(gs[1, 0])
    x, T, M = D['prof_x'], D['prof_T'], D['prof_Mapp']
    inner = (x > 300) & (x < 1000)
    z_nom = 0.85 * (H - BAND / 2.0)
    beam = M * 1e3 / z_nom
    xi, Ti, Bi = x[inner], T[inner], beam[inner]
    b.fill_between(xi, Ti, Bi, where=Ti >= Bi, color=F.BLACK, alpha=0.12,
                   lw=0, label='band carries more')
    b.fill_between(xi, Ti, Bi, where=Ti < Bi, color=F.VERM, alpha=0.15,
                   lw=0, label='band carries less')
    b.plot(xi, Bi, color=F.VERM, lw=2.0, ls=(0, (5, 2)),
           label='beam theory, $M/z$')
    b.plot(xi, Ti, color=F.BLACK, lw=2.4, label='carried by the band')
    dff = Ti - Bi
    xx = bracket_root(dff, xi)
    if np.isfinite(xx):
        b.plot([xx], [np.interp(xx, xi, Ti)], marker='o', ms=6, mfc='white',
               mew=1.6, color='0.35', zorder=6)
        b.annotate('they agree at\none station only',
                   xy=(xx, float(np.interp(xx, xi, Ti))), xytext=(725, 150),
                   fontsize=F.FS_ANNOT, color='0.35', ha='left',
                   arrowprops=dict(arrowstyle='->', lw=0.9, color='0.5',
                                   connectionstyle='arc3,rad=-0.3',
                                   shrinkA=2, shrinkB=6))
    b.axvline(X_CUT, color='0.65', lw=0.9, ls=(0, (2, 2)), zorder=1)
    b.text(X_CUT + 16, 415, 'cut', fontsize=F.FS_ANNOT, color='0.45')
    b.set_xlabel('$x$  (mm)')
    b.set_ylabel('tie force  (kN)')
    b.set_yticks([0, 100, 200, 300, 400])
    b.set_ylim(-25, 450)
    b.legend(fontsize=F.FS_SMALL, loc='upper left', handlelength=1.3,
             labelspacing=0.25, borderaxespad=0.6)
    F.clean(b, grid=True)

    # -- c: the arm the statics needs, against any single value ---------
    c = fig.add_subplot(gs[1, 1])
    zc = D['prof_z']
    z_code = 0.85 * (H - BAND / 2.0)
    zi = zc[inner]
    c.axhspan(H, 1400, color=F.VERM, alpha=0.10, lw=0)
    c.text(985, H + 45, 'deeper than the section', fontsize=F.FS_ANNOT,
           color=F.VERM, va='bottom', ha='right', zorder=8,
           bbox=dict(fc='white', ec='none', pad=1.2, alpha=0.85))
    c.axhspan(0.8 * z_code, 1.2 * z_code, color='0.85', lw=0,
              label='$0.85d \pm 20\,\%$')
    c.axhline(z_code, color='0.45', lw=1.6, ls=(0, (5, 2)),
              label='a single arm, $0.85d$')
    c.axhline(H, color=F.VERM, lw=1.0, ls=(0, (2, 2)))
    c.plot(xi, zi, color=F.BLACK, lw=2.4, label='arm the statics needs')
    xcross = float(np.interp(z_code, zi, xi))
    c.plot([xcross], [z_code], marker='o', ms=6.5, mfc='white', mew=1.7,
           color=F.BLACK, zorder=6)
    c.annotate('right at one station', xy=(xcross, z_code),
               xytext=(870, 430), fontsize=F.FS_ANNOT, color='0.25',
               ha='center',
               arrowprops=dict(arrowstyle='->', lw=0.9, color='0.45',
                               connectionstyle='arc3,rad=-0.32',
                               shrinkA=2, shrinkB=7))
    lo, hi = float(np.nanmin(zi[xi > 400])), float(np.nanmax(zi))
    c.set_xlabel('$x$  (mm)')
    c.set_ylabel('lever arm  (mm)')
    c.set_ylim(0, 1330)
    c.legend(fontsize=F.FS_SMALL, loc='lower right', handlelength=1.3,
             labelspacing=0.25, borderaxespad=0.6)
    F.clean(c, grid=True)

    F.fig_panel(fig, a, 'a', 'principal compression field', y=0.958)
    F.fig_panel(fig, b, 'b', 'tie force and sectional prediction', y=0.372)
    F.fig_panel(fig, c, 'c', 'lever arm required by statics', y=0.372)
    F.save(fig, FIG / 'field.png')
    plt.close(fig)
    print(f'  implied arm {lo:.0f} to {hi:.0f} mm '
          f'({hi/max(lo,1e-9):.1f} times) across the shear span')


# ======================================================================
# 2. the choice of observable
# ======================================================================
def fig_observable():
    trial, pw, tie = D['obs_trial'], D['obs_pw'], D['obs_tie']
    ths, amin = D['obs_theta'], D['obs_argmin']
    fig = plt.figure(figsize=(F.FIG_W, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.42, 1.0],
                          hspace=0.52, wspace=0.30, left=0.085,
                          right=0.985, top=0.935, bottom=0.075)

    # -- a: the two objectives on one field ----------------------------
    a = fig.add_subplot(gs[0, 0])
    k = int(np.argmin(np.abs(ths - 0.20)))
    p = pw[k] / pw[k].max()
    t = np.abs(tie[k]) / np.abs(tie[k]).max()
    # the span between the truth and where the residual is minimized is the
    # error an optimizer given that objective would return
    a.axvspan(ths[k], amin[k], color=F.VERM, alpha=0.10, lw=0, zorder=0)
    a.plot(trial, t, color=F.BLACK, lw=2.6, label='integrated resultant',
           zorder=4)
    a.plot(trial, p, color=F.VERM, lw=2.6, label='pointwise residual',
           zorder=4)
    a.axvline(ths[k], color='0.35', lw=1.3, ls=(0, (4, 2)), zorder=2)
    a.plot([amin[k]], [p.min()], marker='v', ms=8, color=F.VERM,
           mec='white', mew=1.0, zorder=6)
    a.annotate('', xy=(ths[k], 0.24), xytext=(amin[k], 0.24),
               arrowprops=dict(arrowstyle='<->', lw=1.1, color=F.VERM))
    a.text(0.5 * (ths[k] + amin[k]), 0.29,
           f'{100*(amin[k]-ths[k]):.0f} pp', fontsize=F.FS_ANNOT,
           color=F.VERM, ha='center')
    a.text(ths[k] - 0.015, 1.12, 'truth', fontsize=F.FS_ANNOT,
           color='0.35', ha='right')
    a.text(amin[k] + 0.015, 1.12, 'minimized', fontsize=F.FS_ANNOT,
           color=F.VERM, ha='left')
    a.set_xlabel(r'trial section loss  $\hat\theta$')
    a.set_ylabel('objective, normalized')
    a.set_ylim(0, 1.22)
    a.legend(fontsize=F.FS_ANNOT, loc='lower left', handlelength=1.4,
             borderaxespad=0.7)
    F.clean(a, grid=True)

    # -- b: where the pointwise objective puts its minimum --------------
    b = fig.add_subplot(gs[0, 1])
    ok = np.isfinite(amin)
    b.fill_between(ths[ok], ths[ok], amin[ok], color=F.VERM, alpha=0.13,
                   lw=0, label='displacement', zorder=1)
    b.plot([0, 0.72], [0, 0.72], color='0.45', lw=1.4, ls=(0, (4, 2)),
           label='truth', zorder=3)
    b.plot(ths[ok], amin[ok], marker='o', ms=6, lw=2.4, color=F.VERM,
           mec='white', mew=1.0, label='pointwise minimizer', zorder=4)
    dpp = (amin[ok] - ths[ok]) * 100
    b.annotate(f'{dpp.min():.0f} to {dpp.max():.0f} % of section, every state',
               xy=(0.22, 0.5 * (0.22 + amin[ok][2])), xytext=(0.245, 0.635),
               fontsize=F.FS_ANNOT, color=F.VERM, ha='center',
               arrowprops=dict(arrowstyle='->', lw=0.9, color=F.VERM,
                               connectionstyle='arc3,rad=0.3',
                               shrinkA=2, shrinkB=5))
    b.set_xlabel(r'true section loss  $\theta$')
    b.set_ylabel(r'minimizer  $\hat\theta$')
    b.set_xlim(0, 0.72)
    b.set_ylim(0, 0.72)
    b.legend(fontsize=F.FS_ANNOT, loc='lower right', handlelength=1.4,
             borderaxespad=0.7)
    F.clean(b, grid=True)

    # -- c: where in the member the residual responds -------------------
    c = fig.add_subplot(gs[1, 0])
    sm = D['sens_map']
    nrm = sm / max(sm.max(), 1e-12)
    im = c.pcolormesh(D['cx'], D['cy'], nrm, cmap='Oranges',
                      shading='gouraud', vmin=0.0,
                      vmax=float(np.percentile(nrm[nrm > 0], 96)))
    _beam(c)
    frac = float((sm[:3] ** 2).sum() / max((sm ** 2).sum(), 1e-30))
    c.text(1010, 780, 'no response above the band',
           fontsize=F.FS_ANNOT, color='0.35', ha='center')
    c.set_yticks([0, 500, 1000])
    c.set_xlabel('$x$  (mm)')
    c.set_ylabel('$y$  (mm)')
    for s in ('top', 'right'):
        c.spines[s].set_visible(False)

    # -- d: the slope of each objective, which is why one has a false
    #       minimum and the other has a unique root
    d = fig.add_subplot(gs[1, 1])
    gp = np.gradient(pw[k] / pw[k].max(), trial)
    gt = np.gradient(np.abs(tie[k]) / np.abs(tie[k]).max(), trial)
    d.axhline(0, color='0.45', lw=1.0, zorder=2)
    d.fill_between(trial, 0, gp, where=gp < 0, color=F.VERM, alpha=0.13,
                   lw=0, zorder=1)
    d.fill_between(trial, 0, gp, where=gp >= 0, color=F.VERM, alpha=0.30,
                   lw=0, zorder=1)
    d.fill_between(trial, 0, gt, color=F.BLACK, alpha=0.10, lw=0, zorder=1)
    d.plot(trial, gt, color=F.BLACK, lw=2.6, label='integrated resultant',
           zorder=4)
    d.plot(trial, gp, color=F.VERM, lw=2.6, label='pointwise residual',
           zorder=4)
    zc_ = bracket_root(gp, trial)
    d.plot([zc_], [0.0], marker='o', ms=7, mfc='white', mew=1.7,
           color=F.VERM, zorder=6)
    d.annotate('slope changes sign', xy=(zc_, 0.0), xytext=(0.545, -0.34),
               fontsize=F.FS_ANNOT, color=F.VERM, ha='center',
               arrowprops=dict(arrowstyle='->', lw=0.9, color=F.VERM,
                               connectionstyle='arc3,rad=0.3',
                               shrinkA=2, shrinkB=7))
    d.set_xlabel(r'trial section loss  $\hat\theta$')
    d.set_ylabel('slope of the objective')
    d.set_ylim(-1.20, 0.95)
    d.legend(fontsize=F.FS_ANNOT, loc='upper left', handlelength=1.4,
             borderaxespad=0.7)
    F.clean(d, grid=True)

    F.fig_panel(fig, a, 'a', 'both objectives on one field', y=0.948)
    F.fig_panel(fig, b, 'b', 'minimizer against generating value', y=0.948)
    F.fig_panel(fig, c, 'c', 'spatial response to the parameter', y=0.373)
    F.fig_panel(fig, d, 'd', 'slope of each objective', y=0.373)
    F.save(fig, FIG / 'observable_choice.png')
    plt.close(fig)
    sw = 100 * float((pw[k].max() - pw[k].min()) / pw[k].mean())
    tie_sw = 100 * float((np.nanmax(np.abs(tie[k])) - np.nanmin(np.abs(tie[k])))
                         / np.nanmean(np.abs(tie[k])))
    print(f'  pointwise minimizer displaced {dpp.min():.1f} to {dpp.max():.1f} pp; '
          f'swing {sw:.1f} % '
          f'against {tie_sw:.0f} % for the resultant '
          f'({tie_sw/max(sw,1e-9):.0f} times)')


# ======================================================================
# 3. the reaction and its moment arm
# ======================================================================
def fig_reaction():
    fig = plt.figure(figsize=(F.FIG_W, 5.9))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.52, left=0.085,
                          right=0.985, top=0.935, bottom=0.075)
    xs, rs = D['react_x'], D['react_r']
    xc = float((xs * rs).sum() / rs.sum())

    a = fig.add_subplot(gs[0, 0])
    a.axvspan(150, 350, color='0.90', zorder=0)
    a.text(250, a.get_ylim()[1], '', ha='center')
    up = rs < 0
    a.bar(xs[~up], rs[~up], width=34, color=F.SKY, alpha=0.92,
          label='bearing', zorder=3)
    a.bar(xs[up], rs[up], width=34, color=F.VERM, alpha=0.88,
          label='uplift', zorder=3)
    a.axhline(0, color='0.4', lw=0.9, zorder=2)
    a.axvline(250, color='0.45', ls=(0, (4, 2)), lw=1.2, zorder=2)
    a.axvline(xc, color=F.GREEN, lw=2.2, zorder=4)
    a.text(xc + 8, 300, f'centroid\n{xc:.0f} mm', fontsize=F.FS_ANNOT,
           color=F.GREEN, ha='left', va='center')
    a.text(250, -168, 'plate center', fontsize=F.FS_ANNOT, color='0.4',
           ha='center', va='bottom', zorder=8,
           bbox=dict(fc='white', ec='none', pad=1.0, alpha=0.9))
    a.axvline(350, color=F.BLACK, lw=1.6, ls=(0, (2, 2)), zorder=4)
    a.text(422, 600, 'contact centroid,\n350 mm', fontsize=F.FS_ANNOT,
           color='0.2', ha='right', va='top', zorder=8,
           bbox=dict(fc='white', ec='none', pad=1.0, alpha=0.9))
    a.text(250, 600, f'{abs(rs[up].sum()):.0f} kN of\nuplift',
           fontsize=F.FS_ANNOT, color=F.VERM, ha='center', va='top')
    a.set_xlim(115, 425)
    a.set_ylim(-190, 760)
    a.set_xlabel('position on the bearing  (mm)')
    a.set_ylabel('nodal reaction  (kN)')
    a.legend(fontsize=F.FS_SMALL, loc='center left', handlelength=1.2,
             labelspacing=0.25, borderaxespad=0.6)
    F.clean(a, grid=True)

    b = fig.add_subplot(gs[0, 1])
    cent, cth, cdl = D['cent'], D['cent_theta'], D['cent_delta']
    for j, dl in enumerate(cdl):
        b.plot(cth * 100, cent[:, j], marker='o', ms=3.4, lw=1.6,
               color=SER[j % len(SER)], label=f'{dl:.1f}')
    b.axhline(250, color=F.VERM, ls=(0, (5, 2)), lw=1.6)
    b.text(2, 258, 'assumed', fontsize=F.FS_ANNOT, color=F.VERM)
    b.set_xlabel('tie section loss  (%)')
    b.set_ylabel('centroid  (mm)')
    b.set_ylim(235, 400)
    b.legend(fontsize=F.FS_SMALL, loc='center right',
             title='deflection (mm)', title_fontsize=F.FS_SMALL,
             handlelength=1.1, labelspacing=0.22)
    F.clean(b)

    c = fig.add_subplot(gs[1, 0])
    xm = D['mom_x']
    sel = (xm > 260) & (xm < 1010)
    fld = np.abs(D['mom_int'][sel])
    m370, m250 = D['mom_a370'][sel], D['mom_a250'][sel]
    c.fill_between(xm[sel], fld, m250, color=F.VERM, alpha=0.18, lw=0,
                   label='error from the arm')
    c.plot(xm[sel], m250, color=F.VERM, lw=1.8, ls=(0, (5, 2)),
           label='statics, plate center')
    c.plot(xm[sel], m370, color=F.GREEN, lw=1.8, label='statics, centroid')
    c.plot(xm[sel], fld, color=F.BLACK, lw=2.4, label='in the field')
    xg = float(xm[sel][-2]); hi = float(m250[-2]); lo = float(fld[-2])
    c.annotate('', xy=(xg, hi), xytext=(xg, lo),
               arrowprops=dict(arrowstyle='<->', lw=1.0, color='0.25'))
    c.annotate(f'{hi-lo:.0f} kN m', xy=(xg, 0.5 * (hi + lo)),
               xytext=(-10, 14), textcoords='offset points',
               fontsize=F.FS_SMALL, color='0.25', ha='right', va='bottom')
    c.set_xlabel('cut position  (mm)')
    c.set_ylabel('moment  (kN m)')
    c.legend(fontsize=F.FS_SMALL, loc='lower right', handlelength=1.3,
             labelspacing=0.25, borderaxespad=0.6)
    c.set_ylim(-20, 430)
    F.clean(c, grid=True)

    # -- d: what assuming the arm costs, in the units the method reports
    d = fig.add_subplot(gs[1, 1])
    arms, arec = D['arm_a'], D['arm_rec']
    ths_a = D['cent_theta']
    d.axvspan(150, 350, color='0.90', lw=0, zorder=0)
    d.text(344, 0.475, 'plate', fontsize=F.FS_ANNOT, color='0.45',
           ha='right', va='top')
    for i, th in enumerate(ths_a):
        if not np.isfinite(arec[i]).any():
            continue
        d.plot(arms, arec[i], lw=2.0, color=SER[i % len(SER)],
               label=f'{th*100:.0f}')
        d.plot([370], [np.interp(370, arms, arec[i])], marker='o', ms=5,
               mfc='white', mew=1.5, color=SER[i % len(SER)], zorder=6)
        d.axhline(th, color=SER[i % len(SER)], lw=0.7, ls=(0, (2, 3)),
                  alpha=0.55, zorder=1)
    d.axvline(370, color=F.GREEN, lw=1.8, zorder=2)
    d.axvline(250, color='0.45', lw=1.2, ls=(0, (4, 2)), zorder=2)
    d.text(378, 0.44, 'centroid', fontsize=F.FS_ANNOT, color=F.GREEN,
           rotation=90, va='bottom', ha='left')
    d.text(246, 0.255, 'plate center', fontsize=F.FS_ANNOT, color='0.45',
           rotation=90, va='bottom', ha='right')
    d.set_xlabel('assumed position of the reaction  (mm)')
    d.set_ylabel(r'recovered  $\hat\theta$')
    d.set_xlim(235, 445)
    d.set_ylim(0, 0.70)
    d.legend(fontsize=F.FS_SMALL, loc='upper left', ncol=2,
             bbox_to_anchor=(0.0, 1.02), title='true loss (%)',
             title_fontsize=F.FS_SMALL, handlelength=1.1,
             labelspacing=0.22, columnspacing=0.8, borderaxespad=0.6)
    F.clean(d, grid=True)

    F.fig_panel(fig, a, 'a', 'reaction across the bearing', y=0.948)
    F.fig_panel(fig, b, 'b', 'centroid across the state grid', y=0.948)
    F.fig_panel(fig, c, 'c', 'transmitted moment by assumed arm', y=0.437)
    F.fig_panel(fig, d, 'd', 'recovered loss against assumed arm', y=0.437)
    F.save(fig, FIG / 'reaction_centroid.png')
    plt.close(fig)
    print(f'  centroid {D["cent"].min():.0f} to {D["cent"].max():.0f} mm '
          f'over the grid')


# ======================================================================
# 4. recovery
# ======================================================================
def fig_recovery():
    fig = plt.figure(figsize=(F.FIG_W, 6.1))
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.30,
                          left=0.090, right=0.985, top=0.935, bottom=0.075)
    ths, grid, fc = D['rec_theta'], D['rec_grid'], D['rec_f']
    rec = D['rec']
    # the three noise models of Table 2, cached by figdata.py from the
    # same seeded realizations noise_study.py prints, so a panel and a
    # table cell can never disagree
    nm = D['nm_rec']
    nmsty = [('independent', 's', F.SKY, -0.010),
             ('correlated 150 mm', '^', F.GREEN, 0.0),
             ('15 % dropout', 'D', F.PURPLE, 0.010)]

    a = fig.add_subplot(gs[0, 0])
    for i, th in enumerate(ths):
        if not np.isfinite(fc[i]).any():
            continue
        a.plot(grid, fc[i], lw=1.9, color=SER[i % len(SER)],
               label=f'{th*100:.0f}')
    a.axhline(0, color='0.45', lw=0.9)
    for i, th in enumerate(ths):
        r = rec[i, 0, 0]
        if np.isfinite(r):
            a.plot([r], [0], marker='o', ms=4.8, mfc='white', mew=1.4,
                   color=SER[i % len(SER)], zorder=5)
    a.set_xlabel(r'trial section loss  $\hat\theta$')
    a.set_ylabel('couple less statics  (kN m)')
    a.set_ylim(-192, 72)
    a.legend(fontsize=F.FS_SMALL, loc='upper right',
             title='true loss (%)', title_fontsize=F.FS_SMALL,
             handlelength=1.1, labelspacing=0.22, ncol=2, columnspacing=0.8)
    F.clean(a)

    b = fig.add_subplot(gs[0, 1])
    b.plot([0, 0.45], [0, 0.45], color='0.6', lw=1.0, ls=(0, (4, 2)),
           label='truth')
    base = rec[:, 0, 0]
    b.plot(ths, base, marker='o', ms=4.8, lw=1.8, color=F.BLACK,
           mfc='white', mew=1.4, label='no noise', zorder=5)
    for j, (lab, mk, colr, dx) in enumerate(nmsty):
        m = np.nanmean(nm[j], axis=1)
        e = np.nanstd(nm[j], axis=1)
        b.errorbar(ths + dx, m, yerr=e, marker=mk, ms=4.6, lw=1.6,
                   ls='none', capsize=2.2, color=colr, label=lab, zorder=4)
    b.set_xlabel(r'true section loss  $\theta$')
    b.set_ylabel(r'recovered  $\hat\theta$')
    b.set_xlim(-0.03, 0.46)
    b.set_ylim(-0.03, 0.46)
    b.legend(fontsize=F.FS_SMALL, loc='upper left', handlelength=1.4,
             labelspacing=0.22, borderaxespad=0.7)
    F.clean(b)

    c = fig.add_subplot(gs[1, 0])
    rd, dl = D['rec_dl'], D['cent_delta']
    for i, th in enumerate(ths):
        if not np.isfinite(rd[i]).any():
            continue
        c.plot(dl, (rd[i] - th) * 100, marker='o', ms=4, lw=1.8,
               color=SER[i % len(SER)], label=f'{th*100:.0f}')
    c.axhline(0, color='0.45', lw=0.9)
    c.set_xlabel('deflection at measurement  (mm)')
    c.set_ylabel('error  (% of section)')
    c.legend(fontsize=F.FS_SMALL, loc='lower right', ncol=3,
             title='true section loss (%)', title_fontsize=F.FS_SMALL,
             handlelength=1.2, labelspacing=0.25, columnspacing=0.9,
             borderaxespad=0.6)
    # the intact state has no admissible root below 5 mm, so its curve
    # starts where one first exists
    c.axvspan(0.5, 4.6, color=F.VERM, alpha=0.07, lw=0, zorder=0)
    c.text(2.55, 4.6, 'no admissible root\nfor the intact tie',
           fontsize=F.FS_SMALL, color=F.VERM, ha='center', va='top')
    c.set_xlim(0.5, 7.5)
    c.set_ylim(-11.5, 6.5)
    c.set_yticks([-10, -5, 0, 5])
    F.clean(c)

    d = fig.add_subplot(gs[1, 1])
    bias = np.abs(base - ths) * 100
    # the cost of a noise model at a state is the mean displacement of the
    # recovery from its own no-noise root, over the realizations that
    # admit a root at all; the systematic bias is charged separately
    cost = np.full((len(nmsty), len(ths)), np.nan)
    for j in range(len(nmsty)):
        for i in range(len(ths)):
            if np.isfinite(base[i]):
                cost[j, i] = np.nanmean(np.abs(nm[j, i] - base[i])) * 100
    idx = np.arange(len(ths))
    w, step = 0.185, 0.20
    d.bar(idx - 1.5 * step, bias, w, color='0.35', label='systematic bias')
    for j, (lab, mk, colr, dx) in enumerate(nmsty):
        d.bar(idx + (j - 0.5) * step, cost[j], w, color=colr, label=lab)
    # broken over three lines so the label clears the spine on its left
    # and the first bar group on its right; the 0 % slot carries no bar
    d.text(0, 0.6, 'no root\nwithout\nnoise', fontsize=F.FS_SMALL,
           color='0.45', ha='center', va='bottom')
    d.set_xticks(idx)
    d.set_xticklabels([f'{t*100:.0f}' for t in ths])
    d.set_xlim(-0.55, 4.55)
    d.set_xlabel('true section loss  (%)')
    d.set_ylabel('error  (% of section)')
    d.legend(fontsize=F.FS_SMALL, loc='upper left', handlelength=1.2,
             labelspacing=0.22, borderaxespad=0.7)
    d.set_ylim(0, 13.2)
    F.clean(d)

    F.fig_panel(fig, a, 'a', 'identifying function and its root', y=0.948)
    F.fig_panel(fig, b, 'b', 'recovered against generating value', y=0.948)
    F.fig_panel(fig, c, 'c', 'error against load level', y=0.424)
    F.fig_panel(fig, d, 'd', 'bias and the cost of noise', y=0.424)
    F.save(fig, FIG / 'recovery.png')
    plt.close(fig)
    mx = [float(np.nanmax(cost[j])) for j in range(len(nmsty))]
    print(f'  bias {np.nanmin(bias):.1f} to {np.nanmax(bias):.1f} pp; '
          f'cost of 5 % noise at most: independent {mx[0]:.1f} pp, '
          f'correlated {mx[1]:.1f} pp, dropout {mx[2]:.1f} pp')


if __name__ == '__main__':
    fig_field(); fig_observable(); fig_reaction(); fig_recovery()
