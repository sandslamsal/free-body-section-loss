"""Everything the figures plot, computed once and cached.

The figures in this study are drawn from the reference fields rather than
from summary numbers, so the quantity behind every curve is recomputed here
and written to a single archive. Two reasons. A figure whose numbers live
only in the plotting script cannot be checked against the text, and a
figure that recomputes on every redraw is slow enough that it stops being
redrawn.

Run:  python figdata.py        (writes figures/figdata.npz)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

from arclength_oracle import build_mesh, membrane as membrane_np          # noqa: E402
from csfm_constitutive import membrane                                     # noqa: E402
from identify import rho_x_of_theta                                        # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                         # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_nodal import internal_forces                                  # noqa: E402
from recover_utils import element_strains, bracket_root                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "figdata.npz"

NX, NY = 40, 20            # structured mesh: two triangles per cell
BAND = 150.0               # tie-band depth (mm)
X_CUT = 700.0              # cut station, clear of support and load
BAND_W = 50.0              # half-width of the strip that stands for the cut
REF = ("0.00", "3.5")      # the state the field panels are drawn at


# ----------------------------------------------------------------------
# element fields on the structured grid
# ----------------------------------------------------------------------
def cell_fields(prob, xy, u, theta):
    """Cell-centered stress and strain on the NX by NY grid.

    The two triangles of a cell are averaged, which is the natural way to
    read a constant-strain discretization as a field: the pair spans the
    cell and their mean is the cell average exactly, since the two have
    equal area.
    """
    cx, cy, ex, ey, gxy = element_strains(xy, u, NX, NY)
    X = torch.tensor(cx).unsqueeze(-1)
    Y = torch.tensor(cy).unsqueeze(-1)
    st = membrane(torch.tensor(ex).unsqueeze(-1), torch.tensor(ey).unsqueeze(-1),
                  torch.tensor(gxy).unsqueeze(-1),
                  rho_x_of_theta(prob, X, Y, torch.tensor(float(theta))),
                  prob.rho_y(X, Y), prob.mat, soften=True)
    out = {}
    for k, v in (("sx", st["sigma_x"]), ("sy", st["sigma_y"]),
                 ("txy", st["tau_xy"])):
        out[k] = v.squeeze().numpy()
    out.update(cx=cx, cy=cy, ex=ex, ey=ey, gxy=gxy)
    # pair the triangles back into cells
    for k in list(out):
        a = out[k]
        out[k] = 0.5 * (a[0::2] + a[1::2])
    n = out["cx"].size
    assert n == NX * NY, n
    for k in list(out):
        out[k] = out[k].reshape(NY, NX)
    return out


def principal(sx, sy, txy):
    """Principal stresses and the direction of the minor (compressive) one."""
    m = 0.5 * (sx + sy)
    r = np.sqrt((0.5 * (sx - sy)) ** 2 + txy ** 2)
    s1, s3 = m + r, m - r
    ang = 0.5 * np.arctan2(2.0 * txy, sx - sy)      # direction of s1
    return s1, s3, ang + np.pi / 2.0                # rotate to the minor axis


# ----------------------------------------------------------------------
# what the tie carries along the span, and the arm that implies
# ----------------------------------------------------------------------
def tie_profile(prob, f):
    """Band tension and applied moment, station by station along the span.

    T(x) is the horizontal force carried by the band; z(x) is the arm the
    applied moment would need in order to be resisted by that force alone.
    In a beam z is a property of the section and T follows the moment; the
    point of the panel is that here it does not.
    """
    dy = prob.H / NY
    band_rows = f["cy"][:, 0] < BAND
    t = prob.t
    T = (f["sx"][band_rows, :] * dy * t).sum(axis=0) / 1e3          # kN
    x = f["cx"][0, :]
    return x, T


def internal_moment_profile(prob, f, y0):
    """Moment of sigma_x about y0, station by station, in kN m."""
    dy = prob.H / NY
    return (f["sx"] * (f["cy"] - y0) * dy * prob.t).sum(axis=0) / 1e6


# ----------------------------------------------------------------------
# the cut, and the reconciliation posed on band strain alone
# ----------------------------------------------------------------------
def cut_quantities(prob, cx, cy, ex, ey, gxy, area, theta):
    """Normal stress on the strip that stands for the cut, and its parts."""
    sel = np.abs(cx - X_CUT) < BAND_W
    X = torch.tensor(cx[sel]).unsqueeze(-1)
    Y = torch.tensor(cy[sel]).unsqueeze(-1)
    st = membrane(torch.tensor(ex[sel]).unsqueeze(-1),
                  torch.tensor(ey[sel]).unsqueeze(-1),
                  torch.tensor(gxy[sel]).unsqueeze(-1),
                  rho_x_of_theta(prob, X, Y, torch.tensor(float(theta))),
                  prob.rho_y(X, Y), prob.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy()
    ys = cy[sel]
    dA = area / (2.0 * BAND_W) * prob.t
    return sx, ys, dA


def band_couple(prob, cx, cy, ex, ey, gxy, area, theta):
    """Tie resultant, lever arm and their couple, from band strain.

    The band supplies T; axial equilibrium of the cut supplies the fact
    that the compression resultant equals it; and the arm is the distance
    between the two centroids. Only the compression centroid comes from
    the model rather than from the measurement, and it is the one quantity
    of the four that the parameter does not touch.
    """
    sx, ys, dA = cut_quantities(prob, cx, cy, ex, ey, gxy, area, theta)
    inb = ys < BAND
    T = float((sx[inb] * dA).sum()) / 1e3                            # kN
    wT = np.clip(sx[inb], 0.0, None)
    yT = float((wT * ys[inb]).sum() / max(wT.sum(), 1e-9))
    wC = np.clip(-sx[~inb], 0.0, None)
    yC = float((wC * ys[~inb]).sum() / max(wC.sum(), 1e-9))
    z = yC - yT
    return T, z, T * z / 1e3                                         # kN m


def recover_band(prob, cx, cy, ex, ey, gxy, area, lam, a_arm,
                 grid=None):
    """Root of the band couple against statics, and the function it is a root of."""
    if grid is None:
        grid = np.linspace(0.0, 0.70, 71)
    R = lam * prob.P / 2.0
    M_req = R * (X_CUT - a_arm) / 1e6                                # kN m
    f = np.array([band_couple(prob, cx, cy, ex, ey, gxy, area, g)[2] - M_req
                  for g in grid])
    root = bracket_root(f, grid)
    return root, grid, f, M_req


# ----------------------------------------------------------------------
def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    xy = d["xy"]
    area = (prob.L / NX) * (prob.H / NY) / 2.0
    thetas = [float(t) for t in d["theta_true"]]
    deltas = [float(t) for t in d["deltas"]]
    out = {}

    # ---- 1. the field at the reference state -------------------------
    print("field at the reference state ...", flush=True)
    th_r, dl_r = REF
    u_r = d[f"u_{th_r}_{dl_r}"]
    f_r = cell_fields(prob, xy, u_r, float(th_r))
    s1, s3, ang = principal(f_r["sx"], f_r["sy"], f_r["txy"])
    out.update(cx=f_r["cx"], cy=f_r["cy"], sx=f_r["sx"], sy=f_r["sy"],
               txy=f_r["txy"], ex=f_r["ex"], s1=s1, s3=s3, ang=ang,
               lam_ref=float(d[f"lam_{th_r}_{dl_r}"][0]))

    # ---- 2. tie force and the arm it implies, along the span ---------
    x, T = tie_profile(prob, f_r)
    R = out["lam_ref"] * prob.P / 2.0
    M_app = R * np.clip(x - 370.0, 0.0, None) / 1e6                  # kN m
    z_implied = np.where(np.abs(T) > 1.0, M_app * 1e3 / np.maximum(T, 1e-9),
                         np.nan)
    # the arm actually present in the field, station by station
    yT, yC = [], []
    for c in range(NX):
        col_s, col_y = f_r["sx"][:, c], f_r["cy"][:, c]
        wT = np.clip(col_s * (col_y < BAND), 0.0, None)
        wC = np.clip(-col_s * (col_y >= BAND), 0.0, None)
        yT.append((wT * col_y).sum() / max(wT.sum(), 1e-9))
        yC.append((wC * col_y).sum() / max(wC.sum(), 1e-9))
    yT, yC = np.array(yT), np.array(yC)
    out.update(prof_x=x, prof_T=T, prof_Mapp=M_app, prof_z=z_implied,
               prof_yT=yT, prof_yC=yC, prof_zfield=yC - yT)

    # band strain along the span at three deterioration states
    eps_band = []
    for th in (0.0, 0.20, 0.40):
        k = f"u_{th:.2f}_{dl_r}"
        if k not in d:
            eps_band.append(np.full_like(x, np.nan)); continue
        fx = cell_fields(prob, xy, d[k], th)
        rows = fx["cy"][:, 0] < BAND
        eps_band.append(fx["ex"][rows, :].mean(axis=0))
    out["eps_band"] = np.array(eps_band)

    # ---- 3. the two observables against the parameter ----------------
    # The pointwise residual is the continuum quantity a physics-informed
    # formulation minimizes, so it is evaluated as a divergence of the
    # interpolated stress field rather than as an assembled nodal force:
    # the assembled residual is the identifying functional of Section 5 and
    # comparing the method against itself would prove nothing.
    print("observables ...", flush=True)
    dxc, dyc = prob.L / NX, prob.H / NY

    def pointwise(u, theta, rows=None):
        """Mean-square divergence of the interpolated stress field."""
        f = cell_fields(prob, xy, u, theta)
        r1 = (np.gradient(f["sx"], dxc, axis=1)
              + np.gradient(f["txy"], dyc, axis=0))
        r2 = (np.gradient(f["txy"], dxc, axis=1)
              + np.gradient(f["sy"], dyc, axis=0))
        sq = r1 ** 2 + r2 ** 2
        return sq if rows is None else sq[:rows, 1:-1].mean()

    trial = np.linspace(0.0, 0.70, 29)
    pw, tie, argmin = [], [], []
    for th in thetas:
        k = f"u_{th:.2f}_{dl_r}"
        if k not in d:
            pw.append(np.full(trial.size, np.nan))
            tie.append(np.full(trial.size, np.nan))
            argmin.append(np.nan)
            continue
        uu = d[k]
        v = np.array([pointwise(uu, g)[1:-1, 1:-1].mean() for g in trial])
        pw.append(v)
        argmin.append(float(trial[v.argmin()]))
        cx_, cy_, ex_, ey_, gxy_ = element_strains(xy, uu, NX, NY)
        tie.append(np.array([band_couple(prob, cx_, cy_, ex_, ey_, gxy_,
                                         area, g)[0] for g in trial]))
        print(f"  true {th:.2f}: pointwise minimized at {argmin[-1]:.3f}",
              flush=True)
    out.update(obs_trial=trial, obs_pw=np.array(pw), obs_tie=np.array(tie),
               obs_argmin=np.array(argmin), obs_theta=np.array(thetas))

    # ---- 4. where in the member the residual responds ----------------
    print("sensitivity map ...", flush=True)
    u_o = d[f"u_0.20_{dl_r}"]
    h = 0.05
    smap = (np.sqrt(pointwise(u_o, 0.20 + h)) - np.sqrt(pointwise(u_o, 0.20 - h))) / (2 * h)
    out["sens_map"] = np.abs(smap)

    # ---- 5. the dilution law ----------------------------------------
    # The evaluation domain grows upward from the soffit, so beta, the
    # fraction of it occupied by the band, falls from one to the band
    # fraction of the whole member. The relative swing of the objective
    # over the admissible range is measured against it.
    print("dilution ...", flush=True)
    betas, sens = [], []
    for rows in range(3, NY + 1):
        v = np.array([pointwise(u_o, g, rows=rows) for g in trial])
        betas.append(3.0 / rows)
        sens.append(float((v.max() - v.min()) / v.mean()))
    out.update(dil_beta=np.array(betas), dil_sens=np.array(sens))

    # ---- 5b. how the band tension divides between the two materials --
    # The share carried by steel is the signal fraction, and it has to be
    # read at the strain the band actually reaches. Read at a strain where
    # the bar is still elastic it is far smaller, which is how the same
    # calculation can be made to say the parameter is unrecoverable.
    print("signal fraction ...", flush=True)
    eps = np.logspace(-4.3, -1.6, 60)
    e_t = torch.tensor(eps).unsqueeze(-1)
    z_t = torch.zeros_like(e_t)
    y_b = torch.full_like(e_t, 50.0)
    rho_b = rho_x_of_theta(prob, z_t, y_b, torch.tensor(0.0))
    st = membrane(e_t, z_t, z_t, rho_b, prob.rho_y(z_t, y_b), prob.mat,
                  soften=True)
    sx_tot = st["sigma_x"].squeeze().numpy()
    st0 = membrane(e_t, z_t, z_t, torch.zeros_like(rho_b),
                   prob.rho_y(z_t, y_b), prob.mat, soften=True)
    sx_conc = st0["sigma_x"].squeeze().numpy()
    share = np.clip((sx_tot - sx_conc) / np.where(np.abs(sx_tot) > 1e-9,
                                                  sx_tot, np.nan), 0.0, 1.0)
    eps_reached = float(np.abs(f_r["ex"][f_r["cy"] < BAND]).max())
    out.update(sf_eps=eps, sf_share=share, sf_reached=eps_reached,
               sf_yield=prob.mat.fy / prob.mat.Es)
    print(f"  band reaches {eps_reached:.2e}; yield at "
          f"{prob.mat.fy/prob.mat.Es:.2e}", flush=True)

    # ---- 6. the reaction, over the whole grid ------------------------
    print("reactions ...", flush=True)
    xs_ref, rs_ref, cents = None, None, []
    for th in thetas:
        row = []
        for dl in deltas:
            k = f"u_{th:.2f}_{dl:.1f}"
            if k not in d:
                row.append(np.nan); continue
            pr = deepbeam_rho(RHO_NOM * (1 - th)); mh = build_mesh(pr)
            uu = d[k].ravel(); lm = float(d[f"lam_{th:.2f}_{dl:.1f}"][0])
            Rv = internal_forces(uu, pr, mh, th) - lm * mh.F_ref
            fx = np.asarray(mh.fixed, bool)
            xs, rs = [], []
            for n in range(mh.n_node):
                if fx[2 * n + 1] and mh.xy[n, 0] < 600:
                    xs.append(mh.xy[n, 0]); rs.append(Rv[2 * n + 1] / 1e3)
            xs, rs = np.array(xs), np.array(rs)
            row.append(float((xs * rs).sum() / rs.sum()))
            if th == 0.20 and abs(dl - 3.5) < 1e-9:
                xs_ref, rs_ref = xs, rs
        cents.append(row)
    out.update(react_x=xs_ref, react_r=rs_ref,
               cent=np.array(cents), cent_theta=np.array(thetas),
               cent_delta=np.array(deltas))

    # ---- 7. the moment the cut transmits, arm by arm -----------------
    print("moment profile ...", flush=True)
    y0 = prob.H / 2.0
    M_int = internal_moment_profile(prob, f_r, y0)
    xs_m = f_r["cx"][0, :]
    out.update(mom_x=xs_m, mom_int=M_int,
               mom_a250=R * np.clip(xs_m - 250.0, 0, None) / 1e6,
               mom_a370=R * np.clip(xs_m - 370.0, 0, None) / 1e6)

    # ---- 7b. what the assumed reaction arm costs ---------------------
    # The arm is the largest single sensitivity in the method, so the
    # recovered value is swept against it directly rather than reported at
    # one assumed position.
    print("arm sweep ...", flush=True)
    arms = np.linspace(240.0, 440.0, 21)
    arm_rec = np.full((len(thetas), arms.size), np.nan)
    for i, th in enumerate(thetas):
        k = f"u_{th:.2f}_{dl_r}"
        if k not in d:
            continue
        lm = float(d[f"lam_{th:.2f}_{dl_r}"][0])
        cx, cy, ex, ey, gxy = element_strains(xy, d[k], NX, NY)
        for j, a in enumerate(arms):
            arm_rec[i, j] = recover_band(prob, cx, cy, ex, ey, gxy, area,
                                         lm, float(a))[0]
    out.update(arm_a=arms, arm_rec=arm_rec)

    # ---- 8. recovery from band strain, with noise --------------------
    print("recovery ...", flush=True)
    rng = np.random.default_rng(0)
    noises = [0.0, 0.02, 0.05]
    rec = np.full((len(thetas), len(noises), 5), np.nan)
    fcurves = []
    for i, th in enumerate(thetas):
        k = f"u_{th:.2f}_{dl_r}"
        if k not in d:
            fcurves.append(np.full(71, np.nan)); continue
        lm = float(d[f"lam_{th:.2f}_{dl_r}"][0])
        cx, cy, ex, ey, gxy = element_strains(xy, d[k], NX, NY)
        scale = np.abs(ex[cy < BAND]).mean()
        r0_, g_, f_, _ = recover_band(prob, cx, cy, ex, ey, gxy, area,
                                      lm, 370.0)
        fcurves.append(f_)
        for j, nz in enumerate(noises):
            for r in range(5 if nz > 0 else 1):
                if nz == 0.0:
                    rec[i, j, :] = r0_
                    break
                e2 = [a + rng.normal(0.0, nz * scale, a.shape)
                      for a in (ex, ey, gxy)]
                rec[i, j, r] = recover_band(prob, cx, cy, *e2, area,
                                            lm, 370.0)[0]
        print(f"  theta {th:.2f}: no noise {r0_:.3f}, "
              f"2 % {np.nanmean(rec[i,1]):.3f}, 5 % {np.nanmean(rec[i,2]):.3f}",
              flush=True)
    out.update(rec_theta=np.array(thetas), rec_noise=np.array(noises),
               rec=rec, rec_grid=np.linspace(0.0, 0.70, 71),
               rec_f=np.array(fcurves))

    # ---- 8b. recovery under the three noise models --------------------
    # Table 2 quotes the identification under noise a fiber actually
    # gives: three models at one 5 % amplitude, fifty realizations each.
    # The figure has to be drawn from the very realizations the table
    # averages, so the generator in noise_study.py is called here with its
    # own seed and loop order and its output cached, rather than
    # re-simulated from a second stream that would drift from the printed
    # numbers. The import is deferred because noise_study imports this
    # module.
    print("noise models ...", flush=True)
    import noise_study
    nm_theta, nm_rec = noise_study.run_models(d)
    out.update(nm_theta=nm_theta, nm_rec=nm_rec,
               nm_models=np.array(noise_study.MODELS))
    for mi, mdl in enumerate(noise_study.MODELS):
        cells = "  ".join(
            f"{np.nanmean(nm_rec[mi, ti]):.3f}+-{np.nanstd(nm_rec[mi, ti]):.3f}"
            for ti in range(nm_theta.size))
        print(f"  {mdl:>11}: {cells}", flush=True)

    # ---- 9. recovery across load level, no noise ---------------------
    print("load levels ...", flush=True)
    rec_dl = np.full((len(thetas), len(deltas)), np.nan)
    for i, th in enumerate(thetas):
        for j, dl in enumerate(deltas):
            k = f"u_{th:.2f}_{dl:.1f}"
            if k not in d:
                continue
            lm = float(d[f"lam_{th:.2f}_{dl:.1f}"][0])
            cx, cy, ex, ey, gxy = element_strains(xy, d[k], NX, NY)
            rec_dl[i, j] = recover_band(prob, cx, cy, ex, ey, gxy, area,
                                        lm, 370.0)[0]
    out["rec_dl"] = rec_dl

    np.savez_compressed(OUT, **out)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
