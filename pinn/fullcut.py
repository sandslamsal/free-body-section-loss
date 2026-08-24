"""Candidate fix 4: identify from the full-cut first moment, not the band couple.

The ladder decomposition of the 27.8 pp cross-model offset (Task B) found
that its dominant component, about -18 pp, is the moment of the 90 to
104 kN of web tension the generator's field carries above the band, which
the band-couple recipe T.z drops by construction: the recipe integrates
sigma_x over y < 150 mm only, takes the compression centroid from the rest
of the cut, and never asks what tensile stress lives between the two. Only
about -10 pp of the offset is tension in the band itself.

The candidate observable is therefore the first moment of sigma_x over the
WHOLE cut depth,

    M_fc(theta) = - integral over 0 < y < H of sigma_x(theta) (y - y0) t dy,

on the same 100 mm strip at x = 700 mm, with sigma_x evaluated from the
measured strains through the same CSFM map with rho_x(theta) in the band,
reconciled against the same statics target M_req = lam P/2 (700 - 370)/1e6.
Wherever the map carries tension above the band, this integral picks up its
moment automatically, because it never separates the cut into band and rest.

THE PIVOT. The identity behind the observable is moment equilibrium of the
left free body about a point on the cut plane. The support reaction is
vertical through x_R, so its moment is the same about any pivot height, and
the axial force on the cut is exactly zero for this free body (the pin
carries the only horizontal reaction of the beam and whole-beam horizontal
equilibrium zeroes it), so in exact arithmetic every pivot height gives the
same equation. The pivots differ only in how MODEL error propagates:

  bearing level, y0 = 0     the mapped compressive stress magnitude enters
                            with an arm of ~870 mm, and the map's axial
                            closure error N (which reaches -365 kN on the
                            cross-model fields, because the map carries no
                            concrete tension) lands on the target in full;
                            the theta sensitivity collapses to the ~75 mm
                            arm from the tie band to the soffit.

  compression centroid,     the compressive stress magnitude drops out of
  y0 = yC(theta)            the moment identically (that is what a centroid
                            is), exactly the exposure the band couple had:
                            the compression side supplies a position, never
                            a magnitude. The theta sensitivity keeps the
                            full dT/dtheta times (yC - y_tie) of the band
                            couple.

This module therefore pivots at the compression centroid of the mapped
field, and demonstrates the bearing-level alternative numerically rather
than asserting it away.

What is tested, in order:
  1. same-solver CSFM fields, five states, delta 1.0 / 3.5 / 5.0 mm:
     the full cut must not be materially worse than the band couple where
     the band couple is already right;
  2. the fixed-crack Q4 40x20 family at 3.5 mm, against the band couple's
     slope 0.770 / intercept -0.278;
  3. the remainder decomposed by adding band tension: TCM of tcm.py and an
     average band stress beta f_ctm as in tension_chord.py, plus one
     diagnostic row with the beta stress over the whole cut depth, which
     locates how much of what remains is web tension the CSFM map cannot
     translate;
  4. affinity: dM_fc/dtheta on a fine grid, its constancy, the departure
     from the secant, and the ratio to the band couple's sensitivity;
  5. noise: the three 5 per cent protocols of noise_study.py, same seed,
     same draw order, both observables recovered from the SAME realization.

Everything else is the identification unchanged: 50 mm grid, CST
kinematics, cut at x = 700, 100 mm strip, 370 mm arm, wide trial range
[-0.70, 0.70], every root by recover_utils.bracket_root.

Run:  python fullcut.py            (writes figures/fullcut.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figdata as FD                                                      # noqa: E402
import noise_study                                                        # noqa: E402
import tcm as TCM                                                         # noqa: E402
from csfm_constitutive import membrane                                    # noqa: E402
from identify import rho_x_of_theta                                       # noqa: E402
from problem import DeepBeam                                              # noqa: E402
from recover_utils import bracket_root, element_strains                   # noqa: E402

ORACLE = HERE.parent / "oracle"
OUT = HERE.parent / "figures" / "fullcut.json"

ARM = 370.0                     # reaction centroid measured on the reference
THETA = [0.0, 0.10, 0.20, 0.30, 0.40]
WIDE = np.linspace(-0.70, 0.70, 141)         # cross_solver's trial grid
FINE = np.linspace(0.0, 0.70, 281)           # affinity grid, 0.0025 spacing
FCT_EN = 0.30 * 30.0 ** (2.0 / 3.0)          # EN 1992 f_ctm = 2.90 MPa
BETA_TS = 0.10                               # tension_chord.py's nominal


# ----------------------------------------------------------------------
# the observable
# ----------------------------------------------------------------------
def cut_sx(prob, cx, cy, ex, ey, gxy, theta, mode="none", beta=BETA_TS,
           f_ct=FCT_EN):
    """sigma_x on the whole cut strip through the CSFM map, with the chosen
    band-tension model added. mode "beta_full" is the diagnostic that adds
    the same average stress over the WHOLE depth wherever eps_x > 0."""
    sel = np.abs(cx - FD.X_CUT) < FD.BAND_W
    X = torch.tensor(cx[sel]).unsqueeze(-1)
    Y = torch.tensor(cy[sel]).unsqueeze(-1)
    rho = rho_x_of_theta(prob, X, Y, torch.tensor(float(theta)))
    st = membrane(torch.tensor(ex[sel]).unsqueeze(-1),
                  torch.tensor(ey[sel]).unsqueeze(-1),
                  torch.tensor(gxy[sel]).unsqueeze(-1),
                  rho, prob.rho_y(X, Y), prob.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy().copy()
    ys = cy[sel]
    inb = ys < FD.BAND
    if mode == "tcm":
        inc, _s, _sr, _ck = TCM.tcm_increment(
            ex[sel][inb], rho.squeeze().numpy()[inb], theta, prob.mat, f_ct,
            TCM.PHI_0, TCM.LAM_S)
        sx[inb] = sx[inb] + inc
    elif mode == "beta":
        sx[inb] = sx[inb] + beta * f_ct * (sx[inb] > 0.0)
    elif mode == "beta_full":
        sx = sx + beta * f_ct * (ex[sel] > 0.0)
    return sx, ys, ex[sel]


def fullcut_couple(prob, cx, cy, ex, ey, gxy, area, theta, mode="none",
                   beta=BETA_TS, f_ct=FCT_EN, pivot="comp"):
    """The full-cut first moment about the chosen pivot, in kN m.

    Pivoted at the compression centroid the compression cells contribute
    exactly zero, so the moment equals the couple of ALL tension on the cut
    (band and web alike) about that centroid; the band couple is the same
    number with the web tension deleted. Also returns the diagnostics the
    discussion needs: yC, the mapped axial closure N, and the web part.
    """
    sx, ys, _e = cut_sx(prob, cx, cy, ex, ey, gxy, theta, mode, beta, f_ct)
    dA = area / (2.0 * FD.BAND_W) * prob.t
    wC = np.clip(-sx, 0.0, None)
    yC = float((wC * ys).sum() / max(wC.sum(), 1e-9))
    y0 = yC if pivot == "comp" else 0.0
    M = float(-(sx * (ys - y0) * dA).sum()) / 1e6                   # kN m
    N = float((sx * dA).sum()) / 1e3                                # kN
    web = (ys >= FD.BAND) & (sx > 0.0)
    T_web = float((sx[web] * dA).sum()) / 1e3                       # kN
    M_web = float((sx[web] * (yC - ys[web]) * dA).sum()) / 1e6      # kN m
    return M, yC, N, T_web, M_web


def recover_fullcut(prob, s, area, lam, arm=ARM, grid=WIDE, **kw):
    """Root of the full-cut moment against statics, by bracket_root."""
    M_req = lam * prob.P / 2.0 * (FD.X_CUT - arm) / 1e6
    f = np.array([fullcut_couple(prob, *s, area, g, **kw)[0] - M_req
                  for g in grid])
    return bracket_root(f, grid), M_req, f


def recover_bandcouple(prob, s, area, lam, arm=ARM, grid=WIDE):
    """The baseline, verbatim: figdata's band couple on the same grid."""
    M_req = lam * prob.P / 2.0 * (FD.X_CUT - arm) / 1e6
    f = np.array([FD.band_couple(prob, *s, area, g)[2] - M_req
                  for g in grid])
    return bracket_root(f, grid), M_req, f


# ----------------------------------------------------------------------
# fits and family runs
# ----------------------------------------------------------------------
def affine(true, rec):
    """Slope and intercept of recovered against true, on the finite pairs."""
    t = np.array([a for a, b in zip(true, rec) if b is not None], float)
    r = np.array([b for b in rec if b is not None], float)
    out = {"n": int(t.size)}
    if t.size >= 2:
        s, i = np.polyfit(t, r, 1)
        out.update(slope=float(s), intercept=float(i))
    return out


def run_variant(states, prob, area, label, recover, **kw):
    rec, diag = [], []
    for th, s, lam in states:
        root, m_req, _f = recover(prob, s, area, lam, **kw)
        rec.append(None if not np.isfinite(root) else float(root))
        if recover is recover_fullcut:
            M, yC, N, T_web, M_web = fullcut_couple(
                prob, *s, area, th, **{k: v for k, v in kw.items()
                                       if k not in ("grid", "arm")})
            diag.append({"theta": th, "M_at_true_kNm": M, "M_req_kNm": m_req,
                         "yC_mm": yC, "N_kN": N, "T_web_kN": T_web,
                         "M_web_kNm": M_web})
    ok = [v for v in rec if v is not None]
    mono = len(ok) == len(rec) and all(b > a for a, b in zip(ok, ok[1:]))
    fit = affine([th for th, _, _ in states], rec)
    return {"label": label, "rec": rec, "fit": fit, "monotone": bool(mono),
            "n_admissible_nonneg": sum(1 for v in rec
                                       if v is not None and v >= 0.0),
            "diag_at_true": diag}


def load_same_solver(delta):
    d = np.load(ORACLE / "fields_theta.npz")
    return [(th, element_strains(d["xy"], d[f"u_{th:.2f}_{delta}"],
                                 FD.NX, FD.NY),
             float(d[f"lam_{th:.2f}_{delta}"][0])) for th in THETA]


def load_cross(tag="40x20", delta="3.5"):
    d = np.load(ORACLE / "fields_crossmodel.npz")
    return [(th, element_strains(d["xy"], d[f"u_{tag}_{th:.2f}_{delta}"],
                                 FD.NX, FD.NY),
             float(d[f"lam_{tag}_{th:.2f}_{delta}"][0])) for th in THETA]


def show(r):
    f = r["fit"]
    rec = "".join("     none" if v is None else f"{v:>+9.4f}"
                  for v in r["rec"])
    s = f.get("slope"); i = f.get("intercept")
    print(f"{r['label']:<34}{rec}   "
          f"slope {s if s is None else f'{s:6.3f}'}  "
          f"int {i if i is None else f'{i:+7.3f}'}  "
          f"adm {r['n_admissible_nonneg']}/5  "
          f"{'mono' if r['monotone'] else 'NOT mono'}")


# ----------------------------------------------------------------------
# affinity of the observable in theta
# ----------------------------------------------------------------------
def affinity_check(prob, s, area, label, mode="none", pivot="comp"):
    """dM/dtheta on the fine grid: level, constancy, secant departure."""
    if mode == "band":
        M = np.array([FD.band_couple(prob, *s, area, g)[2] for g in FINE])
    else:
        M = np.array([fullcut_couple(prob, *s, area, g, mode=mode,
                                     pivot=pivot)[0] for g in FINE])
    dM = np.gradient(M, FINE)
    chord = M[0] + (M[-1] - M[0]) * (FINE - FINE[0]) / (FINE[-1] - FINE[0])
    dep = float(np.abs(M - chord).max() / max(abs(M[-1] - M[0]), 1e-12))
    out = {"label": label,
           "dM_mean_kNm": float(dM.mean()), "dM_min_kNm": float(dM.min()),
           "dM_max_kNm": float(dM.max()),
           "constancy_spread": float((dM.max() - dM.min())
                                     / max(abs(dM.mean()), 1e-12)),
           "secant_departure_rel": dep}
    print(f"  {label:<34} dM/dtheta {dM.mean():8.1f} kN m "
          f"[{dM.min():8.1f}, {dM.max():8.1f}]  "
          f"spread {out['constancy_spread']:.2e}  "
          f"secant dep {dep:.2e}")
    return out


# ----------------------------------------------------------------------
# noise, mirroring noise_study.py draw for draw
# ----------------------------------------------------------------------
def run_noise(d):
    """The three 5 per cent protocols, one stream seeded 0 consumed in the
    loop order of noise_study.run_models, with the band couple and the full
    cut recovered from the SAME perturbed realization. The band-couple
    numbers therefore reproduce Table 2 exactly, and the comparison is
    paired rather than merely same-protocol."""
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    thetas = [float(t) for t in d["theta_true"]]
    grid = np.linspace(0.0, 0.70, 71)           # noise_study's trial grid
    rng = np.random.default_rng(0)
    rec = np.full((2, len(noise_study.MODELS), len(thetas),
                   noise_study.N_REAL), np.nan)
    for mi, model in enumerate(noise_study.MODELS):
        for ti, th in enumerate(thetas):
            k = f"u_{th:.2f}_{noise_study.DELTA}"
            if k not in d.files:
                continue
            lam = float(d[f"lam_{th:.2f}_{noise_study.DELTA}"][0])
            cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k],
                                                  FD.NX, FD.NY)
            scale = float(np.abs(ex[cy < FD.BAND]).mean())
            for j in range(noise_study.N_REAL):
                sd = 0.05 * scale
                if model == "correlated":
                    pert = [a + noise_study.correlated(rng, cx, cy, sd)
                            for a in (ex, ey, gxy)]
                else:
                    pert = [a + rng.normal(0.0, sd, a.shape)
                            for a in (ex, ey, gxy)]
                keep = np.ones(cx.size, bool)
                if model == "dropout":
                    keep[rng.choice(cx.size, int(0.15 * cx.size),
                                    replace=False)] = False
                s = (cx[keep], cy[keep], pert[0][keep], pert[1][keep],
                     pert[2][keep])
                rec[0, mi, ti, j] = recover_bandcouple(
                    prob, s, area, lam, grid=grid)[0]
                rec[1, mi, ti, j] = recover_fullcut(
                    prob, s, area, lam, grid=grid)[0]
        print(f"  {model} done", flush=True)
    return np.array(thetas), rec


# ----------------------------------------------------------------------
def main() -> None:
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    res = {"families": {}, "affinity": {}, "pivot_demo": {}, "noise": {}}

    fams = [("same solver, 1.0 mm", load_same_solver("1.0")),
            ("same solver, 3.5 mm", load_same_solver("3.5")),
            ("same solver, 5.0 mm", load_same_solver("5.0")),
            ("fixed crack Q4 40x20, 3.5 mm", load_cross())]

    variants = [
        ("band couple (baseline)", recover_bandcouple, {}),
        ("full cut", recover_fullcut, {}),
        ("full cut + TCM (f_ctm EN)", recover_fullcut, {"mode": "tcm"}),
        (f"full cut + beta={BETA_TS:.2f} f_ctm band",
         recover_fullcut, {"mode": "beta"}),
        (f"full cut + beta={BETA_TS:.2f} f_ctm FULL DEPTH",
         recover_fullcut, {"mode": "beta_full"}),
    ]

    for fname, st in fams:
        print(f"\n{'=' * 110}\n{fname}   (theta = 0, .10, .20, .30, .40; "
              f"wide grid [-0.70, 0.70]; arm 370 mm)\n")
        rows = []
        for label, rec_fn, kw in variants:
            rows.append(run_variant(st, prob, area, label, rec_fn, **kw))
            show(rows[-1])
        res["families"][fname] = rows
        if "fixed crack" in fname or "3.5" in fname:
            dg = rows[1]["diag_at_true"]
            print("\n  full-cut diagnostics at true theta:")
            for q in dg:
                print(f"    theta {q['theta']:.2f}: M_fc {q['M_at_true_kNm']:7.1f}"
                      f" vs M_req {q['M_req_kNm']:6.1f} kN m,  yC {q['yC_mm']:4.0f} mm,"
                      f"  N {q['N_kN']:+7.1f} kN,  web T {q['T_web_kN']:5.1f} kN,"
                      f"  web M {q['M_web_kNm']:5.1f} kN m")

    # ---- the pivot, demonstrated -------------------------------------
    print(f"\n{'=' * 110}\npivot demonstration, fixed-crack family, "
          f"theta_true = 0.20\n")
    th, s, lam = fams[3][1][2]
    M_req = lam * prob.P / 2.0 * (FD.X_CUT - ARM) / 1e6
    for pivot, tag in (("comp", "compression centroid"),
                       ("bearing", "bearing level y0 = 0")):
        M, yC, N, _tw, _mw = fullcut_couple(prob, *s, area, th, pivot=pivot)
        Mf = np.array([fullcut_couple(prob, *s, area, g, pivot=pivot)[0]
                       for g in FINE])
        dM = float(np.gradient(Mf, FINE).mean())
        root = bracket_root(np.array(
            [fullcut_couple(prob, *s, area, g, pivot=pivot)[0] - M_req
             for g in WIDE]), WIDE)
        res["pivot_demo"][tag] = {
            "M_at_true_kNm": M, "M_req_kNm": M_req, "N_kN": N,
            "dM_dtheta_kNm": dM,
            "theta_rec": None if not np.isfinite(root) else float(root)}
        print(f"  {tag:<24} M(0.20) {M:8.1f} vs M_req {M_req:6.1f} kN m,  "
              f"dM/dtheta {dM:8.1f} kN m,  rec "
              f"{'none' if not np.isfinite(root) else f'{root:+.3f}'}")

    # ---- affinity ----------------------------------------------------
    print(f"\n{'=' * 110}\naffinity in theta on [0, 0.70], grid spacing "
          f"{FINE[1] - FINE[0]:.4f}\n")
    for fname, st in ((fams[1][0], fams[1][1]), (fams[3][0], fams[3][1])):
        th, s, lam = st[2]                       # theta_true = 0.20 state
        print(f"{fname}, theta_true = 0.20:")
        res["affinity"][fname] = [
            affinity_check(prob, s, area, "band couple", mode="band"),
            affinity_check(prob, s, area, "full cut, comp pivot"),
            affinity_check(prob, s, area, "full cut, bearing pivot",
                           pivot="bearing"),
            affinity_check(prob, s, area, "full cut + TCM", mode="tcm")]

    # ---- noise -------------------------------------------------------
    print(f"\n{'=' * 110}\nnoise: {noise_study.N_REAL} realizations, 5 %, "
          f"three models, seed 0, draw order of noise_study.py\n")
    d = np.load(ORACLE / "fields_theta.npz")
    thetas, rec = run_noise(d)
    obs_names = ("band couple", "full cut")
    for mi, model in enumerate(noise_study.MODELS):
        for oi, oname in enumerate(obs_names):
            cells = []
            for ti in range(thetas.size):
                got = rec[oi, mi, ti][np.isfinite(rec[oi, mi, ti])]
                cells.append(f"{got.mean():.3f}+-{got.std():.3f}({got.size})"
                             if got.size else "none")
            print(f"  {model:>11} {oname:<12}"
                  + "".join(f"{c:>20}" for c in cells))
    res["noise"] = {
        "models": list(noise_study.MODELS), "theta": thetas.tolist(),
        "n_real": noise_study.N_REAL,
        "mean": {o: np.nanmean(rec[oi], axis=-1).tolist()
                 for oi, o in enumerate(obs_names)},
        "sd": {o: np.array([[np.nanstd(rec[oi, mi, ti])
                             for ti in range(thetas.size)]
                            for mi in range(len(noise_study.MODELS))]).tolist()
               for oi, o in enumerate(obs_names)},
        "n_finite": {o: np.isfinite(rec[oi]).sum(axis=-1).tolist()
                     for oi, o in enumerate(obs_names)}}

    # ---- summary: what the change of observable buys -----------------
    bc = res["families"]["fixed crack Q4 40x20, 3.5 mm"][0]["fit"]
    fc = res["families"]["fixed crack Q4 40x20, 3.5 mm"][1]["fit"]
    tc = res["families"]["fixed crack Q4 40x20, 3.5 mm"][2]["fit"]
    bt = res["families"]["fixed crack Q4 40x20, 3.5 mm"][3]["fit"]
    bf = res["families"]["fixed crack Q4 40x20, 3.5 mm"][4]["fit"]
    print(f"\n{'=' * 110}\ncross-model intercept, in units of theta "
          f"(x100 = pp):")
    for name, f in (("band couple", bc), ("full cut", fc),
                    ("full cut + TCM", tc),
                    ("full cut + beta band", bt),
                    ("full cut + beta full depth (diagnostic)", bf)):
        if "intercept" in f:
            print(f"  {name:<42} slope {f['slope']:6.3f}   "
                  f"intercept {f['intercept']:+.3f}")

    res["settings"] = {
        "arm_mm": ARM, "x_cut_mm": FD.X_CUT, "band_mm": FD.BAND,
        "strip_halfwidth_mm": FD.BAND_W, "pivot": "compression centroid",
        "wide_grid": [float(WIDE[0]), float(WIDE[-1]), int(WIDE.size)],
        "fine_grid": [float(FINE[0]), float(FINE[-1]), int(FINE.size)],
        "beta_ts": BETA_TS, "f_ctm_EN_MPa": FCT_EN,
        "tcm": {"f_ct_MPa": FCT_EN, "phi0_mm": TCM.PHI_0,
                "lam_s": TCM.LAM_S}}
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n-> {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
