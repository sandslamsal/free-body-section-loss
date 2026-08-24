"""Identification without the reaction line of action, by differencing two cuts.

The single-cut condition reconciles the band couple against statics as
C(x_c; theta) = R (x_c - a), and a, the line of action of the reaction
inside its bearing plate, is not measured. It is the dominant error source
in this study: the recovered section loss moves about 0.25 percentage
points for every millimeter the assumed arm is wrong, so a plate that is
200 mm wide carries a 25 pp ambiguity unless the contact centroid is
pinned down. That is worth removing algebraically rather than bounding.

Two cuts in the same shear span with no intervening load carry the same
reaction, so subtracting their conditions gives

    C(x_2; theta) - C(x_1; theta) = R (x_2 - x_1)

in which a does not appear. The difference of two functions each affine in
theta is still affine, so a root survives. What differencing costs is
conditioning: the slope of the differenced observable is the difference of
two slopes, and it collapses as the cuts approach each other. This script
measures the exchange rate rather than assuming it, on the stored fields.

Five things are reported, in order:

  1. invariance   the two-cut root against an assumed arm swept 250 to 450
                  mm, beside the single-cut root over the same sweep
  2. recovery     all five deterioration states, two-cut against single-cut
                  at the measured reaction centroid
  3. conditioning |dC/dtheta| for one cut and for the difference, against
                  cut spacing, and the amplification that implies
  4. noise        5 % strain noise, independent and with a 150 mm
                  exponential correlation length, generated exactly as in
                  noise_study.py, standard deviation against cut spacing
  5. the trade    the factor k by which the arm-free form pays for it

Run:  python arm_free.py       (writes ../figures/arm_free.json)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

from arclength_oracle import build_mesh                                    # noqa: E402
from csfm_constitutive import membrane                                     # noqa: E402
from identify import rho_x_of_theta                                        # noqa: E402
from noise_study import CORR_LEN, correlated                               # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                         # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_nodal import internal_forces                                  # noqa: E402
from recover_utils import bracket_root, element_strains                    # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "arm_free.json"

NX, NY = 40, 20            # structured mesh, as everywhere in this study
BAND = 150.0               # tie-band depth (mm)
BAND_W = 50.0              # half-width of the strip that stands for a cut
DELTA = "3.5"              # load level the identification is posed at

# The cut strips have to lie between the support plate and the load plate
# and touch neither. build_mesh fixes the soffit nodes within 100 mm of
# x = 250, so the reaction is delivered over 150 to 350 mm; the load nodes
# run from 900 to 1100 mm. The clear span is therefore 350 to 900 mm, and a
# strip 2*BAND_W wide centered on x_c occupies x_c +- 50. Admissible cut
# stations are the mesh lines from 400 to 850 mm, which caps the spacing of
# any admissible pair at 450 mm on this member.
X_CLEAR = (350.0, 900.0)
STATIONS = tuple(float(x) for x in range(400, 900, 50))
X_PAPER = 700.0            # the single-cut station this study reports at
PAIR = (400.0, 850.0)      # widest admissible pair, the best conditioned one

# Roots are taken on a grid that reaches well outside the admissible range
# of theta. A censored root cannot be averaged, and both the offset of the
# differenced form and its noise push realizations past zero; truncating
# them would report the truncation rather than the spread. The couple is
# monotone in theta over the whole of this range at every station, so the
# first sign change is the root, and the published recovery is reproduced
# to 1e-5 on it. How often a realization lands outside [0, 0.70] is
# reported separately, because that is a real failure of the method.
GRID = np.linspace(-1.0, 1.0, 101)
THETA_ADMISSIBLE = (0.0, 0.70)

H_FD = 0.02                # central-difference step in theta
NOISE = 0.05               # strain noise amplitude, as a fraction of the
                           # mean band strain, matching noise_study.py
N_REAL = 50
MODELS = ("independent", "correlated")


# ----------------------------------------------------------------------
# the observable, at an arbitrary cut station
# ----------------------------------------------------------------------
def band_couple_at(prob, cx, cy, ex, ey, gxy, area, x_cut, theta):
    """figdata.band_couple with the cut station made an argument.

    Identical arithmetic: the band supplies T, axial equilibrium of the cut
    makes the compression resultant equal to it, and the arm is the
    distance between the two stress centroids. Only the station moves, so
    a two-cut condition and the published one-cut condition are built from
    the same observable and any difference between them is the differencing,
    not a change of measurement.
    """
    sel = np.abs(cx - x_cut) < BAND_W
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
    inb = ys < BAND
    T = float((sx[inb] * dA).sum()) / 1e3                            # kN
    wT = np.clip(sx[inb], 0.0, None)
    yT = float((wT * ys[inb]).sum() / max(wT.sum(), 1e-9))
    wC = np.clip(-sx[~inb], 0.0, None)
    yC = float((wC * ys[~inb]).sum() / max(wC.sum(), 1e-9))
    return T, yC - yT, T * (yC - yT) / 1e3                           # kN m


def couple_curves(prob, cx, cy, ex, ey, gxy, area, stations=STATIONS,
                  grid=GRID):
    """C(x_c; theta) on the trial grid, one row per cut station (kN m)."""
    return np.array([[band_couple_at(prob, cx, cy, ex, ey, gxy, area,
                                     s, g)[2] for g in grid]
                     for s in stations])


# ----------------------------------------------------------------------
# the two identifying conditions
# ----------------------------------------------------------------------
def root_single(C, R, x_cut, a, grid=GRID):
    """Root of C(x_c; theta) - R (x_c - a), the published condition."""
    return bracket_root(C - R * (x_cut - a) / 1e6, grid)


def root_two(C1, C2, R, x1, x2, grid=GRID):
    """Root of [C(x_2) - C(x_1)] - R (x_2 - x_1), with a canceled.

    Nothing in this residual refers to the line of action, so the root is
    independent of it by construction and not merely insensitive to it.
    """
    return bracket_root((C2 - C1) - R * (x2 - x1) / 1e6, grid)


def slope(prob, cx, cy, ex, ey, gxy, area, x_cut, theta, h=H_FD):
    """dC/dtheta at one cut, by central difference (kN m per unit theta)."""
    p = band_couple_at(prob, cx, cy, ex, ey, gxy, area, x_cut, theta + h)[2]
    m = band_couple_at(prob, cx, cy, ex, ey, gxy, area, x_cut, theta - h)[2]
    return (p - m) / (2.0 * h)


def reaction_centroid(d, th):
    """Measured line of action of the left reaction, from the solved field.

    This is the quantity the single-cut condition needs and a real test
    does not have: the assembled reaction distributed over the bearing
    nodes, reduced to its first moment. figdata caches the same number.
    """
    pr = deepbeam_rho(RHO_NOM * (1.0 - th))
    mh = build_mesh(pr)
    lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
    Rv = internal_forces(d[f"u_{th:.2f}_{DELTA}"].ravel(), pr, mh, th) \
        - lam * mh.F_ref
    fx = np.asarray(mh.fixed, bool)
    xs, rs = [], []
    for n in range(mh.n_node):
        if fx[2 * n + 1] and mh.xy[n, 0] < 600.0:
            xs.append(mh.xy[n, 0])
            rs.append(Rv[2 * n + 1] / 1e3)
    xs, rs = np.array(xs), np.array(rs)
    return float((xs * rs).sum() / rs.sum()), float(rs.sum())


def pairs_at(spacing):
    """Every admissible ordered cut pair with the given spacing (mm)."""
    return [(a, b) for a in STATIONS for b in STATIONS
            if abs((b - a) - spacing) < 1e-9]


def affine(theta, rec):
    """Least-squares slope and intercept of recovered against true theta."""
    ok = np.isfinite(rec)
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    p = np.polyfit(theta[ok], rec[ok], 1)
    resid = rec[ok] - np.polyval(p, theta[ok])
    return float(p[0]), float(p[1]), float(np.abs(resid).max())


# ----------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / NX) * (prob.H / NY) / 2.0
    thetas = [float(t) for t in d["theta_true"]]
    out: dict = {
        "geometry": {
            "L_mm": prob.L, "H_mm": prob.H, "t_mm": prob.t,
            "support_x_mm": list(prob.x_supp), "load_x_mm": prob.x_load,
            "bearing_mm": prob.bearing, "band_mm": BAND,
            "rho_tie": prob.rho_tie,
            "reaction_delivered_over_mm": [150.0, 350.0],
            "load_delivered_over_mm": [900.0, 1100.0],
            "clear_span_mm": list(X_CLEAR),
            "cut_half_width_mm": BAND_W,
            "admissible_stations_mm": list(STATIONS),
            "max_admissible_spacing_mm": max(STATIONS) - min(STATIONS),
        },
        "delta_mm": float(DELTA), "grid": list(GRID[[0, -1]]) + [len(GRID)],
    }

    # cache the strains and the couple curves at every station, per state
    print("couple curves at every admissible station ...", flush=True)
    state = {}
    for th in thetas:
        k = f"u_{th:.2f}_{DELTA}"
        if k not in d.files:
            continue
        lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
        cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], NX, NY)
        cent, R_asm = reaction_centroid(d, th)
        state[th] = {
            "lam": lam, "R": lam * prob.P / 2.0, "cent": cent,
            "R_assembled_kN": R_asm,
            "strain": (cx, cy, ex, ey, gxy),
            "scale": float(np.abs(ex[cy < BAND]).mean()),
            "C": couple_curves(prob, cx, cy, ex, ey, gxy, area),
        }
    idx = {s: i for i, s in enumerate(STATIONS)}
    ref = 0.20                                   # the state everything is
    st_r = state[ref]                            # verified on

    # ------------------------------------------------------------------
    # 1. is the two-cut root really free of the arm
    # ------------------------------------------------------------------
    print("\n== 1. invariance to the assumed arm, "
          f"theta = {ref:.2f}, delta = {DELTA} mm ==")
    arms = np.linspace(250.0, 450.0, 41)
    R = st_r["R"]
    C_paper = st_r["C"][idx[X_PAPER]]
    C1, C2 = st_r["C"][idx[PAIR[0]]], st_r["C"][idx[PAIR[1]]]
    r_single = np.array([root_single(C_paper, R, X_PAPER, float(a))
                         for a in arms])
    r_two = np.array([root_two(C1, C2, R, *PAIR) for _ in arms])
    ok = np.isfinite(r_single)
    p = np.polyfit(arms[ok], r_single[ok], 1)
    pp_per_mm = float(p[0] * 100.0)
    spread_two = float(np.nanmax(r_two) - np.nanmin(r_two))
    print(f"  {'a (mm)':>9}{'single cut':>13}{'two cut':>18}")
    for a, rs, rt in list(zip(arms, r_single, r_two))[::5]:
        print(f"  {a:>9.0f}{rs:>13.4f}{rt:>18.9f}")
    print(f"  single-cut sensitivity to the assumed arm: "
          f"{pp_per_mm:+.3f} pp/mm")
    print(f"  two-cut spread over the whole sweep:       {spread_two:.3e}")
    if spread_two > 1e-12:
        raise RuntimeError(
            f"two-cut root moved by {spread_two:.3e} over the arm sweep; "
            f"the arm has not canceled, so the implementation is wrong")
    print("  the arm has canceled exactly, not approximately")
    out["invariance"] = {
        "theta": ref, "a_mm": arms.tolist(),
        "single_cut_station_mm": X_PAPER, "two_cut_pair_mm": list(PAIR),
        "root_single": r_single.tolist(), "root_two": r_two.tolist(),
        "single_pp_per_mm": pp_per_mm, "two_spread": spread_two,
    }

    # ------------------------------------------------------------------
    # 2. recovery at every state
    # ------------------------------------------------------------------
    print("\n== 2. recovery, all five states ==")
    print(f"  {'theta':>7}{'cent (mm)':>11}{'1cut a=370':>12}{'bias':>8}"
          f"{'1cut a=cent':>13}{'bias':>8}{'2cut':>10}{'bias':>8}")
    rec = {"theta_true": [], "centroid_mm": [], "lam": [], "R_kN": [],
           "single_a370": [], "single_acent": [], "two_cut": []}
    for th in thetas:
        s = state[th]
        Cp = s["C"][idx[X_PAPER]]
        r370 = root_single(Cp, s["R"], X_PAPER, 370.0)
        rcen = root_single(Cp, s["R"], X_PAPER, s["cent"])
        rtwo = root_two(s["C"][idx[PAIR[0]]], s["C"][idx[PAIR[1]]],
                        s["R"], *PAIR)
        print(f"  {th:>7.2f}{s['cent']:>11.2f}"
              f"{r370:>12.4f}{(r370 - th) * 100:>+8.2f}"
              f"{rcen:>13.4f}{(rcen - th) * 100:>+8.2f}"
              f"{rtwo:>10.4f}{(rtwo - th) * 100:>+8.2f}")
        rec["theta_true"].append(th)
        rec["centroid_mm"].append(s["cent"])
        rec["lam"].append(s["lam"])
        rec["R_kN"].append(s["R"] / 1e3)
        rec["single_a370"].append(r370)
        rec["single_acent"].append(rcen)
        rec["two_cut"].append(rtwo)
    # The single-cut column at a = 370 mm is the published recovery, and it
    # is the only number in this script that already has a value to agree
    # with. Checking it here is what makes the rest of the table readable
    # as a change of condition rather than a change of implementation.
    tt = np.array(rec["theta_true"])
    published = {0.10: 0.0741, 0.20: 0.1569, 0.30: 0.2385, 0.40: 0.3250}
    for th, want in published.items():
        got = rec["single_a370"][rec["theta_true"].index(th)]
        if abs(got - want) > 1e-4:
            raise RuntimeError(
                f"single cut at a = 370 gives {got:.4f} at theta = {th:.2f}, "
                f"against the published {want:.4f}; the observable has "
                f"changed and nothing below is comparable")
    print("  the a = 370 column reproduces the published recovery "
          "(0.0741, 0.1569, 0.2385, 0.3250)")
    fits = {}
    for name in ("single_a370", "single_acent", "two_cut"):
        a1, a0, mx = affine(tt, np.array(rec[name]))
        fits[name] = {"slope": a1, "intercept": a0, "max_resid": mx}
        print(f"  {name:>13}: {a1:.4f}*theta {a0:+.4f}   "
              f"max residual {mx:.4f}")
    print("  the arm assumption buys a small intercept at the price of the "
          "slope; differencing does the reverse")
    out["recovery"] = rec
    out["affine_fits"] = fits

    # ------------------------------------------------------------------
    # 3. what differencing costs in conditioning
    # ------------------------------------------------------------------
    print(f"\n== 3. conditioning, theta = {ref:.2f} ==")
    cxr, cyr, exr, eyr, gxr = st_r["strain"]
    sl = {s: slope(prob, cxr, cyr, exr, eyr, gxr, area, s, ref)
          for s in STATIONS}
    print(f"  {'station (mm)':>13}{'|dC/dtheta| (kN m)':>21}")
    for s in STATIONS:
        print(f"  {s:>13.0f}{abs(sl[s]):>21.2f}")
    spacings = [float(x) for x in range(100, 650, 50)]
    cond = []
    print(f"\n  {'spacing':>8}{'pairs':>7}{'|d(dC)/dtheta|':>16}"
          f"{'range':>18}{'k vs cut 850':>14}{'k vs cut 700':>14}")
    for sp in spacings:
        prs = pairs_at(sp)
        if not prs:
            print(f"  {sp:>8.0f}{0:>7}{'--':>16}{'not admissible':>18}"
                  f"{'--':>14}{'--':>14}")
            cond.append({"spacing_mm": sp, "n_pairs": 0,
                         "admissible": False})
            continue
        ds = np.array([abs(sl[b] - sl[a]) for a, b in prs])
        k850 = abs(sl[max(STATIONS)]) / ds.mean()
        k700 = abs(sl[X_PAPER]) / ds.mean()
        print(f"  {sp:>8.0f}{len(prs):>7}{ds.mean():>16.2f}"
              f"{f'{ds.min():.1f} to {ds.max():.1f}':>18}"
              f"{k850:>14.2f}{k700:>14.2f}")
        cond.append({
            "spacing_mm": sp, "n_pairs": len(prs), "admissible": True,
            "pairs": [list(p) for p in prs],
            "slope_diff_mean": float(ds.mean()),
            "slope_diff_min": float(ds.min()),
            "slope_diff_max": float(ds.max()),
            "k_vs_station_850": float(k850),
            "k_vs_station_700": float(k700),
            "k_analytic_indep_vs_850": float(np.sqrt(2.0) * k850),
        })
    print(f"  spacings above {max(STATIONS) - min(STATIONS):.0f} mm do not "
          f"exist on this member: the clear span between the plate edges is "
          f"{X_CLEAR[1] - X_CLEAR[0]:.0f} mm")
    print("  k is the deterministic slope ratio; with independent couple "
          "errors the noise factor is sqrt(2) k")
    out["conditioning"] = {
        "theta": ref,
        "slope_single": {f"{s:.0f}": float(sl[s]) for s in STATIONS},
        "spacings": cond,
    }

    # ------------------------------------------------------------------
    # 4. noise
    # ------------------------------------------------------------------
    print(f"\n== 4. noise, {NOISE:.0%} of mean band strain, "
          f"{N_REAL} realizations ==")
    print(f"  correlated model: exponential covariance, "
          f"correlation length {CORR_LEN:.0f} mm")
    # One stream, seeded once and consumed in a fixed loop order, exactly
    # the convention of noise_study.py. Every station and every spacing is
    # read off the same realization, so the comparison between the two
    # forms is paired and the spread is not the spread of the draw.
    rng = np.random.default_rng(0)
    roots_s = np.full((len(MODELS), len(thetas), N_REAL, len(STATIONS)),
                      np.nan)
    all_pairs = [p for sp in spacings for p in pairs_at(sp)]
    # the noiseless root of every condition, so that the shift noise puts
    # on the answer can be separated from the spread it puts around it
    clean_s = np.array([[root_single(state[th]["C"][si], state[th]["R"],
                                     stn, state[th]["cent"])
                         for si, stn in enumerate(STATIONS)]
                        for th in thetas])
    clean_d = {p: np.array([root_two(state[th]["C"][idx[p[0]]],
                                     state[th]["C"][idx[p[1]]],
                                     state[th]["R"], p[0], p[1])
                            for th in thetas]) for p in all_pairs}
    roots_d = {p: np.full((len(MODELS), len(thetas), N_REAL), np.nan)
               for p in all_pairs}
    for mi, model in enumerate(MODELS):
        for ti, th in enumerate(thetas):
            s = state[th]
            cx, cy, ex, ey, gxy = s["strain"]
            sd = NOISE * s["scale"]
            for j in range(N_REAL):
                if model == "correlated":
                    pert = [a + correlated(rng, cx, cy, sd)
                            for a in (ex, ey, gxy)]
                else:
                    pert = [a + rng.normal(0.0, sd, a.shape)
                            for a in (ex, ey, gxy)]
                Cn = couple_curves(prob, cx, cy, pert[0], pert[1], pert[2],
                                   area)
                for si, stn in enumerate(STATIONS):
                    roots_s[mi, ti, j, si] = root_single(
                        Cn[si], s["R"], stn, s["cent"])
                for (a, b) in all_pairs:
                    roots_d[(a, b)][mi, ti, j] = root_two(
                        Cn[idx[a]], Cn[idx[b]], s["R"], a, b)
            print(f"  {model:>12} theta {th:.2f} done "
                  f"[{time.time() - t_start:.0f} s]", flush=True)

    def spread(v):
        """Standard deviation over realizations, averaged over the states.

        Pooling the five states directly would fold the state-to-state
        offset into the spread, which is a bias and not a noise, so each
        state is reduced first and the reductions are averaged.
        """
        v = np.asarray(v)
        per = [np.std(v[ti][np.isfinite(v[ti])])
               for ti in range(v.shape[0]) if np.isfinite(v[ti]).sum() > 1]
        return float(np.mean(per)) if per else float("nan")

    def shift(v, clean):
        """Mean displacement noise puts on the answer, averaged over states.

        Noise does not only spread the estimate. The couple is nonlinear in
        strain through the softening law and through the clipping that
        defines the two stress centroids, so a zero-mean strain error does
        not give a zero-mean couple error, and the mean matters more than
        the spread at the cuts near the support.
        """
        v = np.asarray(v)
        per = [np.mean(v[ti][np.isfinite(v[ti])]) - clean[ti]
               for ti in range(v.shape[0])
               if np.isfinite(v[ti]).any() and np.isfinite(clean[ti])]
        return float(np.mean(per)) if per else float("nan")

    def censored(v):
        """Fraction of realizations whose root is outside [0, 0.70]."""
        v = np.asarray(v).ravel()
        lo, hi = THETA_ADMISSIBLE
        return float(np.mean(~((v >= lo) & (v <= hi))))

    noise_out = {"models": list(MODELS), "n_real": N_REAL,
                 "amplitude": NOISE, "corr_len_mm": CORR_LEN,
                 "thetas": thetas, "single": {}, "two": []}
    sd_single = {}
    print("\n  single cut, arm set to the measured centroid")
    print(f"  {'station':>8}" + "".join(
        f"{m + ' sd':>15}{'shift pp':>10}{'sigma_C':>9}{'cens':>7}"
        for m in MODELS))
    for si, stn in enumerate(STATIONS):
        row, cells = {"station_mm": stn, "sd": [], "shift": [],
                      "sigma_C_kNm": [], "censored": []}, ""
        for mi in range(len(MODELS)):
            v = roots_s[mi, :, :, si]
            sdv = spread(v)
            shf = shift(v, clean_s[:, si])
            sig = sdv * abs(sl[stn])
            cen = censored(v)
            row["sd"].append(sdv)
            row["shift"].append(shf)
            row["sigma_C_kNm"].append(sig)
            row["censored"].append(cen)
            cells += (f"{sdv:>15.4f}{shf * 100:>+10.1f}"
                      f"{sig:>9.2f}{cen:>7.2f}")
        print(f"  {stn:>8.0f}" + cells)
        row["sd_correlated_over_independent"] = (
            row["sd"][1] / row["sd"][0] if row["sd"][0] else float("nan"))
        sd_single[stn] = row["sd"]
        noise_out["single"][f"{stn:.0f}"] = row
    print("  sigma_C is the couple error the spread implies, sd times the "
          "slope at that station: the stations near the support are the "
          "noisy ones, because their tie force is small")

    print("\n  two cut, averaged over the admissible pairs at each spacing")
    print(f"  {'spacing':>8}{'pairs':>6}" + "".join(
        f"{m + ' sd':>13}{'shift pp':>10}{'k_far':>7}{'k_700':>7}{'cens':>6}"
        for m in MODELS) + f"{'corr/ind':>8}")
    for sp in spacings:
        prs = pairs_at(sp)
        if not prs:
            continue
        row = {"spacing_mm": sp, "n_pairs": len(prs), "sd": [], "shift": [],
               "k_matched_far_cut": [], "k_vs_station_700": [],
               "censored": [], "sd_by_pair": []}
        cells = ""
        for mi in range(len(MODELS)):
            sds = [spread(roots_d[p][mi]) for p in prs]
            kf = [sds[i] / sd_single[p[1]][mi] for i, p in enumerate(prs)]
            sdv = float(np.mean(sds))
            shf = float(np.mean([shift(roots_d[p][mi], clean_d[p])
                                 for p in prs]))
            cen = float(np.mean([censored(roots_d[p][mi]) for p in prs]))
            row["sd"].append(sdv)
            row["shift"].append(shf)
            row["k_matched_far_cut"].append(float(np.mean(kf)))
            row["k_vs_station_700"].append(sdv / sd_single[X_PAPER][mi])
            row["censored"].append(cen)
            row["sd_by_pair"].append(
                {f"{p[0]:.0f}-{p[1]:.0f}": sds[i]
                 for i, p in enumerate(prs)})
            cells += (f"{sdv:>13.4f}{shf * 100:>+10.1f}"
                      f"{np.mean(kf):>7.2f}"
                      f"{sdv / sd_single[X_PAPER][mi]:>7.2f}{cen:>6.2f}")
        row["sd_correlated_over_independent"] = row["sd"][1] / row["sd"][0]
        print(f"  {sp:>8.0f}{len(prs):>6}" + cells
              + f"{row['sd_correlated_over_independent']:>8.2f}")
        noise_out["two"].append(row)
    print("  k_far compares each pair against the single cut at its own far "
          "station, so it isolates the differencing; k_700 compares against "
          "the station this study reports at")
    out["noise"] = noise_out

    # best arm-known layout against best arm-free layout, which is the
    # comparison a designer actually faces
    best = {}
    for mi, model in enumerate(MODELS):
        b_s = min(STATIONS, key=lambda st: sd_single[st][mi])
        pair_sd = {p: spread(roots_d[p][mi]) for p in all_pairs}
        b_d = min(pair_sd, key=lambda p: pair_sd[p])
        best[model] = {
            "best_single_station_mm": b_s,
            "best_single_sd": sd_single[b_s][mi],
            "best_pair_mm": list(b_d),
            "best_pair_spacing_mm": b_d[1] - b_d[0],
            "best_pair_sd": pair_sd[b_d],
            "k_best_vs_best": pair_sd[b_d] / sd_single[b_s][mi],
            "k_vs_paper_station": pair_sd[b_d] / sd_single[X_PAPER][mi],
        }
        print(f"\n  {model}: best single cut is station "
              f"{b_s:.0f} mm at sd {sd_single[b_s][mi]:.4f}; best pair is "
              f"{b_d[0]:.0f}-{b_d[1]:.0f} mm at sd {pair_sd[b_d]:.4f}")
        print(f"    k best against best = "
              f"{best[model]['k_best_vs_best']:.2f}; against the "
              f"{X_PAPER:.0f} mm station this study reports at, "
              f"{best[model]['k_vs_paper_station']:.2f}")
    noise_out["best"] = best

    # the design rule the sweep was built to test
    short, wide = noise_out["two"][0], noise_out["two"][-1]
    r_short = (short["k_matched_far_cut"][1]
               / short["k_matched_far_cut"][0])
    r_wide = wide["k_matched_far_cut"][1] / wide["k_matched_far_cut"][0]
    print(f"\n  the design rule under test: differencing should hurt most "
          f"when the spacing is short against the {CORR_LEN:.0f} mm "
          f"correlation length")
    print(f"  k_far at {short['spacing_mm']:.0f} mm: independent "
          f"{short['k_matched_far_cut'][0]:.2f}, correlated "
          f"{short['k_matched_far_cut'][1]:.2f}   "
          f"(correlated / independent = {r_short:.2f})")
    r_single_850 = noise_out["single"][f"{max(STATIONS):.0f}"][
        "sd_correlated_over_independent"]
    r_single_700 = noise_out["single"][f"{X_PAPER:.0f}"][
        "sd_correlated_over_independent"]
    print(f"  correlated noise costs a single cut a factor "
          f"{r_single_700:.2f} at {X_PAPER:.0f} mm and {r_single_850:.2f} "
          f"at {max(STATIONS):.0f} mm")
    print(f"  it costs the difference only "
          f"{short['sd_correlated_over_independent']:.2f} at "
          f"{short['spacing_mm']:.0f} mm, rising to "
          f"{wide['sd_correlated_over_independent']:.2f} at "
          f"{wide['spacing_mm']:.0f} mm: below the correlation length the "
          f"error is common to both cuts and differencing removes it")
    print(f"  k_far at {wide['spacing_mm']:.0f} mm: independent "
          f"{wide['k_matched_far_cut'][0]:.2f}, correlated "
          f"{wide['k_matched_far_cut'][1]:.2f}   "
          f"(correlated / independent = {r_wide:.2f})")
    out["design_rule_test"] = {
        "corr_len_mm": CORR_LEN,
        "short_spacing_mm": short["spacing_mm"],
        "wide_spacing_mm": wide["spacing_mm"],
        "k_far_short_independent": short["k_matched_far_cut"][0],
        "k_far_short_correlated": short["k_matched_far_cut"][1],
        "k_far_wide_independent": wide["k_matched_far_cut"][0],
        "k_far_wide_correlated": wide["k_matched_far_cut"][1],
        "k_far_corr_over_indep_short": r_short,
        "k_far_corr_over_indep_wide": r_wide,
        "sd_corr_over_indep_single_700": r_single_700,
        "sd_corr_over_indep_single_850": r_single_850,
        "sd_corr_over_indep_two_short": short[
            "sd_correlated_over_independent"],
        "sd_corr_over_indep_two_wide": wide[
            "sd_correlated_over_independent"],
        "verdict": (
            "the stated rule is not supported in the relative sense: "
            "correlation is the error mode differencing removes, so a "
            "spacing short against the correlation length is where the "
            "differenced form loses least against a single cut. In the "
            "absolute sense the rule holds, because the differenced "
            "standard deviation falls monotonically with spacing under "
            "both noise models."),
    }

    # ------------------------------------------------------------------
    # 5. the trade
    # ------------------------------------------------------------------
    widest = [c for c in cond if c["admissible"]][-1]
    arm_tol = 1.0 / abs(pp_per_mm)                # mm per pp of section loss
    print("\n== 5. the trade ==")
    print(f"  single cut needs the line of action to {arm_tol:.1f} mm per "
          f"percentage point of section loss, so about "
          f"{5.0 * arm_tol:.0f} mm for a 5 pp budget, and there is no "
          f"measurement of it inside a {prob.bearing:.0f} mm plate")
    kb = [best[m]["k_best_vs_best"] for m in MODELS]
    kp = [best[m]["k_vs_paper_station"] for m in MODELS]
    print(f"  two cut needs no line of action at all and is exactly "
          f"invariant to it, at k = {kb[0]:.1f} (independent) and "
          f"{kb[1]:.1f} (correlated) best layout against best layout")
    print(f"  against the {X_PAPER:.0f} mm station this study reports at, "
          f"the same arm-free estimate costs only k = {kp[0]:.1f} and "
          f"{kp[1]:.1f}: most of the penalty is that differencing forces a "
          f"second cut near the support, where the tie force is small and "
          f"the couple is noisy")
    print(f"  at the shortest spacing tried, {short['spacing_mm']:.0f} mm, "
          f"k_far rises to {short['k_matched_far_cut'][0]:.1f} and "
          f"{short['k_matched_far_cut'][1]:.1f}")
    out["trade"] = {
        "arm_tolerance_mm_per_pp": arm_tol,
        "arm_tolerance_mm_for_5pp": 5.0 * arm_tol,
        "bearing_mm": prob.bearing,
        "k_deterministic_widest": widest["k_vs_station_850"],
        "widest_spacing_mm": widest["spacing_mm"],
        "k_far_independent_widest": wide["k_matched_far_cut"][0],
        "k_far_correlated_widest": wide["k_matched_far_cut"][1],
        "shortest_spacing_mm": short["spacing_mm"],
        "k_independent_shortest": short["k_matched_far_cut"][0],
        "k_correlated_shortest": short["k_matched_far_cut"][1],
        "k_best_vs_best": {m: best[m]["k_best_vs_best"] for m in MODELS},
        "k_vs_paper_station": {m: best[m]["k_vs_paper_station"]
                               for m in MODELS},
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}   [{time.time() - t_start:.0f} s]")


if __name__ == "__main__":
    main()
