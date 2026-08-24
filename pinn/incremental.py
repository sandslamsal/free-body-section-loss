"""Candidate fix 3: identify from the CHANGE between two load levels.

The identification of this study reconciles a band couple against statics
at one measured state,

    C(theta; eps) = M_req(lambda),

and a cross-solver test showed what a change of constitutive model costs
it. Fields from a fixed smeared-crack quadrilateral, which carries a real
concrete tensile strength with exponential softening, are identified with
a slope of 0.770 against truth, so the response to theta survives almost
intact, but with an intercept of -0.278: the zero point moves by 27 points
of section loss and four of the five states leave the admissible range
entirely. The map neglects concrete tension, so it under-computes the band
tension the alternative field's own statics requires, and the
reconciliation can only close on a negative section loss.

The hypothesis tested here is that the offset is a property of the model
pair and not of theta, so that it is present at any load level and cancels
when the identification is posed on an increment. Writing the condition at
two levels and subtracting,

    C(theta; eps_2) - C(theta; eps_1) = M_req(lambda_2) - M_req(lambda_1),

any contribution the map misses that is similar at the two states drops
out, while the steel term moves because the band strain moved. It is also
the measurement an inspector can actually make: an existing structure has
no baseline for its absolute state, but the change under a proof load is
observable.

The cost is conditioning. The incremental steel term is a difference of
two similar numbers, and on this member the tie is past yield at every
station from 2.0 mm upward, where the bilinear law hardens at 1053 MPa and
the stress moves by 2 MPa in 500 between 3.5 and 5.0 mm. The pair of
levels is therefore not a free choice, and the module reports every pair
it is given rather than the best one.

Run:  python incremental.py       (writes figures/incremental.json)
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

import figdata as FD                                                       # noqa: E402
from csfm_constitutive import steel_stress                                 # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains, bracket_root                    # noqa: E402

ORACLE = HERE.parent / "oracle"
OUT = HERE.parent / "figures" / "incremental.json"
ARM = 370.0                     # reaction centroid measured on the reference
THETA = [0.0, 0.10, 0.20, 0.30, 0.40]
WIDE = np.linspace(-0.70, 0.70, 281)     # admits negative section loss
N_REAL, CORR_LEN = 50, 150.0
NOISE_MODELS = ("independent", "correlated", "dropout")


# ----------------------------------------------------------------------
# the two residuals
# ----------------------------------------------------------------------
def m_req(prob, lam, arm=ARM):
    """Moment the free body requires at the cut, in kN m."""
    return lam * prob.P / 2.0 * (FD.X_CUT - arm) / 1e6


def couple(prob, state, area, theta):
    """Band couple T z at a trial section loss, in kN m."""
    return FD.band_couple(prob, *state, area, theta)[2]


def tension(prob, state, area, theta):
    """Band tension T and arm z at a trial section loss (kN, mm)."""
    T, z, _ = FD.band_couple(prob, *state, area, theta)
    return T, z


def single(prob, state, area, lam, arm=ARM, grid=WIDE):
    """The identification as written: root of C(theta) - M_req."""
    f = np.array([couple(prob, state, area, g) - m_req(prob, lam, arm)
                  for g in grid])
    return bracket_root(f, grid), f


def incremental(prob, s1, s2, area, lam1, lam2, arm=ARM, grid=WIDE):
    """Root of [C(theta; eps_2) - C(theta; eps_1)] - [M_req_2 - M_req_1].

    The same residual as `single`, differenced across two load levels.

    What the difference actually does is worth writing down, because it is
    not an average of the two single-state answers. Linearise the residual
    at each level, R_k(theta) = s_k (theta - theta_k) with s_k = dC_k/dtheta
    and theta_k the single-state root. The incremental root solves
    R_2 = R_1, so

        theta* = theta_2 + s_1 (theta_2 - theta_1) / (s_2 - s_1),

    an EXTRAPOLATION beyond the higher level with gain s_1/(s_2 - s_1),
    which on this member is about 1.5. Writing the model-form error as the
    moment the map is short of at the true value, delta_k = M_req,k -
    C_k(theta_true), so that theta_k = theta_true + delta_k / s_k, the same
    algebra collapses to

        theta* - theta_true = (delta_2 - delta_1) / (s_2 - s_1).

    That is the whole hypothesis, stated exactly. A shortfall that is the
    same NUMBER OF KILONEWTON METERS at both levels cancels completely. A
    shortfall that is the same NUMBER OF POINTS OF SECTION LOSS at both
    levels does not cancel at all; it passes through unchanged. And
    whatever part of it does change is divided by the incremental
    sensitivity s_2 - s_1, which is smaller than either s_k, so the
    surviving part is amplified. The two identities are checked against the
    computed roots in every table this module prints.
    """
    dM = m_req(prob, lam2, arm) - m_req(prob, lam1, arm)
    f = np.array([couple(prob, s2, area, g) - couple(prob, s1, area, g) - dM
                  for g in grid])
    return bracket_root(f, grid), f


def incremental_T(prob, s1, s2, area, lam1, lam2, arm=ARM, grid=WIDE):
    """The same increment written on the tension rather than on the couple.

    T(theta; eps_2) - T(theta; eps_1) = M_req_2 / z_2(theta) - M_req_1 /
    z_1(theta), with each level's arm read from its own field. Kept because
    it is the form the hypothesis is naturally stated in; it is not
    algebraically the same as the couple increment, because the two levels
    have different arms.
    """
    f = []
    for g in grid:
        T1, z1 = tension(prob, s1, area, g)
        T2, z2 = tension(prob, s2, area, g)
        f.append((T2 - T1)
                 - (m_req(prob, lam2, arm) * 1e3 / z2
                    - m_req(prob, lam1, arm) * 1e3 / z1))
    f = np.array(f)
    return bracket_root(f, grid), f


def sign_changes(f):
    return int(np.sum(np.sign(f[:-1]) != np.sign(f[1:])))


def sensitivity(prob, state, area, theta=0.0, h=1e-4):
    """dT/dtheta and dC/dtheta at a trial value, by central difference."""
    Tp, _ = tension(prob, state, area, theta + h)
    Tm, _ = tension(prob, state, area, theta - h)
    Cp = couple(prob, state, area, theta + h)
    Cm = couple(prob, state, area, theta - h)
    return (Tp - Tm) / (2 * h), (Cp - Cm) / (2 * h)


def affine(true, rec):
    """Fitted slope and intercept of recovered against true."""
    t = np.array([a for a, b in zip(true, rec) if b is not None
                  and np.isfinite(b)])
    r = np.array([b for b in rec if b is not None and np.isfinite(b)])
    if t.size < 2:
        return None
    s, i = np.polyfit(t, r, 1)
    return {"slope": float(s), "intercept": float(i), "n": int(t.size)}


# ----------------------------------------------------------------------
# field families
# ----------------------------------------------------------------------
def band_strain(prob, state):
    """Mean band strain on the strip that stands for the cut, and the
    smeared-steel stress it implies."""
    cx, cy, ex = state[0], state[1], state[2]
    strip = (cy < FD.BAND) & (np.abs(cx - FD.X_CUT) < FD.BAND_W)
    e = float(ex[strip].mean())
    s = float(steel_stress(torch.tensor(ex[strip]), prob.mat).mean())
    return e, s


def same_solver(delta):
    """The reference family at one deflection station."""
    d = np.load(ORACLE / "fields_theta.npz")
    out = []
    for th in THETA:
        u = d[f"u_{th:.2f}_{delta}"]
        out.append((th, element_strains(d["xy"], u, FD.NX, FD.NY),
                    float(d[f"lam_{th:.2f}_{delta}"][0])))
    return out


CROSS_FILE = {"1.0": "fields_crossmodel_low.npz",
              "2.0": "fields_crossmodel_prepeak.npz",
              "3.5": "fields_crossmodel.npz"}


def fixed_crack(delta, tag="40x20"):
    """The alternative-model family at one deflection station."""
    f = ORACLE / CROSS_FILE[delta]
    if not f.exists():
        return None
    d = np.load(f)
    out = []
    for th in THETA:
        u = d[f"u_{tag}_{th:.2f}_{delta}"]
        out.append((th, element_strains(d["xy"], u, FD.NX, FD.NY),
                    float(d[f"lam_{tag}_{th:.2f}_{delta}"][0])))
    return out


# ----------------------------------------------------------------------
def run_pair(prob, area, fam, d1, d2, label):
    """Single-state and incremental identification for one pair of levels."""
    a, b = fam(d1), fam(d2)
    rows = []
    for (th, s1, l1), (_, s2, l2) in zip(a, b):
        r_lo, _f = single(prob, s1, area, l1)
        r_hi, _f = single(prob, s2, area, l2)
        r_in, f_in = incremental(prob, s1, s2, area, l1, l2)
        r_iT, f_iT = incremental_T(prob, s1, s2, area, l1, l2)
        e1, ss1 = band_strain(prob, s1)
        e2, ss2 = band_strain(prob, s2)
        dT1, dC1 = sensitivity(prob, s1, area, th)
        dT2, dC2 = sensitivity(prob, s2, area, th)
        # what the map is short of at the TRUE section loss, in kN m: the
        # whole of the model-form error, expressed where it acts
        sh1 = m_req(prob, l1) - couple(prob, s1, area, th)
        sh2 = m_req(prob, l2) - couple(prob, s2, area, th)
        bias_pred = ((sh2 - sh1) / (dC2 - dC1)) if dC2 != dC1 else None
        rows.append({
            "theta_true": th,
            "lam_lo": l1, "lam_hi": l2,
            "d_lam": l2 - l1,
            "M_req_lo_kNm": m_req(prob, l1), "M_req_hi_kNm": m_req(prob, l2),
            "dM_req_kNm": m_req(prob, l2) - m_req(prob, l1),
            "eps_band_lo": e1, "eps_band_hi": e2,
            "sig_steel_lo_MPa": ss1, "sig_steel_hi_MPa": ss2,
            "d_sig_steel_MPa": ss2 - ss1,
            "dT_dtheta_lo_kN": dT1, "dT_dtheta_hi_kN": dT2,
            "dT_dtheta_incr_kN": dT2 - dT1,
            "dC_dtheta_lo_kNm": dC1,
            "dC_dtheta_hi_kNm": dC2, "dC_dtheta_incr_kNm": dC2 - dC1,
            "cond_ratio": abs((dC2 - dC1) / dC2) if dC2 else None,
            "shortfall_lo_kNm": sh1, "shortfall_hi_kNm": sh2,
            "d_shortfall_kNm": sh2 - sh1,
            "bias_single_hi_pp": None,
            "bias_incr_pred_pp": (None if bias_pred is None
                                  else bias_pred * 100.0),
            "bias_incr_actual_pp": None,
            "rec_lo": None if not np.isfinite(r_lo) else float(r_lo),
            "rec_hi": None if not np.isfinite(r_hi) else float(r_hi),
            "rec_incr": None if not np.isfinite(r_in) else float(r_in),
            "rec_incr_T": None if not np.isfinite(r_iT) else float(r_iT),
            "roots_incr": sign_changes(f_in),
            "roots_incr_T": sign_changes(f_iT),
            "admissible_lo": bool(np.isfinite(r_lo) and r_lo >= 0.0),
            "admissible_hi": bool(np.isfinite(r_hi) and r_hi >= 0.0),
            "admissible_incr": bool(np.isfinite(r_in) and r_in >= 0.0)})
        rows[-1]["bias_single_hi_pp"] = (None if not np.isfinite(r_hi)
                                         else (r_hi - th) * 100.0)
        rows[-1]["bias_incr_actual_pp"] = (None if not np.isfinite(r_in)
                                           else (r_in - th) * 100.0)
    t = [r["theta_true"] for r in rows]
    out = {"label": label, "delta_lo": d1, "delta_hi": d2, "rows": rows,
           "fit_single_lo": affine(t, [r["rec_lo"] for r in rows]),
           "fit_single_hi": affine(t, [r["rec_hi"] for r in rows]),
           "fit_incr": affine(t, [r["rec_incr"] for r in rows]),
           "fit_incr_T": affine(t, [r["rec_incr_T"] for r in rows]),
           "n_admissible_hi": sum(r["admissible_hi"] for r in rows),
           "n_admissible_incr": sum(r["admissible_incr"] for r in rows),
           "monotonic_hi": monotone([r["rec_hi"] for r in rows]),
           "monotonic_incr": monotone([r["rec_incr"] for r in rows])}
    return out


def monotone(v):
    w = [x for x in v if x is not None and np.isfinite(x)]
    return bool(len(w) >= 2 and all(b > a for a, b in zip(w, w[1:])))


def table(p):
    print(f"\n{p['label']}   levels {p['delta_lo']} and {p['delta_hi']} mm")
    print(f"{'theta':>6}{'lam lo':>8}{'lam hi':>8}{'dM req':>9}"
          f"{'eps lo':>10}{'eps hi':>10}{'d sig':>8}"
          f"{'dC/dth hi':>11}{'dC/dth incr':>13}{'cond':>7}"
          f"{'rec hi':>9}{'rec incr':>10}{'incr T':>9}{'roots':>7}")
    for r in p["rows"]:
        f = lambda v, n=4: "  none" if v is None else f"{v:.{n}f}"   # noqa: E731
        print(f"{r['theta_true']:>6.2f}{r['lam_lo']:>8.4f}{r['lam_hi']:>8.4f}"
              f"{r['dM_req_kNm']:>9.2f}"
              f"{r['eps_band_lo']:>10.3e}{r['eps_band_hi']:>10.3e}"
              f"{r['d_sig_steel_MPa']:>8.1f}"
              f"{r['dC_dtheta_hi_kNm']:>11.2f}{r['dC_dtheta_incr_kNm']:>13.2f}"
              f"{r['cond_ratio']:>7.3f}"
              f"{f(r['rec_hi']):>9}{f(r['rec_incr']):>10}"
              f"{f(r['rec_incr_T']):>9}{r['roots_incr']:>7d}")
    for k, lab in (("fit_single_hi", "single state, high level"),
                   ("fit_single_lo", "single state, low level"),
                   ("fit_incr", "increment on the couple"),
                   ("fit_incr_T", "increment on the tension")):
        v = p[k]
        print(f"    {lab:<26}"
              + ("no fit" if v is None else
                 f"slope {v['slope']:+.4f}   intercept {v['intercept']:+.4f}"
                 f"   ({v['n']} states)"))
    print(f"    admissible: single {p['n_admissible_hi']}/5, "
          f"incremental {p['n_admissible_incr']}/5;   monotonic: single "
          f"{p['monotonic_hi']}, incremental {p['monotonic_incr']}")
    print(f"    {'why':<26}"
          f"{'shortfall lo':>14}{'shortfall hi':>14}{'change':>9}"
          f"{'/ dC incr':>11}{'= bias pred':>13}{'bias got':>10}"
          f"{'bias single':>13}")
    for r in p["rows"]:
        g = lambda v: "    --" if v is None else f"{v:+.1f}"        # noqa: E731
        print(f"    theta {r['theta_true']:<20.2f}"
              f"{r['shortfall_lo_kNm']:>14.2f}{r['shortfall_hi_kNm']:>14.2f}"
              f"{r['d_shortfall_kNm']:>9.2f}"
              f"{r['dC_dtheta_incr_kNm']:>11.2f}"
              f"{g(r['bias_incr_pred_pp']):>13}"
              f"{g(r['bias_incr_actual_pp']):>10}"
              f"{g(r['bias_single_hi_pp']):>13}")


# ----------------------------------------------------------------------
# what the increment costs under noise
# ----------------------------------------------------------------------
NOISE_GRID = np.linspace(-1.5, 1.5, 301)


def correlated_draw(rng, cx, cy, sd):
    """One draw from an exponential-covariance field (noise_study.py)."""
    from scipy.spatial import cKDTree
    n = cx.size
    idx = rng.choice(n, size=min(n, 400), replace=False)
    d2 = ((cx[idx, None] - cx[None, idx]) ** 2
          + (cy[idx, None] - cy[None, idx]) ** 2)
    K = np.exp(-np.sqrt(d2) / CORR_LEN) + 1e-8 * np.eye(idx.size)
    z = np.linalg.cholesky(K) @ rng.standard_normal(idx.size)
    _, nn = cKDTree(np.c_[cx[idx], cy[idx]]).query(np.c_[cx, cy])
    return sd * z[nn]


def perturb(rng, model, state, sd):
    """One noisy reading of one level, on the 5 % protocol of noise_study."""
    cx, cy, ex, ey, gxy = state
    if model == "correlated":
        pert = [a + correlated_draw(rng, cx, cy, sd) for a in (ex, ey, gxy)]
    else:
        pert = [a + rng.normal(0.0, sd, a.shape) for a in (ex, ey, gxy)]
    return pert


def noise_pair(prob, area, fam, d1, d2, label, models=NOISE_MODELS):
    """Spread of the single-state and incremental recovery under 5 % noise.

    Each level is a separate reading, so each gets its own draw. Gauge
    dropout is a property of the installation rather than of the reading,
    so the same gauges are missing at both levels; the additive error is
    independent between them, which is the unfavourable case for an
    increment and the honest one for two visits to a structure.
    """
    a, b = fam(d1), fam(d2)
    res = {}
    for model in models:
        rng = np.random.default_rng(0)
        rows = []
        for (th, s1, l1), (_, s2, l2) in zip(a, b):
            cx, cy = s1[0], s1[1]
            sd1 = 0.05 * float(np.abs(s1[2][cy < FD.BAND]).mean())
            sd2 = 0.05 * float(np.abs(s2[2][cy < FD.BAND]).mean())
            hi, inc = [], []
            for _j in range(N_REAL):
                p1 = perturb(rng, model, s1, sd1)
                p2 = perturb(rng, model, s2, sd2)
                keep = np.ones(cx.size, bool)
                if model == "dropout":
                    keep[rng.choice(cx.size, int(0.15 * cx.size),
                                    replace=False)] = False
                q1 = (cx[keep], cy[keep], p1[0][keep], p1[1][keep],
                      p1[2][keep])
                q2 = (cx[keep], cy[keep], p2[0][keep], p2[1][keep],
                      p2[2][keep])
                r, _f = single(prob, q2, area, l2, grid=NOISE_GRID)
                hi.append(r)
                r, _f = incremental(prob, q1, q2, area, l1, l2,
                                    grid=NOISE_GRID)
                inc.append(r)
            hi = np.array(hi); inc = np.array(inc)
            fh, fi = hi[np.isfinite(hi)], inc[np.isfinite(inc)]
            rows.append({
                "theta_true": th,
                "single_mean": float(fh.mean()) if fh.size else None,
                "single_sd": float(fh.std()) if fh.size else None,
                "single_n": int(fh.size),
                "incr_mean": float(fi.mean()) if fi.size else None,
                "incr_sd": float(fi.std()) if fi.size else None,
                "incr_n": int(fi.size),
                "sd_ratio": (float(fi.std() / fh.std())
                             if fh.size and fi.size and fh.std() > 0
                             else None)})
        res[model] = rows
    return {"label": label, "delta_lo": d1, "delta_hi": d2,
            "n_real": N_REAL, "amplitude": 0.05, "models": res}


def noise_table(nz):
    print(f"\n{nz['label']}   levels {nz['delta_lo']} and {nz['delta_hi']} mm"
          f"   ({nz['n_real']} realizations, 5 % noise)")
    print(f"{'model':>13}{'theta':>7}{'single':>22}{'incremental':>24}"
          f"{'sd ratio':>10}")
    for model, rows in nz["models"].items():
        for r in rows:
            f = lambda m, s: ("none" if m is None                   # noqa: E731
                              else f"{m:+.3f} +- {s:.3f}")
            print(f"{model:>13}{r['theta_true']:>7.2f}"
                  f"{f(r['single_mean'], r['single_sd']):>22}"
                  f"{f(r['incr_mean'], r['incr_sd']):>24}"
                  + ("      --" if r["sd_ratio"] is None
                     else f"{r['sd_ratio']:>10.1f}"))
        v = [r["sd_ratio"] for r in rows if r["sd_ratio"] is not None]
        if v:
            print(f"{'':>13}{'mean':>7}{'':>46}{np.mean(v):>10.1f}")


# ----------------------------------------------------------------------
def main() -> None:
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0

    pairs = []
    print("=" * 118)
    print("STEP 2  the same-solver family: the increment must work here "
          "before it means anything anywhere else")
    print("=" * 118)
    for d1, d2 in (("1.0", "3.5"), ("3.5", "5.0"), ("1.0", "2.0"),
                   ("2.0", "3.5")):
        p = run_pair(prob, area, same_solver, d1, d2,
                     "same solver: CST, rotating cracked membrane")
        pairs.append(p)
        table(p)

    print("\n" + "=" * 118)
    print("STEP 3  the fixed smeared-crack family, identified unchanged. "
          "Single-state baseline at 3.5 mm: slope 0.770, intercept -0.278")
    print("=" * 118)
    cross = []
    for d1, d2 in (("1.0", "3.5"), ("2.0", "3.5"), ("1.0", "2.0")):
        if fixed_crack(d1) is None or fixed_crack(d2) is None:
            print(f"  [{d1}, {d2}] not generated")
            continue
        p = run_pair(prob, area, fixed_crack, d1, d2,
                     "new model: fixed crack, Q4 40x20")
        cross.append(p)
        table(p)

    print("\n" + "=" * 118)
    print("STEP 4  conditioning cost, on the 5 % noise protocol of "
          "noise_study.py")
    print("=" * 118)
    noise = [noise_pair(prob, area, same_solver, "1.0", "3.5",
                        "same solver: CST, rotating cracked membrane")]
    for nz in noise:
        noise_table(nz)
    nzx = None
    if fixed_crack("1.0") is not None:
        nzx = noise_pair(prob, area, fixed_crack, "1.0", "3.5",
                         "new model: fixed crack, Q4 40x20",
                         models=("independent",))
        noise_table(nzx)

    out = {"what": "candidate fix 3: identify from the change between two "
                   "load levels rather than from the absolute state",
           "residual_single": "C(theta; eps) - M_req(lambda)",
           "residual_incremental":
               "[C(theta; eps_2) - C(theta; eps_1)] - "
               "[M_req(lambda_2) - M_req(lambda_1)]",
           "bias_law": "theta* - theta_true = (delta_2 - delta_1) / "
                       "(s_2 - s_1), with delta_k = M_req,k - "
                       "C_k(theta_true) the moment shortfall of the map at "
                       "the true value and s_k = dC_k/dtheta. A shortfall "
                       "equal in kN m at both levels cancels; a shortfall "
                       "equal in points of section loss does not cancel at "
                       "all; what changes is divided by the incremental "
                       "sensitivity and so is amplified.",
           "arm_mm": ARM, "x_cut_mm": FD.X_CUT, "band_mm": FD.BAND,
           "trial_grid": [float(WIDE[0]), float(WIDE[-1]), int(WIDE.size)],
           "baseline_single_state_3.5mm": {
               "same_solver": {"slope": 0.8317, "intercept": -0.0092},
               "fixed_crack_40x20": {"slope": 0.7697, "intercept": -0.2779}},
           "same_solver_pairs": pairs, "fixed_crack_pairs": cross,
           "noise": noise + ([nzx] if nzx else [])}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
