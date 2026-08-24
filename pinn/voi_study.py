"""Value of information of the identification, priced by pre-posterior decision analysis.

WHY: a corrosion reading is only worth taking if it can change a decision,
so this study prices the identification in the currency the decision is
made in: expected annual cost. The chain of reliability.py already turns a
reading into a failure probability; here that chain is closed with the two
actions an owner actually has, and the reading's value is the expected cost
it removes. The paper's own twist follows: the reading inherits a hidden
systematic bias from the assumed internal lever arm, so the value of the
reading is recomputed as a function of the arm tolerance, and the tolerance
at which the reading stops paying for itself is the deployment requirement.

Decision setting, with every assumption stated:

1. One inspect-then-act cycle on an annual horizon. Doing nothing costs
   C_f * P_f for the coming year, with P_f the failure probability the
   available information implies. Repair costs C_r, restores theta = 0,
   and leaves the annual cost C_r + C_f * P_f(0). No discounting and no
   repeated inspection; the two actions are compared within the same year.
2. C_f = 1, so every cost is a fraction of the failure consequence.
   C_r in {1e-3, 1e-2, 1e-1} times C_f; the middle value is the headline.
3. Prior on theta: exactly the push-forward prior of reliability.py
   (lognormal corrosion current, median 1.0 uA/cm2, COV 0.5, 50 years,
   four-bar band, clip at 0.7), same seed, rebinned to 400 grid points.
4. Likelihood: the calibrated identification response of this study,
   theta_hat | theta ~ N(0.830 theta - 0.005, 0.025).
5. Capacity and limit state: lambda_R(theta) = 1.286 - 0.711 theta (the
   linear fit reliability.py stores), g = X_R * lambda_R - S, X_R
   lognormal with median 1.0 and COV 0.10, S the annual-maximum Gumbel
   with mean 0.50 and COV 0.15.
6. P_f(theta) is computed ONCE, by the FORM of reliability.py at every
   node of the theta grid, and interpolated (in beta) inside the decision
   loop; the interpolant is verified against the betas of
   figures/reliability.json to 0.02 and the check is reported.
7. Grids: 400 points on theta in [0, 0.7]; 501 points on the reading in
   [-0.46, 1.04], wide enough that the six-sigma support of every biased
   predictive stays on the grid. The setting is one dimensional, so the
   pre-posterior expectation is a quadrature, not a simulation.
8. Prior decision: the cheaper action under the prior. With information:
   observe theta_hat, form the posterior, take the cheaper action given
   it, i.e. repair exactly when the posterior expected P_f exceeds
   C_r / C_f + P_f(0). VoI is the standard pre-posterior expectation of
   the cost the reading removes, and is never negative.
9. Arm twist: an arm error of da mm adds a hidden systematic bias of
   -0.0025 * da (theta units) to the reading; the decision rule still
   assumes the calibrated likelihood. The sign follows the stated law:
   da > 0 is an arm assumed short. The arm sweep of figdata.npz measures
   +0.0027 per mm of assumed arm at theta = 0.2, matching the magnitude.
   Because a deployment tolerance is two sided, the mirrored branch (arm
   assumed long, bias +0.0025 * da) is computed as well: an understating
   bias only suppresses repair, which is already the prior action, so it
   can only decay the VoI to zero, while an overstating bias triggers
   repairs the posterior cannot justify and drives the VoI negative.
10. The with-information cost under bias is the expectation over the TRUE
    joint, p(theta) N(theta_hat; 0.830 theta - 0.005 + bias, 0.025), of
    the cost of the action the bias-unaware rule takes.

Run:  python voi_study.py     (writes ../figures/voi.json)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import reliability as REL                                                # noqa: E402

FIG = HERE.parent / "figures"
OUT = FIG / "voi.json"
REL_JSON = FIG / "reliability.json"
FIGDATA = FIG / "figdata.npz"

N_THETA = 400
N_HAT = 501
HAT_LO, HAT_HI = -0.46, 1.04
C_F = 1.0
C_R = (1e-3, 1e-2, 1e-1)          # times C_f; the middle value is the headline
BIAS_PER_MM = -0.0025             # theta units per mm of arm error (arm short)
DA_MM = np.arange(0.0, 121.0, 10.0)
TOL_BETA = 0.02


# ----------------------------------------------------------------------
# P_f(theta) on the grid, once, and its verification
# ----------------------------------------------------------------------
def pf_of_theta_grid(lam_R, grid):
    """FORM beta at every grid node; beta, not pf, is what interpolates well."""
    beta = np.array([REL.form_beta(lam_R, float(t), 0.0) for t in grid])
    pf = np.array([REL.phi_cdf(-b) for b in beta])
    return beta, pf


def verify_against_reliability_json(grid, beta_g, pf_g, prior, dth):
    """The five MC betas of reliability.json, reproduced from this grid.

    Point cases check the interpolant directly; the prior and the two
    posteriors check the same mixture integration the decision loop uses.
    """
    js = json.loads(REL_JSON.read_text())
    hat_obs = js["identification"]["theta_hat_obs"]

    def mixture_beta(weights):
        w = weights / (weights.sum() * dth)
        pf = float(np.sum(w * pf_g) * dth)
        return -REL.phi_inv(pf), pf

    rows = []
    for name, th in (("point_reading_0.16", 0.16), ("point_true_0.20", 0.20)):
        b = float(np.interp(th, grid, beta_g))
        rows.append((name, b, js["cases"][name]["beta_mc"],
                     js["cases"][name]["beta_form"]))
    b_prior, _ = mixture_beta(prior.copy())
    rows.append(("prior", b_prior, js["cases"]["prior"]["beta_mc"], None))
    like_cal = REL.norm_pdf(hat_obs, REL.IDENT_SLOPE * grid
                            + REL.IDENT_INTERCEPT, REL.IDENT_SD)
    b_cal, _ = mixture_beta(prior * like_cal)
    rows.append(("posterior_calibrated", b_cal,
                 js["cases"]["posterior_calibrated"]["beta_mc"], None))
    like_nai = REL.norm_pdf(hat_obs, grid, REL.IDENT_SD)
    b_nai, _ = mixture_beta(prior * like_nai)
    rows.append(("posterior_naive", b_nai,
                 js["cases"]["posterior_naive"]["beta_mc"], None))

    checks, ok = [], True
    print("check of the P_f(theta) grid against reliability.json betas "
          f"(tolerance {TOL_BETA}):")
    for name, b, b_mc, b_form in rows:
        d_mc = b - b_mc
        passed = abs(d_mc) <= TOL_BETA
        ok &= passed
        extra = "" if b_form is None else f"  dbeta_form={b - b_form:+.4f}"
        print(f"  {name:22s} beta_here={b:.4f}  beta_mc={b_mc:.4f}"
              f"  dbeta_mc={d_mc:+.4f}{extra}  "
              f"{'pass' if passed else 'FAIL'}")
        checks.append({"case": name, "beta_here": b, "beta_mc_json": b_mc,
                       "dbeta_vs_mc": d_mc,
                       "beta_form_json": b_form, "pass": passed})
    if not ok:
        raise RuntimeError("P_f(theta) grid failed the 0.02 beta check")
    return checks


# ----------------------------------------------------------------------
# the pre-posterior machinery, all on grids
# ----------------------------------------------------------------------
def first_crossing(x, v, level=0.0):
    """First x at which v falls to `level` (linear interpolation)."""
    v = np.asarray(v, float)
    for k in range(1, v.size):
        if v[k - 1] > level >= v[k]:
            t = (v[k - 1] - level) / (v[k - 1] - v[k])
            return float(x[k - 1] + t * (x[k] - x[k - 1]))
    return None


def measured_arm_slope():
    """Slope of recovered theta per mm of assumed arm, from this study's
    own arm sweep, as the cross-check of the stated bias law."""
    if not FIGDATA.exists():
        return None
    z = np.load(FIGDATA)
    a, r = z["arm_a"], z["arm_rec"]
    i = int(np.argmin(np.abs(z["obs_theta"] - 0.20)))
    ok = np.isfinite(r[i])
    return float(np.polyfit(a[ok], r[i][ok], 1)[0])


def main() -> None:
    lam_R, cap_info = REL.capacity_curve()

    # the prior of reliability.py, same seed, rebinned to this study's grid
    REL.N_GRID = N_THETA
    rng = np.random.default_rng(REL.SEED)
    grid, prior, prior_stats = REL.prior_on_grid(rng)
    dth = grid[1] - grid[0]
    prior = prior / (prior.sum() * dth)

    print("P_f(theta) by FORM on the grid, once ...", flush=True)
    beta_g, pf_g = pf_of_theta_grid(lam_R, grid)
    beta0 = REL.form_beta(lam_R, 0.0, 0.0)
    pf0 = REL.phi_cdf(-beta0)
    checks = verify_against_reliability_json(grid, beta_g, pf_g, prior, dth)

    pf_prior = float(np.sum(prior * pf_g) * dth)
    beta_prior = -REL.phi_inv(pf_prior)

    # reading grid, unbiased likelihood, and the assessor's posterior
    hat = np.linspace(HAT_LO, HAT_HI, N_HAT)
    dh = hat[1] - hat[0]
    mu = REL.IDENT_SLOPE * grid + REL.IDENT_INTERCEPT

    def like_matrix(bias):
        return REL.norm_pdf(hat[None, :], (mu + bias)[:, None], REL.IDENT_SD)

    L0 = like_matrix(0.0)
    m0 = (prior[:, None] * L0).sum(axis=0) * dth       # prior predictive
    # posterior expected P_f via log weights, so far-out readings resolve
    # to the nearest end of the prior support instead of 0/0
    logw = (np.log(np.maximum(prior, 1e-300))[:, None]
            - 0.5 * ((hat[None, :] - mu[:, None]) / REL.IDENT_SD) ** 2)
    W = np.exp(logw - logw.max(axis=0, keepdims=True))
    Epf = (W * pf_g[:, None]).sum(axis=0) / W.sum(axis=0)

    cdf = np.cumsum(m0) * dh
    q99 = float(np.interp(0.99 * cdf[-1], cdf, hat))

    row_mass = L0.sum(axis=1) * dh
    print(f"prior risk C_f*E[P_f] = {pf_prior:.4e} (beta {beta_prior:.3f});"
          f" P_f(0) = {pf0:.4e}; max E[P_f|reading] = {Epf.max():.4e}")
    print(f"likelihood rows integrate to 1 within "
          f"{np.abs(row_mass - 1).max():.1e}")

    def tidy(v):
        """Quadrature residue of order 1e-20 is not a negative VoI."""
        return 0.0 if abs(v) < 1e-15 else float(v)

    def with_info_cost(Lb, repair_mask, thr):
        """True expected cost of the rule under a (possibly biased) joint."""
        joint = prior[:, None] * Lb * dth
        mb = joint.sum(axis=0)                       # predictive of reading
        cn = (joint * pf_g[:, None]).sum(axis=0)     # do-nothing cost density
        per_hat = np.where(repair_mask, thr * mb, C_F * cn)
        return float(per_hat.sum() * dh)

    # decision rule and baseline per repair cost
    decision, rules = {}, {}
    for cr in C_R:
        thr = cr + C_F * pf0                          # cost of repairing now
        repair = C_F * Epf > thr
        cost_nothing = C_F * pf_prior
        prior_action = "repair" if thr < cost_nothing else "do nothing"
        prior_cost = min(cost_nothing, thr)
        hat_star = first_crossing(hat, thr - C_F * Epf) if repair.any() else None
        cost_perfect = float(np.sum(prior * np.minimum(C_F * pf_g, thr)) * dth)
        evpi = prior_cost - cost_perfect
        voi0 = tidy(prior_cost - with_info_cost(L0, repair, thr))
        rules[cr] = (thr, repair, prior_cost)
        decision[f"{cr:.0e}"] = {
            "C_r": cr, "cost_repair": thr, "cost_nothing_prior": cost_nothing,
            "prior_action": prior_action, "prior_cost": prior_cost,
            "hat_star": hat_star, "p_repair_unbiased":
                float((m0 * repair).sum() * dh),
            "evpi": evpi, "voi_unbiased": voi0,
            "voi_unbiased_pct_of_prior_risk": 100.0 * voi0 / pf_prior,
        }
        star = "none (repair never justified by any reading)" \
            if hat_star is None else f"{hat_star:.3f}"
        print(f"C_r={cr:.0e}: prior action {prior_action}"
              f" (cost {prior_cost:.3e}), repair threshold reading {star},"
              f" EVPI={evpi:.3e}, VoI={voi0:.3e}"
              f" ({100.0 * voi0 / pf_prior:.1f}% of prior risk)")

    # the arm twist: hidden bias, both branches of the tolerance
    print("VoI against arm tolerance ...", flush=True)
    voi = {f"{cr:.0e}": {"arm_short": [], "arm_long": []} for cr in C_R}
    for da in DA_MM:
        for branch, sgn in (("arm_short", 1.0), ("arm_long", -1.0)):
            Lb = like_matrix(sgn * BIAS_PER_MM * da)
            for cr in C_R:
                thr, repair, prior_cost = rules[cr]
                voi[f"{cr:.0e}"][branch].append(
                    tidy(prior_cost - with_info_cost(Lb, repair, thr)))

    crossings = {}
    for cr in C_R:
        key = f"{cr:.0e}"
        vs = np.array(voi[key]["arm_short"])
        vl = np.array(voi[key]["arm_long"])
        v0 = vs[0]
        crossings[key] = {
            "arm_short_bias_minus": first_crossing(DA_MM, vs),
            "arm_long_bias_plus": first_crossing(DA_MM, vl),
            "arm_short_decay_to_1pct_mm":
                first_crossing(DA_MM, vs, 0.01 * v0) if v0 > 0 else None,
            "voi_da0": float(v0),
        }
        fmt = lambda c: "none" if c is None else f"{c:.1f} mm"
        print(f"C_r={cr:.0e}: VoI(0)={v0:.3e};"
              f" zero crossing arm-short {fmt(crossings[key]['arm_short_bias_minus'])},"
              f" arm-long {fmt(crossings[key]['arm_long_bias_plus'])};"
              f" short branch below 1% of VoI(0) at"
              f" {fmt(crossings[key]['arm_short_decay_to_1pct_mm'])}")

    slope = measured_arm_slope()
    out = {
        "docstring_assumptions":
            [a.strip() for a in next(
                p for p in __doc__.split("\n\n")
                if p.lstrip().startswith("1.")).splitlines()],
        "constants": {
            "C_f": C_F, "C_r": list(C_R), "headline_C_r": C_R[1],
            "ident_slope": REL.IDENT_SLOPE,
            "ident_intercept": REL.IDENT_INTERCEPT, "ident_sd": REL.IDENT_SD,
            "bias_theta_per_mm": BIAS_PER_MM,
            "measured_arm_slope_per_mm_figdata": slope,
            "capacity_coef_highest_first": cap_info["coef_highest_first"],
            "seed": REL.SEED, "n_theta": N_THETA, "n_hat": N_HAT,
        },
        "checks": checks, "checks_tolerance_beta": TOL_BETA,
        "pf_grid": {"theta": grid.tolist(), "beta_form": beta_g.tolist(),
                    "pf": pf_g.tolist(), "beta_theta0": beta0,
                    "pf_theta0": pf0},
        "prior": {"density": prior.tolist(), "stats": prior_stats,
                  "pf_prior": pf_prior, "beta_prior": beta_prior},
        "reading": {"grid": hat.tolist(),
                    "predictive_density": m0.tolist(),
                    "Epf_posterior": Epf.tolist(), "q99": q99},
        "decision": decision,
        "voi_vs_arm_mm": {"da_mm": DA_MM.tolist(), **voi},
        "crossings_mm": crossings,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
