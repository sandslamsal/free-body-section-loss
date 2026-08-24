"""Price the reaction-arm assumption in reliability currency.

Section 7 of this study measures a sensitivity of 0.25 percentage points of
recovered section loss per millimeter of error in the assumed position of
the bearing reaction, and shows that the plate center misses the measured
contact centroid by 100 mm. That error survives calibration: the response
law theta_hat = 0.830 theta - 0.005 was estimated with the arm at its true
value, so an arm error passes straight through as a systematic offset in
the reading, understating the loss when the assumed arm is too long. A
biased reading matters only if it changes a computed failure probability,
so this script carries the biased reading through exactly the prior,
likelihood calibration, capacity fit, limit state and Monte Carlo protocol
of reliability.py, at arm errors of 0, 10, 20, 50 and 100 mm of both signs,
and prices each millimeter of bearing uncertainty as a factor on annual
failure probability. Everything reusable is imported from reliability.py
rather than copied, and its random-number draw order is replicated exactly,
so the zero-arm-error case must reproduce the published beta table before
the sweep is trusted; that check runs first and the script aborts if it
fails.

Run:  python arm_reliability.py   (writes ../figures/arm_reliability.json)
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

import reliability as R                                        # noqa: E402

REF_JSON = HERE.parent / "figures" / "reliability.json"
OUT = HERE.parent / "figures" / "arm_reliability.json"

# Section 7: 0.25 percentage points of section loss per millimeter of arm
# error, held between 0.20 and 0.30 across the four deteriorated states.
# Positive arm error means the assumed arm is too long (the plate-center
# direction, +100 mm), which biases the reading DOWN: the longer arm
# demands more moment than the band can supply, so the recovery understates
# the loss.
ARM_SENS = 0.25e-2               # theta units per millimeter
ARM_ERRORS_MM = [0.0, 10.0, 20.0, 50.0, 100.0,
                 -10.0, -20.0, -50.0, -100.0]
CHECK_TOL = 0.01                 # required agreement with reliability.json


def reading_at(arm_err_mm: float) -> tuple[float, float, bool]:
    """The identification's reading at true theta under an arm error."""
    raw = (R.IDENT_SLOPE * R.THETA_TRUE + R.IDENT_INTERCEPT
           - ARM_SENS * arm_err_mm)
    clipped = float(np.clip(raw, 0.0, R.THETA_MAX))
    return clipped, raw, clipped != raw


def assess(reading, grid, prior, lam_R, xr, s, u_theta):
    """Both assessors' posteriors and reliabilities for one reading.

    naive: the reading is taken at face value, likelihood centered on theta.
    calibrated: slope and intercept are removed, but the arm bias is the
    part calibration cannot see, so it stays in the reading.
    """
    likes = {
        "naive": R.norm_pdf(reading, grid, R.IDENT_SD),
        "calibrated": R.norm_pdf(reading,
                                 R.IDENT_SLOPE * grid + R.IDENT_INTERCEPT,
                                 R.IDENT_SD),
    }
    out = {}
    for mode, like in likes.items():
        post, m, sd = R.grid_bayes(grid, prior, like)
        th = R.sample_from_grid(grid, post, u_theta)
        pf, se_pf, beta, se_beta = R.mc_case(lam_R, th, xr, s)
        out[mode] = {
            "posterior_mean": m, "posterior_sd": sd,
            "pf_mc": pf, "se_pf_mc": se_pf,
            "beta_mc": beta, "se_beta_mc": se_beta,
            "beta_form": R.form_beta(lam_R, m, sd),
        }
    return out


def main():
    lam_R, cap_info = R.capacity_curve()

    # identical draw order to reliability.main(): prior samples, then XR,
    # then S, then the common uniforms for theta, all from the same seed
    rng = np.random.default_rng(R.SEED)
    grid, prior, prior_stats = R.prior_on_grid(rng)
    xr = np.exp(math.sqrt(math.log(1.0 + R.XR_COV ** 2))
                * rng.standard_normal(R.N_MC))
    s_sd = R.S_COV * R.S_MEAN
    s_scale = s_sd * math.sqrt(6.0) / math.pi
    s_loc = R.S_MEAN - 0.5772156649015329 * s_scale
    s = s_loc - s_scale * np.log(-np.log(rng.random(R.N_MC)))
    u_theta = rng.random(R.N_MC)

    dth = grid[1] - grid[0]
    prior_norm = prior / (np.sum(prior) * dth)

    # ------------------------------------------------------------------
    # zero-arm-error check: with the published reading 0.160 this run must
    # reproduce every beta in figures/reliability.json, or nothing that
    # follows deserves trust
    # ------------------------------------------------------------------
    ref = json.loads(REF_JSON.read_text())
    zero = assess(R.THETA_HAT_OBS, grid, prior, lam_R, xr, s, u_theta)
    th_prior = R.sample_from_grid(grid, prior_norm, u_theta)
    pf_p, _, beta_p, _ = R.mc_case(lam_R, th_prior, xr, s)
    pf_t, _, beta_t, _ = R.mc_case(lam_R, np.full(R.N_MC, R.THETA_TRUE),
                                   xr, s)
    pf_r, _, beta_r, _ = R.mc_case(lam_R, np.full(R.N_MC, R.THETA_HAT_OBS),
                                   xr, s)
    reproduced = {
        "prior": beta_p,
        "posterior_calibrated": zero["calibrated"]["beta_mc"],
        "posterior_naive": zero["naive"]["beta_mc"],
        "point_true_0.20": beta_t,
        "point_reading_0.16": beta_r,
    }
    check = {}
    worst = 0.0
    for name, beta in reproduced.items():
        stored = ref["cases"][name]["beta_mc"]
        diff = abs(beta - stored)
        worst = max(worst, diff)
        check[name] = {"beta_stored": stored, "beta_reproduced": beta,
                       "abs_diff": diff}
    print("zero-arm-error check against figures/reliability.json:")
    for name, c in check.items():
        print(f"  {name:24s} stored={c['beta_stored']:.4f}"
              f"  reproduced={c['beta_reproduced']:.4f}"
              f"  |diff|={c['abs_diff']:.2e}")
    if worst > CHECK_TOL:
        raise SystemExit(f"zero-arm-error check FAILED: max |dbeta| {worst:.4f}"
                         f" exceeds {CHECK_TOL}")
    print(f"  PASS (max |dbeta| = {worst:.2e} <= {CHECK_TOL})\n")

    # ------------------------------------------------------------------
    # the sweep: reading = 0.830*0.20 - 0.005 - 0.0025*arm_err, clipped
    # ------------------------------------------------------------------
    sweep = []
    for e in ARM_ERRORS_MM:
        reading, raw, clipped = reading_at(e)
        row = {
            "arm_error_mm": e,
            "bias_pp": -ARM_SENS * e * 100.0,
            "reading_raw": raw,
            "reading": reading,
            "reading_clipped": clipped,
        }
        row.update(assess(reading, grid, prior, lam_R, xr, s, u_theta))
        sweep.append(row)

    # reference for the understatement factor: this run's calibrated
    # posterior at zero arm error (reproduces the published 3.99e-5 within
    # Monte Carlo error; the published pair is carried alongside)
    ref_row = next(r for r in sweep if r["arm_error_mm"] == 0.0)
    pf_ref = ref_row["calibrated"]["pf_mc"]
    beta_ref = ref_row["calibrated"]["beta_mc"]
    for row in sweep:
        for mode in ("naive", "calibrated"):
            pf = row[mode]["pf_mc"]
            row[mode]["pf_understatement_factor"] = (
                pf_ref / pf if pf > 0.0 else float("inf"))
            row[mode]["dbeta_vs_ref"] = row[mode]["beta_mc"] - beta_ref

    # ------------------------------------------------------------------
    # what knowing the bearing to 20 mm rather than 100 mm buys
    # ------------------------------------------------------------------
    def cal(e):
        return next(r for r in sweep if r["arm_error_mm"] == e)["calibrated"]

    f100, f20, f50, f10 = (cal(100.0)["pf_understatement_factor"],
                           cal(20.0)["pf_understatement_factor"],
                           cal(50.0)["pf_understatement_factor"],
                           cal(10.0)["pf_understatement_factor"])
    buy = f100 / f20
    sentence = (
        f"Carried through the reliability chain, the 100 mm by which the "
        f"plate center misses the contact centroid understates the annual "
        f"failure probability by a factor of {f100:.1f} "
        f"(beta {cal(100.0)['beta_mc']:.2f} in place of {beta_ref:.2f}), "
        f"whereas an arm known to 20 mm understates it by no more than a "
        f"factor of {f20:.1f} (beta {cal(20.0)['beta_mc']:.2f}); "
        f"instrumenting the bearing to 20 mm therefore buys a factor of "
        f"{buy:.1f} on the annual failure probability the assessment "
        f"reports.")

    out = {
        "convention": {
            "arm_error_mm": "assumed arm minus true arm; +100 mm is the "
                            "plate center in place of the contact centroid",
            "bias": "theta_hat offset = -0.0025 per mm of arm error "
                    "(0.25 pp/mm, Section 7); a too-long arm understates "
                    "the loss",
            "reading": "0.830*0.20 - 0.005 - 0.0025*arm_error, clipped to "
                       "[0, 0.7]",
            "theta_true": R.THETA_TRUE,
            "arm_sensitivity_pp_per_mm": 100.0 * ARM_SENS,
            "modes": {
                "naive": "likelihood centered on theta as if the reading "
                         "were unbiased",
                "calibrated": "slope and intercept removed; the arm bias "
                              "is the part calibration cannot remove",
            },
        },
        "mc": {"n_mc": R.N_MC, "seed": R.SEED,
               "common_random_numbers": True},
        "zero_arm_error_check": {
            "tolerance": CHECK_TOL, "max_abs_dbeta": worst,
            "passed": worst <= CHECK_TOL, "cases": check,
        },
        "reference": {
            "pf": pf_ref, "beta": beta_ref,
            "published_pf": ref["cases"]["posterior_calibrated"]["pf_mc"],
            "published_beta": ref["cases"]["posterior_calibrated"]["beta_mc"],
            "note": "calibrated posterior at zero arm error, reading "
                    "0.830*0.20 - 0.005 = 0.161; the published table used "
                    "the rounded reading 0.160",
        },
        "sweep": sweep,
        "headline": {
            "understatement_factor_at_10mm": f10,
            "understatement_factor_at_20mm": f20,
            "understatement_factor_at_50mm": f50,
            "understatement_factor_at_100mm": f100,
            "factor_bought_by_20mm_vs_100mm": buy,
            "sentence": sentence,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))

    hdr = (f"{'arm(mm)':>8s} {'bias(pp)':>9s} {'reading':>8s} "
           f"{'mode':>11s} {'post_m':>7s} {'pf':>10s} {'beta':>7s} "
           f"{'se_b':>6s} {'b_form':>7s} {'factor':>7s}")
    print(hdr)
    for row in sweep:
        for mode in ("naive", "calibrated"):
            r = row[mode]
            print(f"{row['arm_error_mm']:8.0f} {row['bias_pp']:9.1f} "
                  f"{row['reading']:8.3f} {mode:>11s} "
                  f"{r['posterior_mean']:7.4f} {r['pf_mc']:10.3e} "
                  f"{r['beta_mc']:7.4f} {r['se_beta_mc']:6.4f} "
                  f"{r['beta_form']:7.4f} "
                  f"{r['pf_understatement_factor']:7.2f}")
    print()
    print(sentence)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
