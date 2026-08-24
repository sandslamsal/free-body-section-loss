"""Reliability updating for the identified tie-corrosion state.

This study closes the loop from identification to decision: a corrosion
reading is only worth publishing if it changes a computed failure
probability, so the chain below carries the identified loss ratio theta
through a prior, a likelihood, a posterior, and a limit state, and reports
the reliability index each stage implies.

Modeling choices, and why:

Capacity curve. The reference solver stores equilibrium load factors at
prescribed mid-span deflections; the largest common station, delta = 7.0 mm,
stands in for capacity. That is a proxy, not a converged limit point, so the
increment from delta = 5.0 to 7.0 mm is computed per theta: a small increment
means the curve has flattened and the proxy is honest. The capacity is fitted
as a polynomial in theta; linear is preferred because a two-coefficient law
is what a code calibration would carry, and the quadratic is used only if the
linear residual says otherwise.

Prior. Uniform corrosion of the four tie bars at a lognormal corrosion
current density, median 1.0 uA/cm2 and COV 0.5, converted at 0.0116 mm per
year per uA/cm2 (Andrade, RILEM TC 154-EMC) over a 50 year exposure. The
current density is lognormal because field corrosion rates are positive and
right-skewed, and the COV of 0.5 reflects the scatter RILEM reports between
nominally identical exposures. The area loss of a bar of initial radius r0
under a radius loss r_loss is theta = 1 - (1 - r_loss/r0)^2, clipped to
[0, 0.7] because beyond that the section model itself has lost meaning. The
prior density is built by pushing 10^6 lognormal samples through this
transform onto a fine theta grid, which keeps the clip mass and the
nonlinear stretch of the transform without any distributional assumption
on theta itself.

Likelihood. The identification does not return the true theta; its measured
mean response is theta_hat = 0.830 theta - 0.005 with an aleatory standard
deviation of 0.025 at 5 percent strain noise (both taken from the sampling
experiments of this study). The calibrated likelihood therefore evaluates
the observed reading against 0.830 theta - 0.005. A naive analyst who takes
the reading at face value uses the identity response instead; both
posteriors are computed on the same grid so their difference is exactly the
cost of not calibrating the bias.

Limit state. g = X_R * lambda_R(theta) - S. X_R is a lognormal resistance
model uncertainty, median 1.0 and COV 0.10, the conventional allowance for
what the mechanical model itself gets wrong. S is the annual-maximum load
factor, Gumbel with mean 0.50 and COV 0.15, because annual maxima of
sustained-plus-transient load effects are extreme values and the Gumbel is
the standard choice. Monte Carlo with 10^7 samples is the reference answer;
FORM with theta replaced by a normal fitted to each posterior's mean and
standard deviation is reported alongside because FORM is the convention in
which code targets are stated. Common random numbers are used across cases
so that differences between betas are not noise.

Run:  python reliability.py     (writes ../figures/reliability.json)
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

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "reliability.json"

THETAS = [0.00, 0.10, 0.20, 0.30, 0.40]
DELTA_CAP = 7.0            # capacity proxy station (mm)
DELTA_PREV = 5.0           # previous station, to check the proxy

# prior physics
RATE = 0.0116              # mm/year radius loss per uA/cm2 (Andrade)
YEARS = 50.0
AS_BAND = 0.012 * 150.0 * 300.0          # 540 mm2 of band steel
R0 = math.sqrt((AS_BAND / 4.0) / math.pi)  # four equal bars
ICORR_MEDIAN = 1.0         # uA/cm2
ICORR_COV = 0.5
THETA_MAX = 0.7
N_PRIOR_SAMPLES = 1_000_000
N_GRID = 2001

# identification response (measured in this study)
IDENT_SLOPE = 0.834
IDENT_INTERCEPT = -0.0099
IDENT_SD = 0.025
THETA_TRUE = 0.20
THETA_HAT_OBS = 0.157      # what the identification returns at the true 0.20

# limit state
XR_MEDIAN = 1.0
XR_COV = 0.10
S_MEAN = 0.50
S_COV = 0.15
N_MC = 10_000_000
SEED = 20260821


# ----------------------------------------------------------------------
# standard-normal helpers (scipy if present, scalar fallbacks otherwise)
# ----------------------------------------------------------------------
try:
    from scipy.special import ndtr as _ndtr, ndtri as _ndtri

    def phi_cdf(x: float) -> float:
        return float(_ndtr(x))

    def phi_inv(p: float) -> float:
        return float(_ndtri(p))

except ImportError:  # pragma: no cover

    def phi_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def phi_inv(p: float) -> float:
        """Acklam's rational approximation, refined by one Halley step."""
        if not 0.0 < p < 1.0:
            raise ValueError("p out of (0,1)")
        a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
             4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01,
             2.445134137142996e+00, 3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        elif p <= phigh:
            q = p - 0.5
            r = q * q
            x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        e = phi_cdf(x) - p
        u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
        return x - u / (1 + x * u / 2)


def norm_pdf(x, mu, sd):
    z = (np.asarray(x, dtype=float) - mu) / sd
    return np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


# ----------------------------------------------------------------------
# step 1: capacity curve
# ----------------------------------------------------------------------
def capacity_curve():
    z = np.load(FIELDS)

    def lam_at(th, d):
        return float(np.asarray(z[f"lam_{th:.2f}_{d:.1f}"]).reshape(-1)[0])

    deltas_all = [1.0, 2.0, 3.5, 5.0, 7.0]
    lam_cap = np.array([lam_at(th, DELTA_CAP) for th in THETAS])
    lam_prev = np.array([lam_at(th, DELTA_PREV) for th in THETAS])
    lam_max = np.array([max(lam_at(th, d) for d in deltas_all)
                        for th in THETAS])
    incr = lam_cap - lam_prev
    incr_pct = 100.0 * incr / lam_cap

    th = np.array(THETAS)
    lin = np.polyfit(th, lam_cap, 1)
    resid = lam_cap - np.polyval(lin, th)
    ss_tot = float(np.sum((lam_cap - lam_cap.mean()) ** 2))
    r2_lin = 1.0 - float(np.sum(resid ** 2)) / ss_tot

    quad = np.polyfit(th, lam_cap, 2)
    resid_q = lam_cap - np.polyval(quad, th)
    r2_quad = 1.0 - float(np.sum(resid_q ** 2)) / ss_tot

    use_quad = r2_lin < 0.99
    coef = quad if use_quad else lin

    def lam_R(theta):
        return np.clip(np.polyval(coef, theta), 0.0, None)

    info = {
        "theta": THETAS,
        "lam_delta7": lam_cap.tolist(),
        "lam_delta5": lam_prev.tolist(),
        "increment_5_to_7": incr.tolist(),
        "increment_5_to_7_pct": incr_pct.tolist(),
        "lam_max_over_deltas": lam_max.tolist(),
        "proxy_vs_peak_pct": (100.0 * (lam_cap - lam_max) / lam_max).tolist(),
        "fit": "quadratic" if use_quad else "linear",
        "coef_highest_first": coef.tolist(),
        "r2_linear": r2_lin,
        "r2_quadratic": r2_quad,
    }
    return lam_R, info


# ----------------------------------------------------------------------
# step 2: prior on theta
# ----------------------------------------------------------------------
def prior_on_grid(rng):
    sigma_ln = math.sqrt(math.log(1.0 + ICORR_COV ** 2))
    mu_ln = math.log(ICORR_MEDIAN)
    icorr = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=N_PRIOR_SAMPLES)
    r_loss = RATE * YEARS * icorr
    theta = 1.0 - (1.0 - r_loss / R0) ** 2
    theta = np.clip(theta, 0.0, THETA_MAX)

    edges = np.linspace(0.0, THETA_MAX, N_GRID + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    density, _ = np.histogram(theta, bins=edges, density=True)

    stats = {
        "mean": float(theta.mean()),
        "sd": float(theta.std()),
        "p_clipped_at_max": float(np.mean(theta >= THETA_MAX - 1e-12)),
        "p_theta_gt_0.4": float(np.mean(theta > 0.4)),
        "r0_mm": R0,
    }
    return grid, density, stats


# ----------------------------------------------------------------------
# step 3: posteriors on the grid
# ----------------------------------------------------------------------
def grid_bayes(grid, prior, like):
    post = prior * like
    dth = grid[1] - grid[0]
    post /= np.sum(post) * dth
    mean = float(np.sum(grid * post) * dth)
    sd = float(math.sqrt(max(np.sum((grid - mean) ** 2 * post) * dth, 0.0)))
    return post, mean, sd


# ----------------------------------------------------------------------
# step 4a: Monte Carlo
# ----------------------------------------------------------------------
def sample_from_grid(grid, density, u):
    dth = grid[1] - grid[0]
    cdf = np.cumsum(density) * dth
    cdf /= cdf[-1]
    return np.interp(u, cdf, grid)


def mc_case(lam_R, theta_samples, xr, s):
    g = xr * lam_R(theta_samples) - s
    pf = float(np.mean(g < 0.0))
    n = g.size
    se_pf = math.sqrt(max(pf * (1.0 - pf), 1e-300) / n)
    beta = -phi_inv(pf) if 0.0 < pf < 1.0 else float("inf")
    # delta-method standard error on beta
    pdf_at = math.exp(-0.5 * beta * beta) / math.sqrt(2 * math.pi)
    se_beta = se_pf / pdf_at if pdf_at > 0 else float("nan")
    return pf, se_pf, beta, se_beta


# ----------------------------------------------------------------------
# step 4b: FORM (HLRF, exact marginal transforms, theta as fitted normal)
# ----------------------------------------------------------------------
def form_beta(lam_R, theta_mean, theta_sd):
    sig_lnR = math.sqrt(math.log(1.0 + XR_COV ** 2))
    s_sd = S_COV * S_MEAN
    s_scale = s_sd * math.sqrt(6.0) / math.pi
    s_loc = S_MEAN - 0.5772156649015329 * s_scale
    with_theta = theta_sd > 0.0
    ndim = 3 if with_theta else 2

    def g_of_u(u):
        xr = math.exp(sig_lnR * u[0])            # lognormal, median 1
        p = min(max(phi_cdf(u[1]), 1e-15), 1.0 - 1e-15)
        s = s_loc - s_scale * math.log(-math.log(p))   # Gumbel max
        th = theta_mean + (theta_sd * u[2] if with_theta else 0.0)
        return xr * float(lam_R(th)) - s

    u = np.zeros(ndim)
    eps = 1e-6
    for _ in range(200):
        g0 = g_of_u(u)
        grad = np.zeros(ndim)
        for i in range(ndim):
            up = u.copy()
            up[i] += eps
            grad[i] = (g_of_u(up) - g0) / eps
        gn2 = float(grad @ grad)
        if gn2 == 0.0:
            break
        u_new = ((grad @ u - g0) / gn2) * grad
        if np.linalg.norm(u_new - u) < 1e-8:
            u = u_new
            break
        u = u_new
    beta = float(np.linalg.norm(u))
    if g_of_u(np.zeros(ndim)) < 0.0:
        beta = -beta
    return beta


# ----------------------------------------------------------------------
def main():
    lam_R, cap_info = capacity_curve()

    rng = np.random.default_rng(SEED)
    grid, prior, prior_stats = prior_on_grid(rng)

    like_cal = norm_pdf(THETA_HAT_OBS, IDENT_SLOPE * grid + IDENT_INTERCEPT,
                        IDENT_SD)
    like_naive = norm_pdf(THETA_HAT_OBS, grid, IDENT_SD)
    post_cal, m_cal, s_cal = grid_bayes(grid, prior, like_cal)
    post_naive, m_naive, s_naive = grid_bayes(grid, prior, like_naive)

    dth = grid[1] - grid[0]
    prior_norm = prior / (np.sum(prior) * dth)
    m_prior = float(np.sum(grid * prior_norm) * dth)
    s_prior = float(math.sqrt(np.sum((grid - m_prior) ** 2 * prior_norm) * dth))

    # common random numbers across all cases
    xr = np.exp(math.sqrt(math.log(1.0 + XR_COV ** 2))
                * rng.standard_normal(N_MC))
    s_sd = S_COV * S_MEAN
    s_scale = s_sd * math.sqrt(6.0) / math.pi
    s_loc = S_MEAN - 0.5772156649015329 * s_scale
    s = s_loc - s_scale * np.log(-np.log(rng.random(N_MC)))
    u_theta = rng.random(N_MC)

    cases = {
        "prior": (sample_from_grid(grid, prior_norm, u_theta),
                  m_prior, s_prior),
        "posterior_calibrated": (sample_from_grid(grid, post_cal, u_theta),
                                 m_cal, s_cal),
        "posterior_naive": (sample_from_grid(grid, post_naive, u_theta),
                            m_naive, s_naive),
        "point_true_0.20": (np.full(N_MC, THETA_TRUE), THETA_TRUE, 0.0),
        "point_reading_0.16": (np.full(N_MC, THETA_HAT_OBS),
                               THETA_HAT_OBS, 0.0),
    }

    results = {}
    for name, (th_s, th_m, th_sd) in cases.items():
        pf, se_pf, beta, se_beta = mc_case(lam_R, th_s, xr, s)
        b_form = form_beta(lam_R, th_m, th_sd)
        results[name] = {
            "theta_mean": th_m, "theta_sd": th_sd,
            "pf_mc": pf, "se_pf_mc": se_pf,
            "beta_mc": beta, "se_beta_mc": se_beta,
            "beta_form": b_form,
        }

    dbeta_post = (results["posterior_naive"]["beta_mc"]
                  - results["posterior_calibrated"]["beta_mc"])
    dbeta_point = (results["point_reading_0.16"]["beta_mc"]
                   - results["point_true_0.20"]["beta_mc"])
    dbeta_post_form = (results["posterior_naive"]["beta_form"]
                       - results["posterior_calibrated"]["beta_form"])
    dbeta_point_form = (results["point_reading_0.16"]["beta_form"]
                        - results["point_true_0.20"]["beta_form"])

    out = {
        "capacity": cap_info,
        "prior": prior_stats,
        "posterior_calibrated": {"mean": m_cal, "sd": s_cal},
        "posterior_naive": {"mean": m_naive, "sd": s_naive},
        "identification": {"slope": IDENT_SLOPE, "intercept": IDENT_INTERCEPT,
                           "sd": IDENT_SD, "theta_hat_obs": THETA_HAT_OBS},
        "limit_state": {"XR_median": XR_MEDIAN, "XR_cov": XR_COV,
                        "S_mean": S_MEAN, "S_cov": S_COV, "n_mc": N_MC},
        "cases": results,
        "headline": {
            "dbeta_naive_minus_calibrated_mc": dbeta_post,
            "dbeta_reading_minus_true_mc": dbeta_point,
            "dbeta_naive_minus_calibrated_form": dbeta_post_form,
            "dbeta_reading_minus_true_form": dbeta_point_form,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))

    print("capacity proxy: lam at delta=7.0, increment from delta=5.0")
    for i, th in enumerate(THETAS):
        print(f"  theta={th:.2f}  lam7={cap_info['lam_delta7'][i]:.4f}"
              f"  lam5={cap_info['lam_delta5'][i]:.4f}"
              f"  incr={cap_info['increment_5_to_7'][i]:+.4f}"
              f" ({cap_info['increment_5_to_7_pct'][i]:+.2f}%)"
              f"  peak={cap_info['lam_max_over_deltas'][i]:.4f}"
              f" (proxy {cap_info['proxy_vs_peak_pct'][i]:+.2f}% vs peak)")
    print(f"fit: {cap_info['fit']}  coef={cap_info['coef_highest_first']}"
          f"  R2_lin={cap_info['r2_linear']:.5f}"
          f"  R2_quad={cap_info['r2_quadratic']:.5f}")
    print(f"prior: mean={prior_stats['mean']:.4f} sd={prior_stats['sd']:.4f}"
          f"  P[clip@0.7]={prior_stats['p_clipped_at_max']:.2e}"
          f"  P[theta>0.4]={prior_stats['p_theta_gt_0.4']:.4f}")
    print(f"posterior calibrated: mean={m_cal:.4f} sd={s_cal:.4f}")
    print(f"posterior naive:      mean={m_naive:.4f} sd={s_naive:.4f}")
    print()
    hdr = (f"{'case':24s} {'theta_m':>8s} {'pf':>12s} {'se_pf':>10s}"
           f" {'beta_mc':>8s} {'se_b':>7s} {'beta_form':>9s}")
    print(hdr)
    for name, r in results.items():
        print(f"{name:24s} {r['theta_mean']:8.4f} {r['pf_mc']:12.3e}"
              f" {r['se_pf_mc']:10.2e} {r['beta_mc']:8.4f}"
              f" {r['se_beta_mc']:7.4f} {r['beta_form']:9.4f}")
    print()
    print(f"headline: naive - calibrated  dbeta = {dbeta_post:+.4f} (MC), "
          f"{dbeta_post_form:+.4f} (FORM)")
    print(f"          reading - true      dbeta = {dbeta_point:+.4f} (MC), "
          f"{dbeta_point_form:+.4f} (FORM)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
