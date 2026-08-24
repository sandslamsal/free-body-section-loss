"""a90/95 detection thresholds from the POD study (Berens NDE reliability convention).

Convention
----------
In NDE reliability practice (Berens, "NDE Reliability Data Analysis", Metals
Handbook 9th ed., Vol. 17, ASM International, 1989), a detection capability is
quoted not as the flaw size where the *estimated* POD curve crosses 90%
(``a90``), but as the size where the LOWER 95% confidence bound on POD crosses
90% (``a90/95``). The bound matters because each POD point here is estimated
from exactly 200 Bernoulli trials: an observed POD of 0.93 (186/200) has a
one-sided 95% Clopper-Pearson lower bound of only ~0.89, so a state that
*looks* detected 9 times out of 10 is not yet demonstrably so at 95%
confidence. Quoting a90/95 makes the finite-sample uncertainty part of the
claim instead of hiding it.

Method
------
For each noise model, at each damage state theta (fraction of tie-steel loss),
the number of successes is k = round(POD * 200). The exact (Clopper-Pearson)
one-sided lower 95% bound is the 5th percentile of Beta(k, n - k + 1):

    LCB = scipy.stats.beta.ppf(0.05, k, n - k + 1)     for 0 < k < n
    LCB = 0.05**(1/n)                                  for k = n  (exact)
    LCB = 0                                            for k = 0

a90/95 is the smallest loss at which LCB >= 0.90, obtained by linear
interpolation between adjacent states (in percent loss); a90 is the same for
the point estimate itself. If a curve is already above 0.90 at the smallest
tested state (10% loss), the threshold is reported as "<= 10%"; if it never
reaches 0.90 inside [10%, 40%], the value at 40% is reported instead.

Run with /usr/local/bin/python3.12 (default python3 lacks numpy/scipy).
"""

import json
import pathlib

from scipy.stats import beta

N_TRIALS = 200
HERE = pathlib.Path(__file__).resolve().parent
POD_JSON = HERE.parent / "figures" / "pod.json"


def lower_bound_95(k: int, n: int) -> float:
    """One-sided exact (Clopper-Pearson) lower 95% confidence bound on p."""
    if k == 0:
        return 0.0
    if k == n:
        return 0.05 ** (1.0 / n)  # exact closed form for the all-success case
    return float(beta.ppf(0.05, k, n - k + 1))


def crossing(thetas_pct, values, target=0.90):
    """Smallest loss (percent) where `values` reaches `target`, linearly
    interpolated between states. Returns (value, kind) with kind in
    {'at_or_below_first', 'interpolated', 'never'}."""
    if values[0] >= target:
        return thetas_pct[0], "at_or_below_first"
    for (t0, v0), (t1, v1) in zip(zip(thetas_pct, values), zip(thetas_pct[1:], values[1:])):
        if v0 < target <= v1:
            return t0 + (t1 - t0) * (target - v0) / (v1 - v0), "interpolated"
    return None, "never"


def main():
    data = json.loads(POD_JSON.read_text())
    for model in ("independent", "correlated"):
        pods = data[model]["pod"]
        thetas_pct = [float(t) * 100.0 for t in pods]
        pod_vals, lcb_vals, ks = [], [], []
        for t, p in pods.items():
            k = round(float(p) * N_TRIALS)
            ks.append(k)
            pod_vals.append(k / N_TRIALS)
            lcb_vals.append(lower_bound_95(k, N_TRIALS))

        print(f"\n=== {model} noise ===")
        for t, k, p, lo in zip(thetas_pct, ks, pod_vals, lcb_vals):
            print(f"  theta = {t:4.0f}% loss: k = {k:3d}/{N_TRIALS}, "
                  f"POD = {p:.3f}, 95% lower bound = {lo:.4f}")

        a90, kind90 = crossing(thetas_pct, pod_vals)
        a9095, kind95 = crossing(thetas_pct, lcb_vals)

        if kind90 == "never":
            print(f"  a90    : POD never reaches 0.90 in [10%, 40%]; "
                  f"POD at 40% = {pod_vals[-1]:.3f}")
        elif kind90 == "at_or_below_first":
            print(f"  a90    <= {a90:.1f}% loss (POD already {pod_vals[0]:.3f} at the smallest state)")
        else:
            print(f"  a90     = {a90:.1f}% loss (linear interpolation)")

        if kind95 == "never":
            print(f"  a90/95 : lower bound never reaches 0.90 in [10%, 40%]; "
                  f"bound at 40% = {lcb_vals[-1]:.4f}")
        elif kind95 == "at_or_below_first":
            print(f"  a90/95 <= {a9095:.1f}% loss (bound already {lcb_vals[0]:.4f} at the smallest state)")
        else:
            print(f"  a90/95  = {a9095:.1f}% loss (linear interpolation)")


if __name__ == "__main__":
    main()
