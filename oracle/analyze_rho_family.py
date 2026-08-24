"""Robust peak extraction for the extended tie-loss reference family.

The displacement-controlled secant solver occasionally snaps between
equilibrium basins on the descending branch, producing isolated spikes
that corrupt a naive argmax peak. This script filters each curve of
deepbeam_oracle_rhosweep_ext.json with a 5-point median filter, takes the
limit point from the filtered curve, and stores a monotone display
envelope (running max to the peak, running min after), which is the same
presentation rule the manuscript's other reference curves use.

Outputs:
  deepbeam_rho_family_clean.json   per-loss envelope + robust peaks
  rho_family_clean_diag.png        raw vs envelope diagnostic (viridis)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt


def robust_envelope(delta: np.ndarray, lam: np.ndarray,
                    conv: np.ndarray, jump: float = 0.20,
                    d_min: float = 5.0) -> dict:
    """Limit point of the FOLLOWED branch.

    Median-filter the converged points (kills isolated spikes), then
    truncate at the first basin snap: a single-step change of the
    filtered curve larger than `jump`, which no physical 0.25 mm
    displacement increment produces on this beam (yield steps are
    <= ~0.12). Everything past the snap belongs to a different
    equilibrium branch and is discarded, not smoothed over. The limit
    point is the crest of the trustworthy segment."""
    d = delta[conv]
    l_raw = lam[conv]
    l_med = medfilt(l_raw, kernel_size=5)

    # display truncation: first RAW single-step basin snap (yield steps on
    # this beam are <= ~0.12; snaps are 0.2-0.7)
    dl_raw = np.abs(np.diff(l_raw))
    snaps = np.where((dl_raw > jump) & (d[1:] > d_min))[0]
    n = int(snaps[0]) + 1 if len(snaps) else len(l_med)

    # limit point: FIRST crest of the filtered curve, i.e. the first local
    # maximum followed by a sustained drop (tol below crest for `sustain`
    # consecutive steps before the curve exceeds the crest again). The
    # solver's late re-rise onto the spurious stiff-branch attractor
    # (lambda ~ 2.44 at every rho, physically inadmissible) never
    # satisfies "first", so it is excluded by construction.
    tol, sustain = 0.05, 4
    ipk = None
    for i in range(1, min(n, len(l_med)) - 1):
        if d[i] < d_min or l_med[i] < l_med[i - 1] or l_med[i] < l_med[i + 1]:
            continue
        run = 0
        for j in range(i + 1, len(l_med)):
            if l_med[j] > l_med[i] + tol and run < sustain:
                break
            run = run + 1 if l_med[j] < l_med[i] - tol else 0
            if run >= sustain:
                ipk = i
                break
        if ipk is not None:
            break
    if ipk is None:                             # monotone to the window end
        ipk = int(np.argmax(l_med[:n]))

    # The crest was located on the FILTERED curve, which is right for
    # robustness but wrong for the value: a median filter clips and
    # displaces a genuine peak. Refine the location and read the value
    # from the RAW curve within a +/- 1 mm neighbourhood of the crest.
    w = 4
    lo, hi = max(0, ipk - w), min(len(l_raw), ipk + w + 1)
    ipk_med = ipk
    ipk = lo + int(np.argmax(l_raw[lo:hi]))
    lam_pk = float(l_raw[ipk])

    # display window: keep the softening branch (running-min is immune to
    # upward snaps, medfilt kills isolated dips) until the filtered curve
    # is captured by the spurious stiff-branch attractor, i.e. sits above
    # the crest for two consecutive steps. The window always contains the
    # crest itself.
    n = len(l_med)
    for j in range(max(ipk, ipk_med) + 1, len(l_med) - 1):
        if (l_med[j] > l_med[ipk_med] + tol
                and l_med[j + 1] > l_med[ipk_med] + tol):
            n = j
            break
    n = max(n, ipk + 1)

    d_t, l_t = d[:n], l_med[:n]
    env = l_t.copy()
    env[ipk] = lam_pk                           # crest carries the raw value
    for i in range(1, ipk + 1):                 # ascending: running max
        env[i] = max(env[i], env[i - 1])
    for i in range(ipk + 1, len(env)):          # descending: running min
        env[i] = min(env[i], env[i - 1])
    return {"delta": d_t.tolist(), "lam_env": env.tolist(),
            "delta_all": d.tolist(), "lam_raw": l_raw.tolist(),
            "trunc_delta": float(d_t[-1]),
            "peak_lam": lam_pk, "peak_delta": float(d[ipk]),
            "peak_at_edge": bool(ipk >= n - 3)}


def main() -> None:
    here = Path(__file__).resolve().parent
    src = json.loads((here / "deepbeam_oracle_rhosweep_ext.json").read_text())
    clean = {"rho_nominal": src["rho_nominal"], "nx": src["nx"],
             "ny": src["ny"], "delta_max": src["delta_max"],
             "n_steps": src["n_steps"], "curves": []}

    fig, (ax_fam, ax_pk) = plt.subplots(1, 2, figsize=(11, 4.2))
    losses = [c["loss"] for c in src["curves"]]
    cmap = plt.cm.viridis
    print(f"{'loss':>5}  {'rho_tie':>8}  {'peak lam':>8}  "
          f"{'peak delta':>10}  {'trunc':>6}  {'edge':>5}  held")
    for c in sorted(src["curves"], key=lambda c: c["loss"]):
        r = robust_envelope(np.array(c["delta"]), np.array(c["lam"]),
                            np.array(c["converged"], dtype=bool))
        r.update({"loss": c["loss"], "rho_tie": c["rho_tie"],
                  "held_out": c["held_out"],
                  "wall_time_s": c["wall_time_s"]})
        clean["curves"].append(r)
        col = cmap(c["loss"] / max(losses))
        ax_fam.plot(r["delta_all"], r["lam_raw"], color=col, lw=0.6,
                    alpha=0.30)
        ax_fam.plot(r["delta"], r["lam_env"], color=col, lw=1.6,
                    label=f"{c['loss'] * 100:.0f}%")
        ax_pk.plot(c["loss"] * 100, r["peak_lam"], "o", color=col, ms=6)
        print(f"{c['loss']:5.2f}  {c['rho_tie']:8.5f}  "
              f"{r['peak_lam']:8.3f}  {r['peak_delta']:10.2f}  "
              f"{r['trunc_delta']:6.2f}  {str(r['peak_at_edge']):>5}  "
              f"{c['held_out']}")

    ax_fam.set_xlabel("midspan deflection [mm]")
    ax_fam.set_ylabel(r"load factor $\lambda$")
    ax_fam.set_title("raw (thin) vs envelope (thick)")
    ax_fam.legend(fontsize=7, ncol=2, title="tie loss")
    ax_pk.set_xlabel("tie section loss [%]")
    ax_pk.set_ylabel(r"$\lambda_{\mathrm{peak}}$")
    ax_pk.set_title("robust limit point vs loss")
    fig.tight_layout()
    fig.savefig(here / "rho_family_clean_diag.png", dpi=150)

    with open(here / "deepbeam_rho_family_clean.json", "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n-> deepbeam_rho_family_clean.json, rho_family_clean_diag.png")


if __name__ == "__main__":
    main()
