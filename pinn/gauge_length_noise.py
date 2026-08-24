"""What the gauge length of Section 7.7 does to the error budget it sits in.

Section 7.7 establishes that a bonded fiber must average strain over at
least about nine tenths of a crack spacing, 260 mm on this member, before
the constitutive map is applied, or the phase of the crack pattern swings
the recovered value over 55 to 89 per cent of the section. That requirement
was stated and then, in the first version of this study, asserted to leave
the noise results of Table 2 standing. It does not, and the reason is
geometric.

The cut strip is 100 mm wide and its four gauge stations span 67 mm. An
instrument that averages over 260 mm draws each of those readings from
+-130 mm about its own station, so two stations 25 mm apart share about
ninety per cent of their averaging window. Three things follow:

  * independent gauge noise is not a model the required instrument can
    deliver, at any amplitude;
  * an exponential covariance with a 150 mm correlation length is shorter
    than the averaging window itself, so it understates the correlation;
  * the strip holds roughly one independent reading rather than four.

This module replaces the assumed covariance with the one the instrument
imposes, and it does so by construction rather than by assuming a kernel.
White readout noise boxcar-averaged over a length l has, exactly, the
triangular autocorrelation 1 - |dx|/l on |dx| < l. Generating the field
that way makes the correlation a consequence of the stated gauge length
instead of a second free parameter. Rows at different depths are separate
sensors and are drawn independently.

Two questions are answered separately, because they have different causes:

  bias    the instrument smears the strain field before the map, and the
          band force rises steeply through the shear span, so averaging
          over 260 mm might displace the answer even with no noise at all.
          Measured by running the identification on the pre-averaged field.

  spread  the readings inside one strip are no longer independent, so
          whatever averaging the strip integral was doing is gone.
          Measured by re-running Table 2 and the detection curve under the
          triangular covariance.

Run:  /usr/local/bin/python3.12 gauge_length_noise.py
      -> figures/gauge_length_noise.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figdata as FD                                                       # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "gauge_length_noise.json"

DELTA = "3.5"
THETA = [0.0, 0.10, 0.20, 0.30, 0.40]
NOISE = 0.05                  # of the mean band strain, as everywhere else
N_TABLE = 50                  # matches Table 2
N_POD = 200                   # matches the detection curve
PFA = 0.05
ARM = 370.0

# the requirement of Section 7.7, and the two neighbors that bracket it
ELL = (0.0, 130.0, 260.0)     # mm; 0 = point gauge, the old assumption
ELL_REQ = 260.0               # 0.9 s_rm at the most deteriorated state
DX_RAW = 5.0                  # the fine grid the boxcar average runs on


# ----------------------------------------------------------------------
def rows_of(cy):
    """Element rows, which are the separate sensor lines."""
    return np.unique(np.round(cy, 6))


def gauge_average(cx, cy, a, ell, band_only=True):
    """The instrument: strain averaged over `ell` along the fiber, before
    the constitutive map is applied.

    Restricted to the tie band, because that is where the fiber is: it is
    bonded to the tie bar and reads strain along it. Averaging the
    compression zone as well would smear the lever arm, which no fiber on
    the soffit can do, and it is what made the first version of this
    comparison report a bias shift four times too large.
    """
    if ell <= 0.0:
        return a
    out = a.copy()
    sel = (cy < FD.BAND) if band_only else np.ones(cy.shape, bool)
    for y in rows_of(cy[sel]):
        m = np.isclose(cy, y) & sel
        xs, vals = cx[m], a[m]
        out[m] = np.array([vals[np.abs(xs - x) <= 0.5 * ell].mean()
                           for x in xs])
    return out


def gauge_noise(rng, cx, cy, sd, ell):
    """Readout noise as a gauge-length instrument delivers it.

    White noise boxcar-averaged over `ell` has the triangular correlation
    1 - |dx|/ell exactly, so the field is generated that way rather than
    from an assumed kernel: the correlation is then a consequence of the
    gauge length rather than a second parameter. The average is rescaled
    by sqrt(n) so the reported reading keeps the stated amplitude, which
    is the convention every other noise model in this study uses.
    """
    if ell <= 0.0:
        return rng.normal(0.0, sd, cx.shape)
    n = max(1, int(round(ell / DX_RAW)))
    xs = np.arange(cx.min() - ell, cx.max() + ell, DX_RAW)
    k = np.ones(n) / n
    out = np.empty_like(cx)
    for y in rows_of(cy):
        m = np.isclose(cy, y)
        w = rng.standard_normal(xs.size + n - 1)
        f = np.convolve(w, k, mode="valid")[: xs.size] * np.sqrt(n)
        out[m] = sd * np.interp(cx[m], xs, f)
    return out


def selftest(rng):
    """The generated field must actually have the triangular correlation,
    or every number below is about a different instrument."""
    n = max(1, int(round(ELL_REQ / DX_RAW)))
    xs = np.arange(0.0, 4000.0, DX_RAW)
    k = np.ones(n) / n
    acc = np.zeros(n + 1)
    reps = 400
    for _ in range(reps):
        w = rng.standard_normal(xs.size + n - 1)
        f = np.convolve(w, k, mode="valid")[: xs.size] * np.sqrt(n)
        f -= f.mean()
        for j in range(n + 1):
            acc[j] += float(np.mean(f[: f.size - j] * f[j:]) / np.var(f))
    got = acc / reps
    want = np.maximum(0.0, 1.0 - np.arange(n + 1) / n)
    err = float(np.max(np.abs(got - want)))
    print(f"  self test: triangular correlation reproduced to {err:.3f} "
          f"over one gauge length")
    # and the number the argument turns on
    share = 1.0 - 25.0 / ELL_REQ
    print(f"  two stations 25 mm apart share {100 * share:.0f} % of their "
          f"{ELL_REQ:.0f} mm window")
    return err


# ----------------------------------------------------------------------
def load(d, th):
    k = f"u_{th:.2f}_{DELTA}"
    lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
    cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], FD.NX, FD.NY)
    return cx, cy, ex, ey, gxy, lam


def recover(prob, area, cx, cy, comps, lam):
    return FD.recover_band(prob, cx, cy, comps[0], comps[1], comps[2],
                           area, lam, ARM)[0]


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    out = {"what": "the gauge length of Section 7.7 priced in the currency "
                   "of Table 2 and the detection curve",
           "ell_required_mm": ELL_REQ,
           "noise_amplitude": NOISE,
           "correlation": "triangular, support = gauge length, generated by "
                          "boxcar-averaging white noise rather than assumed"}

    print("Does the instrument Section 7.7 requires leave Table 2 standing?")
    print("=" * 70)
    rng = np.random.default_rng(11)
    out["selftest_corr_err"] = selftest(rng)

    # ---- 1. bias: the instrument smears the field before the map --------
    print("\n1. the field pre-averaged over the gauge length, no noise")
    print(f"{'theta':>7}" + "".join(f"{f'l={e:.0f}':>12}" for e in ELL))
    bias = {}
    for th in THETA:
        cx, cy, ex, ey, gxy, lam = load(d, th)
        row = []
        for e in ELL:
            comps = [gauge_average(cx, cy, ex, e), ey, gxy]
            row.append(recover(prob, area, cx, cy, comps, lam))
        bias[f"{th:.2f}"] = [None if not np.isfinite(v) else float(v)
                             for v in row]
        print(f"{th:>7.2f}" + "".join(
            "        none" if not np.isfinite(v) else f"{v:>12.4f}"
            for v in row))
    out["noiseless_by_gauge_length"] = {"ell_mm": list(ELL), "theta": bias}

    dd = [bias[f"{t:.2f}"][2] for t in THETA if bias[f"{t:.2f}"][2] is not None
          and bias[f"{t:.2f}"][0] is not None]
    d0 = [bias[f"{t:.2f}"][0] for t in THETA if bias[f"{t:.2f}"][2] is not None
          and bias[f"{t:.2f}"][0] is not None]
    shift = [100 * (a - b) for a, b in zip(dd, d0)]
    print(f"   pre-averaging moves the noiseless answer by "
          f"{min(shift):+.2f} to {max(shift):+.2f} points of section")
    out["preaverage_shift_pp"] = [float(min(shift)), float(max(shift))]

    # ---- 2. spread: Table 2 under the covariance the gauge imposes ------
    print(f"\n2. Table 2 re-run at {NOISE:.0%} noise, {N_TABLE} realizations")
    print(f"{'theta':>7}" + "".join(f"{f'l={e:.0f} mm':>18}" for e in ELL))
    rng = np.random.default_rng(3)
    table = {}
    for th in THETA:
        cx, cy, ex, ey, gxy, lam = load(d, th)
        scale = float(np.abs(ex[cy < FD.BAND]).mean())
        sd = NOISE * scale
        cells, rec_all = [], {}
        for e in ELL:
            base = [gauge_average(cx, cy, ex, e), ey, gxy]
            got = []
            for _ in range(N_TABLE):
                comps = [b + gauge_noise(rng, cx, cy, sd, e) for b in base]
                got.append(recover(prob, area, cx, cy, comps, lam))
            got = np.array(got)
            ok = got[np.isfinite(got)]
            rec_all[f"{e:.0f}"] = {
                "mean": None if not ok.size else float(ok.mean()),
                "sd": None if not ok.size else float(ok.std()),
                "n_admissible": int(ok.size), "n": N_TABLE}
            cells.append("             none" if not ok.size else
                         f"{ok.mean():.3f}+-{ok.std():.3f} ({ok.size:2d})")
        table[f"{th:.2f}"] = rec_all
        print(f"{th:>7.2f}" + "".join(f"{c:>18}" for c in cells))
    out["table2_by_gauge_length"] = table

    for e in ELL:
        sds = [table[f"{t:.2f}"][f"{e:.0f}"]["sd"] for t in THETA[1:]
               if table[f"{t:.2f}"][f"{e:.0f}"]["sd"] is not None]
        if sds:
            print(f"   l = {e:>5.0f} mm: spread on the deteriorated states "
                  f"{100*min(sds):.1f} to {100*max(sds):.1f} points")

    # ---- 3. detection, at matched false-alarm rate ----------------------
    print(f"\n3. detection at {PFA:.0%} false alarms, {N_POD} realizations")
    rng = np.random.default_rng(5)
    pod = {}
    for e in ELL:
        cx, cy, ex, ey, gxy, lam = load(d, 0.0)
        scale = float(np.abs(ex[cy < FD.BAND]).mean())
        sd = NOISE * scale
        base = [gauge_average(cx, cy, ex, e), ey, gxy]
        sound = []
        for _ in range(N_POD):
            comps = [b + gauge_noise(rng, cx, cy, sd, e) for b in base]
            r = recover(prob, area, cx, cy, comps, lam)
            sound.append(r if np.isfinite(r) else 0.0)
        thr = float(np.quantile(sound, 1.0 - PFA))
        curve = {}
        for th in THETA[1:]:
            cx, cy, ex, ey, gxy, lam = load(d, th)
            base = [gauge_average(cx, cy, ex, e), ey, gxy]
            hits = 0
            for _ in range(N_POD):
                comps = [b + gauge_noise(rng, cx, cy, sd, e) for b in base]
                r = recover(prob, area, cx, cy, comps, lam)
                if np.isfinite(r) and r > thr:
                    hits += 1
            curve[f"{th:.2f}"] = hits / N_POD
        pod[f"{e:.0f}"] = {"threshold": thr, "pod": curve}
        print(f"   l = {e:>5.0f} mm: threshold {thr:.3f}, "
              + " ".join(f"P({k})={v:.2f}" for k, v in curve.items()))
    out["pod_by_gauge_length"] = pod

    # ---- 4. the design rule that follows -------------------------------
    print("\n4. what it means for a sensor plan")
    strip = 2 * FD.BAND_W
    print(f"   the {strip:.0f} mm strip holds "
          f"{max(1.0, strip / ELL_REQ):.2f} independent readings at "
          f"l = {ELL_REQ:.0f} mm, against 4 stations")
    for target in (4, 8, 16):
        print(f"   {target:2d} independent readings need "
              f"{target * ELL_REQ / 1000:.1f} m of instrumented tie")
    out["design_rule"] = {
        "strip_mm": float(strip),
        "independent_readings_in_strip": float(max(1.0, strip / ELL_REQ)),
        "instrumented_length_for_n_independent_m":
            {str(n): n * ELL_REQ / 1000.0 for n in (4, 8, 16)}}

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
