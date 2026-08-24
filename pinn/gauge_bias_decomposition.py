"""Why does meeting the gauge-length requirement move the bias?

Section 7.8 reported that averaging the strain over 260 mm before the
constitutive map moves the noiseless recovery up by about three points of
the section, and attributed it to sawtooth removal. That attribution was
not earned. The field the averaging was applied to is the smeared finite
element field of fields_theta.npz, which has no sawtooth in it: the
sawtooth exists only in Section 7.7, where it is synthesised by applying a
tension chord profile to that same smeared strain. So the reported shift
could equally be the instrument smearing the along-span GRADIENT of band
strain, which Fig. 3b shows is steep through the shear span.

The two mechanisms are separable, and they transfer differently:

  gradient smearing   depends on the geometry and the load path, so it does
                      not carry to another member without recomputation
  sawtooth removal    depends on the crack spacing and the constitutive
                      map, so it carries wherever the tension chord model
                      does

They are separable here because `sawtooth.fiber_reading` applies the gauge
as a quadrature over the tension chord profile at fixed element strain: it
averages the sawtooth and leaves the along-span gradient untouched. A
boxcar over the element grid does the opposite. So:

  A  smooth field, point reading           the Table 2 baseline
  B  smooth field, boxcar over l           A + gradient smearing
  C  sawtooth, point gauge, phase mean     the Section 7.7 instrument
  D  sawtooth, gauge l, phase mean         C + sawtooth removal
  E  sawtooth, gauge l, then boxcar        both, as a real fiber delivers

and the decomposition is checked against itself by requiring

  (E - C)  ~  (B - A) + (D - C).

Every sawtooth cell is a mean over the crack phase, not one realization,
because the phase is unknown and a single draw of it is not a result.

Run:  /usr/local/bin/python3.12 gauge_bias_decomposition.py
      -> figures/gauge_bias_decomposition.json
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
import sawtooth as SW                                                      # noqa: E402
import tcm                                                                # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains, bracket_root                    # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "gauge_bias_decomposition.json"

DELTA = "3.5"
THETA = [0.10, 0.20, 0.30, 0.40]
ELL = 260.0                    # 0.9 s_rm at the most deteriorated state
ARM = 370.0
N_PHASE = 24                   # phases per state; the mean is what is quoted
GRID = np.linspace(-0.70, 0.70, 281)


def boxcar(cx, cy, a, ell):
    """Average along the fiber over `ell`: the along-span part.

    Restricted to the tie band, and applied to the axial component alone,
    because that is the instrument Section 7.7 specifies and the one
    `sawtooth.fiber_reading` models: a fiber bonded to the tie bar reads
    strain along the bar. It does not read eps_y, it does not read the
    shear, and it is nowhere near the compression zone that sets the lever
    arm. Averaging those as well would be a different instrument, and it
    was the error in the first version of this comparison.
    """
    if ell <= 0.0:
        return a
    out = a.copy()
    inb = cy < FD.BAND
    for y in np.unique(np.round(cy[inb], 6)):
        m = np.isclose(cy, y) & inb
        xs, v = cx[m], a[m]
        out[m] = np.array([v[np.abs(xs - x) <= 0.5 * ell].mean() for x in xs])
    return out


def recover(prob, area, cx, cy, ex, ey, gxy, lam):
    m_req = lam * prob.P / 2.0 * (FD.X_CUT - ARM) / 1e6
    f = np.array([FD.band_couple(prob, cx, cy, ex, ey, gxy, area, q)[2] - m_req
                  for q in GRID])
    return bracket_root(f, GRID)


def phase_mean(fn, s_r):
    """Mean over a full crack spacing, endpoint excluded so no phase counts
    twice. NaNs are carried, not dropped: a cell that fails to root at some
    phases is reported with its admissible fraction rather than averaged
    over the survivors."""
    ph = np.linspace(0.0, s_r, N_PHASE, endpoint=False)
    v = np.array([fn(float(p)) for p in ph])
    ok = np.isfinite(v)
    return (float(np.mean(v[ok])) if ok.any() else float("nan"),
            float(np.std(v[ok])) if ok.any() else float("nan"),
            float(ok.mean()))


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    f_ct = tcm.f_ctm(prob.mat.fc)

    print("Which mechanism moves the bias when the gauge length is met?")
    print("=" * 72)
    print(f"  gauge length {ELL:.0f} mm, {N_PHASE} phases per state, "
          f"noiseless throughout\n")
    hdr = (f"{'theta':>6}{'A point':>10}{'B boxcar':>10}{'C saw pt':>10}"
           f"{'D saw ell':>11}{'E both':>10}"
           f"{'grad':>8}{'saw':>8}{'sum':>8}{'E-C':>8}")
    print(hdr)
    rows = []
    for th in THETA:
        k = f"u_{th:.2f}_{DELTA}"
        lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
        cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], FD.NX, FD.NY)
        s_r, _phi = SW.spacing(th, prob, f_ct)

        A = recover(prob, area, cx, cy, ex, ey, gxy, lam)
        B = recover(prob, area, cx, cy, boxcar(cx, cy, ex, ELL), ey, gxy, lam)

        def saw(phase, gauge, box):
            e2, _ = SW.fiber_reading(cx, cy, ex, th, FD.X_CUT + phase,
                                     gauge, prob, f_ct)
            if box:
                e2 = boxcar(cx, cy, e2, ELL)
            return recover(prob, area, cx, cy, e2, ey, gxy, lam)

        C, C_sd, C_ok = phase_mean(lambda p: saw(p, 0.0, False), s_r)
        D, D_sd, D_ok = phase_mean(lambda p: saw(p, ELL, False), s_r)
        E, E_sd, E_ok = phase_mean(lambda p: saw(p, ELL, True), s_r)

        grad, sawr = 100 * (B - A), 100 * (D - C)
        tot, meas = grad + sawr, 100 * (E - C)
        print(f"{th:>6.2f}{A:>10.4f}{B:>10.4f}{C:>10.4f}{D:>11.4f}{E:>10.4f}"
              f"{grad:>+8.2f}{sawr:>+8.2f}{tot:>+8.2f}{meas:>+8.2f}")
        rows.append(dict(theta=th, s_rm_mm=float(s_r),
                         A_smooth_point=float(A), B_smooth_boxcar=float(B),
                         C_saw_point=float(C), C_sd=C_sd, C_admissible=C_ok,
                         D_saw_gauge=float(D), D_sd=D_sd, D_admissible=D_ok,
                         E_saw_gauge_boxcar=float(E), E_sd=E_sd,
                         E_admissible=E_ok,
                         gradient_pp=grad, sawtooth_pp=sawr,
                         sum_pp=tot, measured_pp=meas))

    g = np.array([r["gradient_pp"] for r in rows])
    sw = np.array([r["sawtooth_pp"] for r in rows])
    tot = g + sw
    print(f"\n  gradient smearing : {g.min():+.2f} to {g.max():+.2f} pp "
          f"(mean {g.mean():+.2f})")
    print(f"  sawtooth removal  : {sw.min():+.2f} to {sw.max():+.2f} pp "
          f"(mean {sw.mean():+.2f})")
    share = 100 * np.abs(g).mean() / (np.abs(g).mean() + np.abs(sw).mean())
    print(f"  gradient share of the total shift: {share:.0f} %")
    err = np.array([abs(r["sum_pp"] - r["measured_pp"]) for r in rows])
    print(f"  decomposition closes to {err.max():.2f} pp worst case")

    # ---- knock-on: averaging attenuates a gradient, and the arm-free form
    #      IS a gradient, so its differenced slope must be re-checked -------
    print("\n  the arm-free pair reads a difference between two stations, "
          "which is\n  a gradient, so averaging attenuates the signal it "
          "lives on:")
    th = 0.20
    lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
    cx, cy, ex, ey, gxy = element_strains(d["xy"], d[f"u_{th:.2f}_{DELTA}"],
                                          FD.NX, FD.NY)
    pairs = []
    for sp in (260.0, 290.0, 450.0):
        x1, x2 = 700.0 - 0.0, 700.0 + sp
        if x2 > 1150.0:
            x1, x2 = 400.0, 400.0 + sp
        def couple_at(xc, e):
            old = FD.X_CUT; FD.X_CUT = xc
            _T, _z, C = FD.band_couple(prob, cx, cy, e, ey, gxy, area, th)
            FD.X_CUT = old
            return C
        raw = couple_at(x2, ex) - couple_at(x1, ex)
        av = boxcar(cx, cy, ex, ELL)
        smoothed = couple_at(x2, av) - couple_at(x1, av)
        att = 100 * (1 - abs(smoothed) / max(abs(raw), 1e-12))
        pairs.append(dict(spacing_mm=sp, x1=x1, x2=x2,
                          differenced_couple_raw_kNm=float(raw),
                          differenced_couple_averaged_kNm=float(smoothed),
                          attenuation_pct=float(att)))
        print(f"    pair {x1:.0f}-{x2:.0f} mm ({sp:.0f} apart): "
              f"differenced couple {raw:+8.2f} -> {smoothed:+8.2f} kN m, "
              f"attenuated {att:5.1f} %")

    OUT.write_text(json.dumps(
        {"what": "does the gauge length move the bias by smearing the "
                 "along-span gradient or by removing the sawtooth",
         "gauge_length_mm": ELL, "n_phase": N_PHASE,
         "cells": "A smooth/point, B smooth/boxcar, C saw/point, "
                  "D saw/gauge, E saw/gauge+boxcar; all phase means",
         "rows": rows,
         "gradient_pp_range": [float(g.min()), float(g.max())],
         "sawtooth_pp_range": [float(sw.min()), float(sw.max())],
         "gradient_share_pct": float(share),
         "decomposition_closure_pp": float(err.max()),
         "arm_free_attenuation": pairs}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
