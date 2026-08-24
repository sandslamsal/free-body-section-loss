"""How much material-model error is absorbed into the recovered section loss,
priced at every load level rather than at one.

The recovery of Section 6 uses the same solver, the same constitutive and
the same discretization on both sides of the identification, so its accuracy
is a statement about self-consistency before it is one about a structure.
That is an inverse crime, and the honest way to price it is to commit one
deliberately and measure what it costs. Here the fields are generated with
the nominal material and the identification is performed with a perturbed
one: the concrete strength, the yield strength and the steel modulus are
each moved in turn by a realistic amount, and the resulting shift in the
recovered value is reported. Whatever a perturbation buys is model error
being absorbed into theta, which is the quantity an assessment would report
as damage.

The first version of this study ran at one deflection, 3.5 mm, and the
budget quoted its numbers as though they held at every load. They cannot.
The parameter is read through the steel stress in the tie band, and which
material constant sets that stress depends on whether the band has yielded:
at 3.5 mm the strip the reconciliation integrates is entirely past yield, so
the yield strength governs and the modulus is invisible, while at a 1 mm
service deflection the band is wholly elastic and the two must exchange
roles. A budget line that does not name the load level it belongs to is
therefore not a property of the method but of one station on the
equilibrium path. This module sweeps all five stored stations and reports
the fraction of the band past yield beside each, so the mechanism is
visible next to the sensitivity it produces.

Run:  python model_error.py          (all five stations, writes the JSON)
      python model_error.py 3.5      (one station, the original table)
"""
from __future__ import annotations

import dataclasses
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
OUT = HERE.parent / "figures" / "model_error_by_load.json"

DELTA = "3.5"                  # the single station the first version used
DELTAS = ("1.0", "2.0", "3.5", "5.0", "7.0")     # every stored station

# The budget perturbations, unchanged from the single-station study.
CASES = [
    ("nominal",              dict()),
    ("f_c  +10 %",           dict(fc=1.10)),
    ("f_c  -10 %",           dict(fc=0.90)),
    ("f_y  +5 %",            dict(fy=1.05)),
    ("f_y  -5 %",            dict(fy=0.95)),
    ("E_s  +5 %",            dict(Es=1.05)),
    ("E_s  -5 %",            dict(Es=0.95)),
]

# A second, smaller perturbation of the two steel constants. It is not part
# of the budget table; it is there to establish that the shift scales with
# the size of the material error, which is what licenses quoting the term at
# the tolerance the constant is actually known to rather than at five per
# cent.
AUX = [
    ("f_y  +2 %",            dict(fy=1.02)),
    ("E_s  +2 %",            dict(Es=1.02)),
]

# The reference row every shift is measured from, and the two families the
# reversal is a statement about.
REF_CASE = "nominal"
FAMILY = {"f_c": ("f_c  +10 %", "f_c  -10 %"),
          "f_y": ("f_y  +5 %", "f_y  -5 %"),
          "E_s": ("E_s  +5 %", "E_s  -5 %")}


def recover_with(prob, cx, cy, ex, ey, gxy, area, lam):
    """Section loss reconciled from band strain at the measured reaction arm."""
    return FD.recover_band(prob, cx, cy, ex, ey, gxy, area, lam, 370.0)[0]


def perturbed(base, pert):
    """The identification model, with one material constant moved."""
    mat = dataclasses.replace(
        base.mat,
        fc=base.mat.fc * pert.get("fc", 1.0),
        fy=base.mat.fy * pert.get("fy", 1.0),
        Es=base.mat.Es * pert.get("Es", 1.0))
    return dataclasses.replace(base, mat=mat)


def yield_fractions(prob, ex, cx, cy):
    """Share of the tie band past yield, two ways.

    The first is over the whole band, which is the quantity the text
    already quotes; the second is over the strip that stands for the cut,
    which is the only steel the reconciliation actually integrates and is
    therefore the one that decides which material constant the recovered
    value responds to. The third is the share of that strip lying within
    five per cent of the yield strain, which is what makes a station
    respond differently to the two directions of the same perturbation: a
    perturbation reaches only the elements whose branch it changes.
    """
    inb = cy < FD.BAND
    cut = inb & (np.abs(cx - FD.X_CUT) < FD.BAND_W)
    lim = prob.mat.fy / prob.mat.Es
    knee = float(((ex[cut] > 0.95 * lim) & (ex[cut] < 1.05 * lim)).mean())
    return (float((ex[inb] > lim).mean()), float((ex[cut] > lim).mean()), knee)


def load_strains(d, thetas, deltas):
    """Element strains for every stored state, computed once.

    The kinematics do not depend on the identification model, so the same
    strain field serves every perturbation; recomputing it per case would
    dominate the run time and change nothing.
    """
    cache = {}
    for dl in deltas:
        for th in thetas:
            k = f"u_{th:.2f}_{dl}"
            if k not in d.files:
                continue
            cache[(th, dl)] = (
                element_strains(d["xy"], d[k], FD.NX, FD.NY),
                float(d[f"lam_{th:.2f}_{dl}"][0]))
    return cache


def station(base, area, cache, thetas, delta, cases):
    """Recovered section loss for every perturbation at one load level."""
    rows = {}
    for name, pert in cases:
        prob = perturbed(base, pert)
        row = []
        for th in thetas:
            item = cache.get((th, delta))
            if item is None:
                row.append(np.nan)
                continue
            (cx, cy, ex, ey, gxy), lam = item
            row.append(recover_with(prob, cx, cy, ex, ey, gxy, area, lam))
        rows[name] = np.array(row, float)
    return rows


def shift_pp(row, ref):
    """Mean absolute shift in percentage points, its sign, and its support.

    A state that admits no root on either side contributes nothing, so the
    average is over the states the two models agree are admissible. That
    count is carried alongside because it is not the same at every load
    level: the shortfall is largest at service, and there the intact state
    falls off the admissible range for most models.
    """
    dif = np.asarray(row, float) - np.asarray(ref, float)
    n = int(np.isfinite(dif).sum())
    if n == 0:
        return np.nan, np.nan, 0
    return (float(np.nanmean(np.abs(dif)) * 100),
            float(np.nanmean(dif) * 100), n)


def family_shift(shifts, keys):
    """Largest absolute shift over the directions of one material constant."""
    vals = [shifts[k][0] for k in keys if np.isfinite(shifts[k][0])]
    return max(vals) if vals else np.nan


def _fmt(v, w=9, p=3):
    return f"{('none' if not np.isfinite(v) else f'{v:.{p}f}'):>{w}}"


def print_station_table(thetas, rows, delta):
    """The table the single-station version printed, unchanged in shape."""
    print(f"section loss recovered with a perturbed identification model"
          f"   (delta = {delta} mm)\n")
    print(f"{'model':>12}" + "".join(f"{f'th={t:.2f}':>9}" for t in thetas)
          + f"{'mean shift':>12}")
    ref = rows[REF_CASE]
    for name, row in rows.items():
        s = 0.0 if name == REF_CASE else shift_pp(row, ref)[0]
        print(f"{name:>12}" + "".join(_fmt(v) for v in row)
              + f"{s:>11.1f} pp")


def main(deltas=DELTAS) -> None:
    deltas = [deltas] if isinstance(deltas, str) else list(deltas)
    d = np.load(FIELDS)
    base = DeepBeam()
    area = (base.L / FD.NX) * (base.H / FD.NY) / 2.0
    thetas = [float(t) for t in d["theta_true"]]
    cache = load_strains(d, thetas, deltas)

    # ---- 1. recovery and mechanism, station by station ---------------
    rec, shifts, yfrac, peak = {}, {}, {}, {}
    for dl in deltas:
        rows = station(base, area, cache, thetas, dl, CASES + AUX)
        rec[dl] = rows
        ref = rows[REF_CASE]
        shifts[dl] = {k: (0.0, 0.0, int(np.isfinite(ref).sum()))
                      if k == REF_CASE else shift_pp(v, ref)
                      for k, v in rows.items()}
        # the mechanism, read at the true parameter of each state
        yb, yc, pk, kn = [], [], [], []
        for th in thetas:
            item = cache.get((th, dl))
            if item is None:
                continue
            (cx, cy, ex, _, _), _ = item
            a, b, c = yield_fractions(base, ex, cx, cy)
            yb.append(a)
            yc.append(b)
            kn.append(c)
            pk.append(ex[cy < FD.BAND].max() / (base.mat.fy / base.mat.Es))
        yfrac[dl] = (float(np.mean(yb)), float(np.mean(yc)),
                     float(np.mean(kn)))
        peak[dl] = float(np.max(pk))

    if len(deltas) == 1:
        print_station_table(thetas, {k: rec[deltas[0]][k] for k, _ in CASES},
                            deltas[0])
        return

    # ---- 2. the per-station tables -----------------------------------
    for dl in deltas:
        print_station_table(thetas, {k: rec[dl][k] for k, _ in CASES}, dl)
        print()

    # ---- 3. the table the budget needs -------------------------------
    print("=" * 78)
    print("mean shift in recovered theta (pp) against load level\n")
    head = "".join(f"{f'{x} mm':>10}" for x in deltas)
    print(f"{'perturbation':>14}{head}")
    for name, _ in CASES:
        if name == REF_CASE:
            continue
        cells = "".join(
            f"{shifts[dl][name][0]:>10.2f}" if np.isfinite(shifts[dl][name][0])
            else f"{'none':>10}" for dl in deltas)
        print(f"{name:>14}{cells}")
    print(f"{'-' * 14}{'-' * (10 * len(deltas))}")
    print(f"{'band past yield':>14}"
          + "".join(f"{100 * yfrac[dl][0]:>9.0f}%" for dl in deltas))
    print(f"{'cut past yield':>14}"
          + "".join(f"{100 * yfrac[dl][1]:>9.0f}%" for dl in deltas))
    cells = ""
    for dl in deltas:
        ns = [shifts[dl][k][2] for k, _ in CASES if k != REF_CASE]
        cells += (f"{min(ns):>10d}" if min(ns) == max(ns)
                  else f"{f'{min(ns)}-{max(ns)}':>10}")
    print(f"{'states used':>14}{cells}")
    print(f"{'peak band eps':>14}"
          + "".join(f"{peak[dl]:>9.2f}y" for dl in deltas))
    print(f"{'cut at the knee':>14}"
          + "".join(f"{100 * yfrac[dl][2]:>9.0f}%" for dl in deltas))

    # ---- 4. does the reversal happen? --------------------------------
    fam = {dl: {f: family_shift(shifts[dl], k) for f, k in FAMILY.items()}
           for dl in deltas}
    lo, hi = deltas[0], deltas[-1]
    reversed_lo = fam[lo]["E_s"] > fam[lo]["f_y"]
    reversed_hi = all(fam[dl]["f_y"] > fam[dl]["E_s"]
                      for dl in deltas if float(dl) >= 5.0)
    ok = bool(reversed_lo and reversed_hi)
    print("\n" + "=" * 78)
    print("reversal test: modulus governs the elastic band, yield strength "
          "the yielded one")
    for dl in deltas:
        print(f"  delta {dl:>4} mm   cut {100 * yfrac[dl][1]:3.0f} % yielded"
              f"   f_y {fam[dl]['f_y']:5.2f} pp   E_s {fam[dl]['E_s']:5.2f} pp"
              f"   f_c {fam[dl]['f_c']:5.2f} pp"
              f"   governs: {'E_s' if fam[dl]['E_s'] > fam[dl]['f_y'] else 'f_y'}")
    print(f"\n  reversal confirmed: {ok}")

    # ---- 5. does the shift scale with the size of the error? ---------
    print("\nshift per one per cent of material error (pp), from +5 % and +2 %")
    print(f"{'constant':>14}{head}")
    for big, small in (("f_y  +5 %", "f_y  +2 %"), ("E_s  +5 %", "E_s  +2 %")):
        cells = ""
        for dl in deltas:
            a, b = shifts[dl][big][0] / 5.0, shifts[dl][small][0] / 2.0
            cells += (f"{0.5 * (a + b):>10.2f}" if np.isfinite(a + b)
                      else f"{'none':>10}")
        print(f"{big.split()[0]:>14}{cells}")

    # ---- 6. what the budget should quote at service ------------------
    svc = deltas[0]
    print("\n" + "=" * 78)
    print(f"at the service station (delta = {svc} mm) the material term of the "
          "budget is")
    print(f"  steel modulus known to 5 %      {fam[svc]['E_s']:5.2f} pp")
    print(f"  yield strength known to 5 %     {fam[svc]['f_y']:5.2f} pp")
    print(f"  concrete strength known to 10 % {fam[svc]['f_c']:5.2f} pp")
    print(f"against {fam[DELTA]['f_y']:.2f} pp for the yield strength and "
          f"{fam[DELTA]['E_s']:.2f} pp for the modulus at {DELTA} mm.")

    # ---- 7. the archive ----------------------------------------------
    def j(a):
        return [None if not np.isfinite(v) else round(float(v), 6) for v in a]

    doc = {
        "what": "material-model error absorbed into the recovered section "
                "loss, by load level",
        "arm_mm": 370.0, "x_cut_mm": FD.X_CUT, "band_mm": FD.BAND,
        "theta_true": thetas, "deltas_mm": [float(x) for x in deltas],
        "eps_yield": base.mat.fy / base.mat.Es,
        "recovered": {dl: {k: j(v) for k, v in rec[dl].items()}
                      for dl in deltas},
        "mean_abs_shift_pp": {k: {dl: (None if not np.isfinite(shifts[dl][k][0])
                                       else round(shifts[dl][k][0], 3))
                                  for dl in deltas}
                              for k, _ in CASES + AUX if k != REF_CASE},
        "signed_mean_shift_pp": {k: {dl: (None if not np.isfinite(shifts[dl][k][1])
                                          else round(shifts[dl][k][1], 3))
                                     for dl in deltas}
                                 for k, _ in CASES + AUX if k != REF_CASE},
        "states_used": {k: {dl: shifts[dl][k][2] for dl in deltas}
                        for k, _ in CASES + AUX if k != REF_CASE},
        "family_shift_pp": {dl: {f: (None if not np.isfinite(v) else round(v, 3))
                                 for f, v in fam[dl].items()} for dl in deltas},
        "yield_fraction_band": {dl: round(yfrac[dl][0], 4) for dl in deltas},
        "yield_fraction_cut": {dl: round(yfrac[dl][1], 4) for dl in deltas},
        "peak_band_strain_over_yield": {dl: round(peak[dl], 3) for dl in deltas},
        "cut_within_5pc_of_yield": {dl: round(yfrac[dl][2], 4) for dl in deltas},
        "reversal_confirmed": ok,
        "governs": {dl: ("E_s" if fam[dl]["E_s"] > fam[dl]["f_y"] else "f_y")
                    for dl in deltas},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DELTAS)
