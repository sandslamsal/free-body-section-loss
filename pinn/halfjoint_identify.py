"""Section-loss recovery on the Cambridge half-joint open data.

Desnerck, Lees & Morley, Eng. Struct. 127 (2016), 152 (2017), 161 (2018);
open data CC BY 4.0.  A half-joint is a dapped end, the D-region archetype,
and the 2016 open data reports the MEASURED bar forces at the inner nib
against applied load, which is the quantity the study's identifying
condition computes.

Identifying condition, as in the study:

    T(theta) = T_c + (1 - theta) S   must equal   T_req,

  S     = the tie resultant the MEASURED strain implies at nominal section,
          taken from the authors' own assembled bar forces (2016 Fig. 16);
  T_c   = 0, the constitutive map neglects concrete tension;
  T_req = what statics on the free body requires;
  theta is admissible on [0, 0.70] (identify.THETA_MAX).

Free bodies:
  (V) vertical equilibrium of the nib bounded by the corner crack,
        R = VSt1 + VSt2 + VDiagn + V_concrete,  T_req = R.
      No lever arm, no crack angle in the force balance, no plane sections.
  (H) horizontal tie demand from the published strut-and-tie model,
        HStack_req = 0.916 R.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DATA = Path("/Users/sandeshlamsal/Desktop/CSFD/Research/P4-ES/data")
AZ = 0.916            # HStack_req / R from the published STM, halfjoint_geometry.py
THETA_MAX = 0.70      # identify.THETA_MAX
FLOOR = 0.019         # the study's stated false-positive floor

FAIL = {"NS-REF": 402.3, "NS-ND": 244.9, "NS-NU": 295.8, "NS-RS": 358.7,
        "NS-LR": 261.9}
FCRACK = {"NS-REF": 107.0, "NS-ND": 102.3, "NS-NU": 100.5, "NS-RS": 98.6}
TIE_GOVERNED = ("NS-REF", "NS-ND", "NS-NU")


def load():
    raw = (DATA / "desnerck_fig16_barforces.csv").read_text().strip().split("\n")
    cols = raw[0].split(",")[1:]
    out: dict[str, list] = {}
    for line in raw[1:]:
        p = line.split(",")
        out.setdefault(p[0], []).append(
            [np.nan if x == "" else float(x) for x in p[1:]])
    return {k: np.array(v) for k, v in out.items()}, cols


def bisect(S: float, T_req: float, lo: float, hi: float, tol=1e-12):
    """Root of (1 - theta) S - T_req on [lo, hi], or None if none lies there."""
    f = lambda t: (1.0 - t) * S - T_req
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def at_load(a, cols, Ft, hw=3.0):
    F = a[:, 0]
    m = (F >= Ft - hw) & (F <= Ft + hw)
    if m.sum() < 3:
        return None
    with np.errstate(all="ignore"):
        return {c: float(np.nanmedian(a[m, k])) for k, c in enumerate(cols)}


def main() -> None:
    d, cols = load()
    res: dict = {"free_body": "vertical equilibrium of the nib cut by the "
                              "corner crack", "az_horizontal": AZ}

    print("=" * 86)
    print("T3  POSITIVE CONTROL -- sound ties, truth theta = 0")
    print("=" * 86)
    print("theta_adm  : root inside the study's admissible range [0, 0.70]")
    print("theta_free : unconstrained root, i.e. the systematic bias\n")
    per_spec = {}
    for nm in ("NS-REF", "NS-ND", "NS-NU", "NS-RS"):
        a = d[nm]
        F0 = float(np.nanmin(a[:, 0]))
        Fu = FAIL[nm]
        tag = "tie-governed" if nm in TIE_GOVERNED else \
              "shear failure in the full-depth section -- NOT tie-governed"
        print(f"--- {nm}  F_ult {Fu} kN, F_crack {FCRACK[nm]} kN, "
              f"gauge datum {F0:.1f} kN   [{tag}]")
        print(f"{'F':>7}{'F/Fu':>6}{'R':>7}{'S_V':>8}{'Vc/R':>7}"
              f"{'thV_adm':>9}{'thV_free':>10}"
              f"{'S_H':>8}{'H_req':>8}{'thH_adm':>9}{'thH_free':>10}")
        rows = []
        for Ft in [FCRACK[nm], 125, 150, 175, 200, 225, 250, 275, 300, 325,
                   336.5, 350, 375, Fu]:
            if Ft > Fu + 1:
                continue
            v = at_load(a, cols, min(Ft, Fu))
            if v is None:
                continue
            R = Ft - F0
            Sv, Sh = v["VStack"], v["HStack"]
            tv_a = bisect(Sv, R, 0.0, THETA_MAX) if Sv > 1e-6 else None
            tv_f = 1 - R / Sv if Sv > 1e-6 else None
            th_a = bisect(Sh, AZ * R, 0.0, THETA_MAX) if Sh > 1e-6 else None
            th_f = 1 - AZ * R / Sh if Sh > 1e-6 else None
            g = lambda x: "     --" if x is None else f"{x:+7.3f}"
            print(f"{Ft:>7.1f}{Ft/Fu:>6.2f}{R:>7.1f}{Sv:>8.1f}"
                  f"{1-Sv/R:>7.2f}{g(tv_a):>9}{g(tv_f):>10}"
                  f"{Sh:>8.1f}{AZ*R:>8.1f}{g(th_a):>9}{g(th_f):>10}")
            rows.append(dict(F=Ft, R=R, S_V=Sv, S_H=Sh,
                             thV_adm=tv_a, thV_free=tv_f,
                             thH_adm=th_a, thH_free=th_f))
        per_spec[nm] = rows
        fv = [r["thV_free"] for r in rows if r["thV_free"] is not None]
        fh = [r["thH_free"] for r in rows if r["thH_free"] is not None]
        av = [r for r in rows if r["thV_adm"] is not None]
        ah = [r for r in rows if r["thH_adm"] is not None]
        print(f"   vertical  : admissible root at "
              f"{len(av)}/{len(rows)} load levels; best bias "
              f"{max(fv):+.3f} ({max(fv)*100:+.1f} pp) at "
              f"F/Fu = {rows[int(np.argmax(fv))]['F']/Fu:.2f}")
        print(f"   horizontal: admissible root at {len(ah)}/{len(rows)} "
              f"load levels; largest FALSE POSITIVE "
              f"{max(fh):+.3f} ({max(fh)*100:+.1f} pp)\n")
    res["T3"] = per_spec

    print("=" * 86)
    print("T3 SUMMARY against the study's 1.9 pp false-positive floor")
    print("=" * 86)
    for nm in ("NS-REF", "NS-ND", "NS-NU", "NS-RS"):
        rows = per_spec[nm]
        nadmV = sum(r["thV_adm"] is not None for r in rows)
        nadmH = sum(r["thH_adm"] is not None for r in rows)
        bestV = max(r["thV_free"] for r in rows if r["thV_free"] is not None)
        worstH = max(r["thH_free"] for r in rows if r["thH_free"] is not None)
        print(f"  {nm:8s} vertical: {nadmV} admissible roots out of {len(rows)}"
              f", best {bestV*100:+6.1f} pp "
              f"[{'within' if abs(bestV) <= FLOOR else 'outside'} 1.9 pp]"
              f" | horizontal: {nadmH} admissible, worst {worstH*100:+6.1f} pp")

    # ---- lever-arm sensitivity of the horizontal form ----------------------
    print("\nlever-arm sensitivity of the moment form (NS-REF at 300 kN):")
    a = d["NS-REF"]; F0 = float(np.nanmin(a[:, 0]))
    v = at_load(a, cols, 300.0); R = 300.0 - F0
    print(f"   d(theta)/d(a/z) = -R/S_H = {-R/v['HStack']:.3f} per unit a/z")
    print(f"   1.9 pp of section loss = {0.019*v['HStack']/R*100:.2f} % error "
          f"in the assumed a/z")
    az0 = v["HStack"] / R
    print(f"   a/z that would return theta = 0 here: {az0:.3f} "
          f"(STM value {AZ:.3f}, {100*(az0/AZ-1):+.0f} %)")
    res["az_sensitivity"] = dict(dtheta_daz=-R/v["HStack"], az_zeroing=az0)

    # ---- T2 -----------------------------------------------------------------
    print("\n" + "=" * 86)
    print("T2  CAPACITY PREDICTION FOR NS-LR FROM A KNOWN 50 % SECTION LOSS")
    print("=" * 86)
    ref = d["NS-REF"]; i = int(np.nanargmax(ref[:, 0]))
    pk = {c: float(ref[i, k]) for k, c in enumerate(cols)}
    F0r = float(np.nanmin(ref[:, 0])); Rr = FAIL["NS-REF"] - F0r
    A12, A10 = np.pi/4*144, np.pi/4*100
    cap = dict(ubar=3*A12*559/1e3, diag=4*A12*559/1e3, stirrup=2*A10*596/1e3)
    print("NS-REF at failure, measured (2016 Fig. 16) vs the source's own fu:")
    print(f"   U-bars    {pk['Ubar']:6.1f} kN / {cap['ubar']:6.1f} kN "
          f"= {pk['Ubar']/cap['ubar']*100:5.1f} %")
    print(f"   diagonals {pk['Diagn']:6.1f} kN / {cap['diag']:6.1f} kN "
          f"= {pk['Diagn']/cap['diag']*100:5.1f} %")
    print(f"   stirrup 1 {pk['VSt1']:6.1f} kN / {cap['stirrup']:6.1f} kN "
          f"= {pk['VSt1']/cap['stirrup']*100:5.1f} %")
    print(f"   stirrup 2 {pk['VSt2']:6.1f} kN / {cap['stirrup']:6.1f} kN "
          f"= {pk['VSt2']/cap['stirrup']*100:5.1f} %")
    print("   -> the nib tie of NS-REF is at its rupture capacity, matching "
          "the reported failure mode")
    V_ref, V_lr = pk["VStack"], 0.5*(pk["VDiagn"] + pk["VSt1"]) + pk["VSt2"]
    Vc = Rr - V_ref
    th_eff = 1 - V_lr/V_ref
    meas, mref = FAIL["NS-LR"], FAIL["NS-REF"]
    pa = mref * V_lr / V_ref
    pb = V_lr + Vc + F0r
    ph = mref * 0.5
    print(f"\nvertical free body at NS-REF failure: R {Rr:.1f}, "
          f"S_V {V_ref:.1f}, residual V_c {Vc:.1f} kN ({Vc/Rr*100:.0f} % of R)")
    print(f"NS-LR: diagonals, U-bars and stirrup 1 milled to 50 % area over a "
          f"100 mm zone; stirrup 2 intact")
    print(f"   vertical tie after milling {V_lr:.1f} kN, "
          f"equivalent uniform loss theta_eff = {th_eff:.3f}")
    print(f"   (a) tie and concrete share scale together : {pa:6.1f} kN "
          f"vs {meas} ({(pa/meas-1)*100:+5.1f} %)")
    print(f"   (b) concrete share held at {Vc:.1f} kN        : {pb:6.1f} kN "
          f"vs {meas} ({(pb/meas-1)*100:+5.1f} %)")
    print(f"   (c) horizontal free body, both ties halved: {ph:6.1f} kN "
          f"vs {meas} ({(ph/meas-1)*100:+5.1f} %)")
    print(f"\n   measured capacity ratio {meas}/{mref} = {meas/mref:.3f}")
    print(f"   effective tie loss implied by measured capacity {1-meas/mref:.3f}")
    print(f"   effective tie loss predicted by the free body   {th_eff:.3f}")
    print(f"   error {100*(th_eff-(1-meas/mref)):+.1f} pp of section loss, "
          f"{(pa/meas-1)*100:+.1f} % of capacity")
    res["T2"] = dict(V_ref=V_ref, V_lr=V_lr, V_c=Vc, theta_eff=th_eff,
                     theta_from_capacity=1-meas/mref,
                     pred_a=pa, pred_b=pb, pred_h=ph, measured=meas,
                     err_pp=100*(th_eff-(1-meas/mref)),
                     err_pct_capacity=(pa/meas-1)*100)
    v_nd = pk["VSt1"] + pk["VSt2"]
    print(f"\n   cross-check on the removal specimens (same free body):")
    print(f"     NS-ND, diagonal removed: {mref*v_nd/V_ref:6.1f} kN vs "
          f"{FAIL['NS-ND']} ({(mref*v_nd/V_ref/FAIL['NS-ND']-1)*100:+.0f} %)")
    print(f"     NS-NU, U-bars removed  : {mref:6.1f} kN vs {FAIL['NS-NU']} "
          f"({(mref/FAIL['NS-NU']-1)*100:+.0f} %) "
          f"-- the vertical free body is blind to the horizontal tie")

    (DATA / "halfjoint_results.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {DATA/'halfjoint_results.json'}")


if __name__ == "__main__":
    main()
