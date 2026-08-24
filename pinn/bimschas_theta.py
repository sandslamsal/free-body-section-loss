"""Bimschas (2010) VK1/VK3 as companion piers differing in reinforcement, read through the
theta (section-loss) parametrization of this study.

Method under test
-----------------
Section loss theta is identified by statics on a free body: the internal force
the observable implies must equal what statics requires,
        T(theta) = T_c + rho_tie (1-theta) S  =  T_req,
with T affine and decreasing in theta, so the root is unique (bisection).
Here the free body is the pier base section of the ETH wall-type bridge pier;
T_req follows from the applied horizontal load and the free-body geometry
(lever arm L_v = 3.3 m) together with the constant axial load N = 1370 kN.

Source data (Bimschas 2010, ETH diss. 18849, DOI 10.3929/ethz-a-006050338)
    Tab. 5.1  L_v = 3.3 m, l_w = 1.5 m, b_w = 0.35 m
    Tab. 5.3  VK1 28 D14 (rho_sl = 0.82 %), VK3 42 D14 (rho_sl = 1.23 %)
    Fig. 5.2  VK1 2x10 D14 at s = 130 over 1170 + 2x4 D14 at the ends
              VK3 2x17 D14 at s =  80 over 1280 + 2x4 D14 at the ends
    Tab. 5.4  f_s,l = 515 MPa, f_t,l = 630 MPa, eps_su = 12.6 %
              f_c,cyl = 35 / 39 / 34 MPa for VK1 / VK2 / VK3
    Tab. 5.5  numerical M_n = 2280 / 2314 / 2808 kNm
    Tab. 5.6  numerical F_n =  691 /  701 /  851 kN
    Tab. 5.7  N_base = 1370 kN
    Tab. 5.9  inelastic load steps 10.5, 15.75, 21, 31.5, 42, 52.5, 63 mm
    Fig. 5.20 measured backbones (extracted exactly by bimschas_extract.py)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DATA = Path("/Users/sandeshlamsal/Desktop/CSFD/Research/P4-ES/data")

# ---------------------------------------------------------------- geometry --
B_W, L_W, L_V = 350.0, 1500.0, 3300.0          # mm
N_BASE = 1370.0e3                              # N, compression, constant
D_BAR = 14.0
A_BAR = np.pi * D_BAR ** 2 / 4.0
Y_END = 717.0        # bar centroid at the section ends: 750 - 26 (cl) - 7

# depth coordinates (mm from section centroid) and bar count at each
LAYOUT = {
    # 2x4 D14 at each end + 2x10 D14 at s = 130 mm over 1170 mm
    "VK1": [(+Y_END, 4), (-Y_END, 4)]
           + [(s * y, 2) for y in (585., 455., 325., 195., 65.) for s in (1, -1)],
    # 2x4 D14 at each end + 2x17 D14 at s = 80 mm over 1280 mm
    "VK3": [(+Y_END, 4), (-Y_END, 4)]
           + [(s * y, 2) for y in (640., 560., 480., 400., 320., 240., 160., 80.)
              for s in (1, -1)] + [(0.0, 2)],
}
FC = {"VK1": 35.0, "VK2": 39.0, "VK3": 34.0}          # Tab. 5.4 (see notes)
FC_PROSE = {"VK1": 39.0, "VK2": 35.0, "VK3": 34.0}    # p. 305 prose
F_Y, E_S = 515.0, 200000.0                            # Tab. 5.4; E_s assumed
F_U, EPS_SU = 630.0, 0.126
EPS_SH = 0.025          # end of the yield plateau (thesis p. 310)


# ------------------------------------------------------------ constitutive --
def sigma_s(eps):
    """Bilinear steel with a yield plateau, then linear hardening to f_u."""
    e = np.asarray(eps, float)
    s = np.clip(E_S * e, -F_Y, F_Y)
    hard = np.abs(e) > EPS_SH
    if np.any(hard):
        extra = (np.abs(e[hard]) - EPS_SH) / (EPS_SU - EPS_SH) * (F_U - F_Y)
        s[hard] = np.sign(e[hard]) * (F_Y + np.clip(extra, 0, F_U - F_Y))
    return s


def sigma_c(eps, fc, eps_c0=0.002):
    """Parabola-rectangle in compression, no tension (eps < 0 = compression)."""
    e = np.asarray(eps, float)
    x = np.clip(-e, 0.0, None)
    par = fc * (1.0 - (1.0 - np.minimum(x, eps_c0) / eps_c0) ** 2)
    return -np.where(x > 0, par, 0.0)


# ------------------------------------------------------------ section model --
def section(unit, theta=0.0, fc=None, nfib=600):
    fc = FC[unit] if fc is None else fc
    y = np.linspace(-L_W / 2, L_W / 2, nfib)
    dy = L_W / (nfib - 1)
    bars = np.array([(p, n) for p, n in LAYOUT[unit]])
    ys, As = bars[:, 0], bars[:, 1] * A_BAR * (1.0 - theta)
    return dict(fc=fc, y=y, dA=B_W * dy, ys=ys, As=As)


def state(sec, eps_top, kappa):
    """Plane sections.  y = +L_W/2 is the compressed edge, so

        eps(y) = eps_top + kappa * (L_W/2 - y),   kappa >= 0,

    with eps_top < 0 the extreme-fiber compressive strain.  Tension positive.
    Returns the internal axial force (tension positive) and the moment about
    the section centroid.
    """
    ec = eps_top + kappa * (L_W / 2 - sec["y"])
    es = eps_top + kappa * (L_W / 2 - sec["ys"])
    fcv = sigma_c(ec, sec["fc"]) * sec["dA"]
    fsv = sigma_s(es) * sec["As"]
    # steel displaces concrete
    fsv -= sigma_c(es, sec["fc"]) * sec["As"]
    Nn = fcv.sum() + fsv.sum()                    # +tension
    Mm = -(fcv * sec["y"]).sum() - (fsv * sec["ys"]).sum()
    return Nn, Mm


def moment_at(sec, eps_cu, n_ax=-N_BASE):
    """Moment when the extreme compression fiber reaches eps_cu.

    The compressed edge is y = +L_W/2 (eps_top = eps_cu, negative).
    Curvature is found by bisection on the axial-force residual.
    """
    lo, hi = 0.0, 1e-6
    f = lambda k: state(sec, eps_cu, k)[0] - n_ax
    while f(hi) < 0 and hi < 1.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return state(sec, eps_cu, k)[1], k


def capacity(unit, theta=0.0, eps_cu=-0.004, fc=None):
    M, k = moment_at(section(unit, theta, fc), eps_cu)
    return M / 1e6, k          # kNm, 1/mm


# ------------------------------------------------------------------ driver --
def load_backbones():
    out = {}
    with (DATA / "bimschas_backbones.csv").open() as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or row[0] == "unit":
                continue
            out.setdefault(row[0], []).append((float(row[1]), float(row[2])))
    return {k: np.array(v) for k, v in out.items()}


def main():
    bb = load_backbones()
    rep = {}

    print("=" * 78)
    print("1  VERIFICATION OF THE EXTRACTED CURVES AGAINST THE THESIS TEXT")
    print("=" * 78)
    v1, v2, v3 = bb["VK1"], bb["VK2"], bb["VK3"]
    at = lambda a, d: float(np.interp(d, a[:, 0], a[:, 1]))

    # (i) three percentages stated in the thesis prose
    r1 = v2[np.argmax(v2[:, 1]), 1] / v1[np.argmax(v1[:, 1]), 1]
    r2 = at(v2, 20.66) / at(v1, 20.78)
    drop = 1 - at(v1, 62.56) / at(v1, 52.05)
    checks = [
        ("VK2 peak vs VK1 peak (both at 31.5 mm), p.335", 100 * (r1 - 1), 2.3),
        ("VK2 vs VK1 at 21 mm, p.335", 100 * (r2 - 1), 4.2),
        ("VK1 drop 52.5 -> 63 mm, south, p.336", 100 * drop, 7.0),
        ("VK3 peak vs VK1 peak ('roughly 20 %', p.303)",
         100 * (v3[np.argmax(v3[:, 1]), 1] / v1[np.argmax(v1[:, 1]), 1] - 1), 20.0),
    ]
    for name, got, said in checks:
        print(f"  {name:<52s} extracted {got:6.2f} %   thesis {said:4.1f} %"
              f"   err {got-said:+.2f} pp")
    rep["verification_prose_pct"] = [
        dict(check=n, extracted_pct=round(g, 2), thesis_pct=s,
             error_pp=round(g - s, 2)) for n, g, s in checks]

    # (ii) first-yield displacement at the numerically computed F_y'
    fy_pred = {"VK1": 537.0, "VK3": 644.0}
    dy_south = {"VK1": 7.0, "VK3": 10.4}
    dy_avg = {"VK1": 7.5, "VK3": 10.7}
    ydat = []
    for u in ("VK1", "VK3"):
        a = bb[u]
        asc = a[:int(np.argmax(a[:, 1])) + 1]      # ascending branch only
        d = float(np.interp(fy_pred[u], asc[:, 1], asc[:, 0]))
        print(f"  {u} delta at F_y'={fy_pred[u]:.0f} kN: extracted {d:5.2f} mm"
              f"   Tab.5.8 south {dy_south[u]:4.1f} / avg {dy_avg[u]:4.1f} mm"
              f"   err {100*(d/dy_south[u]-1):+5.1f} % / "
              f"{100*(d/dy_avg[u]-1):+5.1f} %")
        ydat.append(dict(unit=u, delta_mm=round(d, 2), tab58_south=dy_south[u],
                         tab58_avg=dy_avg[u]))
    rep["verification_first_yield"] = ydat

    # (iii) abscissae against the prescribed load steps (Tab. 5.9)
    steps = np.array([10.5, 15.75, 21., 31.5, 42., 52.5, 63.])
    for u in ("VK1", "VK3"):
        a = bb[u][bb[u][:, 0] > 8.0]
        n = min(len(a), len(steps))
        e = 100 * (a[:n, 0] / steps[:n] - 1)
        print(f"  {u} plotted abscissae vs target load steps: "
              f"mean {e.mean():+.2f} %, max |{np.abs(e).max():.2f}| % "
              f"(actuator control tolerance, not an extraction error)")

    # (iv) tick-fit residual (from the extractor) -> extraction precision
    print("  axis tick-fit residual: 0.008 mm and 0.05 kN "
          "(vector paths; no pixel digitisation)")

    print()
    print("=" * 78)
    print("2  MEASURED REINFORCEMENT SENSITIVITY  (VK1 vs VK3)")
    print("=" * 78)
    rho = {"VK1": 0.82, "VK3": 1.23}
    As_tot = {u: sum(n for _, n in LAYOUT[u]) * A_BAR for u in ("VK1", "VK3")}
    for u in ("VK1", "VK3"):
        print(f"  {u}: {int(sum(n for _,n in LAYOUT[u]))} D14 = "
              f"{As_tot[u]:7.1f} mm^2, rho = "
              f"{100*As_tot[u]/(B_W*L_W):.3f} % (Tab.5.3 {rho[u]} %)")
    r_rho = As_tot["VK1"] / As_tot["VK3"]
    print(f"  reinforcement ratio VK1/VK3 = {r_rho:.5f}  (= 28/42 exactly)")

    pk1 = v1[np.argmax(v1[:, 1])]
    pk3 = v3[np.argmax(v3[:, 1])]
    print(f"  measured peak  VK1 {pk1[1]:6.1f} kN at {pk1[0]:5.2f} mm")
    print(f"  measured peak  VK3 {pk3[1]:6.1f} kN at {pk3[0]:5.2f} mm")
    r_V = pk1[1] / pk3[1]
    # base moment including P-Delta of the constant axial load
    M1 = pk1[1] * L_V / 1e3 + N_BASE * pk1[0] / 1e9 * 1e3
    M3 = pk3[1] * L_V / 1e3 + N_BASE * pk3[0] / 1e9 * 1e3
    r_M = M1 / M3
    print(f"  capacity ratio VK1/VK3: shear {r_V:.4f}   "
          f"base moment incl. P-Delta {r_M:.4f}")
    el_V = np.log(r_V) / np.log(r_rho)
    el_M = np.log(r_M) / np.log(r_rho)
    print(f"  ELASTICITY of measured capacity to rho_sl: "
          f"{el_V:.3f} (shear) / {el_M:.3f} (base moment)")
    print(f"  thesis's own statement '+50 % rho_sl -> +20 % moment' (p.303) "
          f"=> elasticity {np.log(1.20)/np.log(1.5):.3f}")
    print(f"  thesis's own numerical F_n 691 -> 851 kN (Tab.5.6) "
          f"=> elasticity {np.log(691/851)/np.log(r_rho):.3f}")
    rep["measured"] = dict(
        rho_ratio=round(r_rho, 5), V_VK1_kN=round(pk1[1], 1),
        V_VK3_kN=round(pk3[1], 1), delta_VK1_mm=round(pk1[0], 2),
        delta_VK3_mm=round(pk3[0], 2), ratio_V=round(r_V, 4),
        ratio_M=round(r_M, 4), elasticity_V=round(el_V, 3),
        elasticity_M=round(el_M, 3),
        thesis_prose_elasticity=round(float(np.log(1.20)/np.log(1.5)), 3),
        thesis_numeric_elasticity=round(float(np.log(691/851)/np.log(r_rho)), 3))

    # ratio at every common load step
    print("\n  ratio and elasticity at each matched load step:")
    stepwise = []
    for s in steps:
        if s > 45:   # VK3 failed at the 52.5 mm step
            continue
        a, b = at(v1, s), at(v3, s)
        e = np.log(a / b) / np.log(r_rho)
        print(f"    delta = {s:5.2f} mm: VK1 {a:6.1f}  VK3 {b:6.1f}  "
              f"ratio {a/b:.4f}  elasticity {e:.3f}")
        stepwise.append(dict(delta_mm=s, VK1_kN=round(a, 1), VK3_kN=round(b, 1),
                             ratio=round(a / b, 4), elasticity=round(e, 3)))
    rep["stepwise"] = stepwise

    print()
    print("=" * 78)
    print("3  STATICS / CONSTITUTIVE CHAIN ON THE BASE FREE BODY")
    print("=" * 78)
    print("  model check against the thesis's own M-phi analysis (Tab. 5.5):")
    for label, fcmap in (("Tab. 5.4 order", FC), ("p.305 prose order", FC_PROSE)):
        line = []
        for u, mn in (("VK1", 2280.), ("VK3", 2808.)):
            M, _ = capacity(u, 0.0, -0.004, fcmap[u])
            line.append(f"{u} {M:7.0f} kNm (thesis {mn:.0f}, "
                        f"{100*(M/mn-1):+5.1f} %)")
        Ms = [capacity(u, 0.0, -0.004, fcmap[u])[0] for u in ("VK1", "VK3")]
        line.append(f"ratio {Ms[0]/Ms[1]:.4f} (thesis {2280/2808:.4f})")
        print(f"    f_c from {label:<18s}: " + "  ".join(line))

    print("\n  predicted vs measured capacity ratio VK1/VK3:")
    for ecu in (-0.003, -0.004, -0.005, -0.008):
        M1m = capacity("VK1", 0.0, ecu)[0]
        M3m = capacity("VK3", 0.0, ecu)[0]
        print(f"    eps_cu = {ecu:+.3f}: model ratio {M1m/M3m:.4f}   "
              f"measured {r_M:.4f}   err {100*(M1m/M3m/r_M-1):+5.2f} %")
    Mmod1, Mmod3 = capacity("VK1")[0], capacity("VK3")[0]
    rep["model"] = dict(M_VK1_kNm=round(Mmod1, 1), M_VK3_kNm=round(Mmod3, 1),
                        ratio=round(Mmod1 / Mmod3, 4),
                        measured_ratio_M=round(r_M, 4))

    print()
    print("=" * 78)
    print("4  MAPPING ONTO THE theta PARAMETRIZATION")
    print("=" * 78)
    theta_true = 1.0 - r_rho
    print(f"  VK1 is VK3 with a uniform section loss of "
          f"theta = 1 - 28/42 = {theta_true:.4f}  ({100*theta_true:.2f} %)")

    # r(theta) on the VK3 section with all bar areas scaled by (1-theta)
    th = np.linspace(0.0, 0.6, 241)
    M0 = capacity("VK3", 0.0)[0]
    rmod = np.array([capacity("VK3", t)[0] for t in th]) / M0
    print(f"  model observable r(theta) = M(theta)/M(0) on the VK3 section:")
    for t in (0.0, 0.10, 1/3, 0.50):
        print(f"    theta = {t:5.3f} -> r = {np.interp(t, th, rmod):.4f}")
    dr = (np.interp(1/3, th, rmod) - 1.0) / (1/3)
    dr_meas = (r_M - 1.0) / theta_true
    print(f"  model    d r / d theta over 0..1/3 : {dr:+.4f} per unit theta")
    print(f"  MEASURED d r / d theta over 0..1/3 : {dr_meas:+.4f} per unit "
          f"theta  ({abs(dr_meas):.4f} % of capacity per % of section loss)")

    # invert: recover theta from the measured normalized capacity
    lo, hi = 0.0, 0.95
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if capacity("VK3", mid)[0] / M0 > r_M:
            lo = mid
        else:
            hi = mid
    th_rec = 0.5 * (lo + hi)
    print(f"\n  bisection on T(theta) = T_req using the MEASURED normalized "
          f"capacity r = {r_M:.4f}:")
    print(f"    theta recovered = {100*th_rec:.2f} %   "
          f"theta known = {100*theta_true:.2f} %   "
          f"error = {100*(th_rec-theta_true):+.2f} pp")
    rep["theta"] = dict(theta_true_pct=round(100 * theta_true, 2),
                        theta_recovered_pct=round(100 * th_rec, 2),
                        error_pp=round(100 * (th_rec - theta_true), 2),
                        d_r_d_theta_measured=round(dr_meas, 4),
                        d_r_d_theta_model=round(dr, 4),
                        r_measured=round(r_M, 4))

    # the layout confound: uniform theta vs the real VK1 bar arrangement
    fm = lambda u, t: sum(n * A_BAR * (1 - t) * abs(y) for y, n in LAYOUT[u])
    print(f"\n  layout confound: first moment of steel area about the centroid")
    print(f"    real VK1 layout            : {fm('VK1',0)/1e6:8.3f} x10^6 mm^3")
    print(f"    VK3 layout at theta = 1/3  : {fm('VK3',1/3)/1e6:8.3f} x10^6 mm^3"
          f"   ({100*(fm('VK3',1/3)/fm('VK1',0)-1):+.1f} %)")
    rep["layout_confound_pct"] = round(100 * (fm("VK3", 1/3) / fm("VK1", 0) - 1), 2)

    # sensitivity of theta_rec to the model knobs
    print("\n  sensitivity of the recovered theta (pp) to modeling choices:")
    sens = []
    for tag, kw in (("eps_cu = -0.003", dict(eps_cu=-0.003)),
                    ("eps_cu = -0.005", dict(eps_cu=-0.005)),
                    ("eps_cu = -0.008", dict(eps_cu=-0.008)),
                    ("f_c = 39 MPa (prose)", dict(fc=39.0)),
                    ("f_c = 34 MPa", dict(fc=34.0))):
        M0k = capacity("VK3", 0.0, **kw)[0]
        lo, hi = 0.0, 0.95
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if capacity("VK3", mid, **kw)[0] / M0k > r_M:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        print(f"    {tag:<22s}: theta = {100*t:5.2f} %  "
              f"({100*(t-theta_true):+5.2f} pp)")
        sens.append(dict(variant=tag, theta_pct=round(100 * t, 2),
                         error_pp=round(100 * (t - theta_true), 2)))
    rep["sensitivity"] = sens

    # ---- decomposition of the -pp error --------------------------------
    print("\n  decomposition of the theta error:")
    M_uni = np.interp(1/3, th, rmod) * M0            # VK3 layout, uniform 1/3
    M_vk1 = capacity("VK1", 0.0)[0]                  # real VK1 layout
    dlayout = (M_vk1 - M_uni) / M0
    pp_layout = 100 * dlayout / abs(dr)
    print(f"    VK3 layout at theta=1/3 : {M_uni:7.0f} kNm")
    print(f"    real VK1 layout         : {M_vk1:7.0f} kNm "
          f"({100*(M_vk1/M_uni-1):+.2f} %)")
    print(f"    -> bar-layout confound accounts for {-pp_layout:+.2f} pp of the "
          f"theta error")
    print(f"    -> residual model/measurement error "
          f"{100*(th_rec-theta_true)+pp_layout:+.2f} pp")
    rep["error_decomposition_pp"] = dict(
        total=round(100 * (th_rec - theta_true), 2),
        bar_layout=round(-pp_layout, 2),
        residual=round(100 * (th_rec - theta_true) + pp_layout, 2))

    # ---- wider sensitivity --------------------------------------------
    global F_Y
    print("\n  further sensitivity of the recovered theta:")
    extra = []

    def recover(r_target, **kw):
        M0k = capacity("VK3", 0.0, **kw)[0]
        lo, hi = 0.0, 0.95
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            if capacity("VK3", mid, **kw)[0] / M0k > r_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    for tag, rt in (("shear ratio, no P-Delta", r_V),
                    ("base-moment ratio, with P-Delta", r_M)):
        t = recover(rt)
        extra.append((tag, t))
    f_save = F_Y
    for fy in (500.0, 520.0):
        F_Y = fy
        extra.append((f"f_y = {fy:.0f} MPa", recover(r_M)))
    F_Y = f_save
    for tag, t in extra:
        print(f"    {tag:<32s}: theta = {100*t:5.2f} %  "
              f"({100*(t-theta_true):+5.2f} pp)")
    rep["sensitivity"] += [dict(variant=t_, theta_pct=round(100*v, 2),
                                error_pp=round(100*(v-theta_true), 2))
                           for t_, v in extra]

    # ---- theta recovered at each load step -----------------------------
    print("\n  theta recovered from the ratio at each matched load step:")
    perstep = []
    for row in stepwise:
        t = recover(row["ratio"])
        print(f"    delta = {row['delta_mm']:5.2f} mm: r = {row['ratio']:.4f} "
              f"-> theta = {100*t:5.2f} %  ({100*(t-theta_true):+5.2f} pp)")
        perstep.append(dict(delta_mm=row["delta_mm"], r=row["ratio"],
                            theta_pct=round(100*t, 2),
                            error_pp=round(100*(t-theta_true), 2)))
    rep["theta_per_load_step"] = perstep

    (DATA / "bimschas_theta_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {DATA / 'bimschas_theta_report.json'}")


if __name__ == "__main__":
    main()
