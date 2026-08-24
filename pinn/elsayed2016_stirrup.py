"""
Free-body identification of reinforcement section loss on the corroded short-beam
tests of El-Sayed, Hussain & Shuraim (2016), J. Civ. Eng. Manag. 22(4): 491-499,
doi:10.3846/13923730.2014.897990 (CC BY 4.0).

Identifying condition under test (the study's own):

    T(theta) = T_c + A_s (1-theta) sigma_s(eps_measured)   must equal   T_req(statics)

with T_c ~ 0 (constitutive map neglects concrete tension), T affine and decreasing
in theta, so the root is found by bisection, not optimization.

Three analyses are run:

  T1-LONG   identification on the LONGITUDINAL tie at midspan.  Measured strain is
            available for 13 of 14 beams (Table 3) and the ground truth is
            theta = 0 by construction: "The longitudinal steel bars were epoxy
            coated to preclude corrosion of these elements" (Sec. 2.3).
            Degenerate ground truth -> measures BIAS, not discrimination.

  T1-STIR   identification on the STIRRUPS from a free body cut by the inclined
            crack.  Measured stirrup strain exists ONLY for the five uncorroded
            control beams ("No strain gauges were attached to the stirrups to be
            corroded as they would be destroyed during the accelerated corrosion
            phase", Sec. 2.4), so again theta = 0 is the only available truth.

  T2        constitutive test: does the weighed ASTM G1-03 mass loss (Table 2)
            predict the measured ultimate load and reported strength reduction
            (Table 3)?  Run forward (theta -> capacity) and inverse
            (capacity -> theta).

Run with /usr/local/bin/python3.12
"""
import csv, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
CSV  = os.path.join(DATA, "elsayed2016_tables.csv")

# ---------------------------------------------------------------- source data
def load():
    rows = []
    with open(CSV) as f:
        for line in f:
            if line.startswith("#"):
                continue
            rows.append(line)
    rd = csv.DictReader(rows)
    out = []
    for r in rd:
        def num(k):
            v = r[k].strip()
            return float(v) if v else None
        out.append(dict(
            beam=r["beam"], ad=num("ad"), s=num("s_mm"), target=num("target_loss"),
            wmax=num("wmax"), wavg=num("wavg"), loss=num("mass_loss"), fc=num("fc"),
            Pdiag=num("P_diag"), Pult=num("P_ult"), red=num("red_pct"),
            defl=num("defl"), eps_long=num("eps_long"), eps_conc=num("eps_conc"),
            eps_stir=num("eps_stirrup"), mode=r["mode"].strip()))
    return out

# --------------------------------------------------- geometry and materials
# Reported by the source (Sec. 2.1, 2.2, Fig. 1):
B      = 200.0        # web width, mm
H      = 350.0        # overall depth, mm
LSPAN  = 2400.0       # simply supported clear span, mm
N_LONG = 4            # 4 phi 25 main tensile bars, single layer (Fig. 1)
DB     = 25.0
N_TOP  = 2            # 2 phi 10
DSTIR  = 8.0          # phi 8 two-legged stirrups
NLEGS  = 2
COVER_SIDE = 20.0     # side cover TO THE STIRRUP, the only cover reported

# Source's own steel properties (Sec. 2.2 and Sec. 3.2):
FY   = {"long": 480.0, "top": 530.0, "stir": 495.0}    # MPa, Sec. 2.2
EPSY = {"long": 2400e-6, "stir": 2600e-6}              # Sec. 3.2, quoted verbatim
ES   = {k: FY[k]/EPSY[k] for k in EPSY}                # 200.0 and 190.4 GPa
EH_RATIO = 0.0        # perfectly plastic post-yield unless swept

# NOT reported by the source -> assumption, swept below.
D_NOM = 300.0         # effective depth to the centroid of the 4 phi 25 bars
D_ALT = 350.0 - 20.0 - DSTIR - DB/2.0     # = 309.5 mm if bottom cover = side cover

A_LONG = N_LONG * math.pi/4.0 * DB**2                  # 1963.50 mm^2
A_V    = NLEGS  * math.pi/4.0 * DSTIR**2               # 100.53 mm^2

def steel_stress(eps, kind):
    """Bilinear law with the source's own fy and yield strain."""
    E, fy, ey = ES[kind], FY[kind], EPSY[kind]
    e = abs(eps)
    if e <= ey:
        return E*e
    return fy + EH_RATIO*E*(e - ey)

# ---------------------------------------------------------------- lever arms
def z_frac(d):
    "z = 0.9 d : the ordinary design assumption."
    return 0.9*d

def z_cracked(d, fc):
    "Elastic cracked-transformed-section lever arm, tension steel only."
    Ec = 4700.0*math.sqrt(fc)
    n  = ES["long"]/Ec
    rho= A_LONG/(B*d)
    rn = rho*n
    k  = math.sqrt(2*rn + rn*rn) - rn
    return (1.0 - k/3.0)*d

def T_stressblock(M, d, fc):
    """Rectangular stress block closed on the section: T = M/(d - T/(1.7 fc b)).
    Returns (T_req, z)."""
    A = 1.0/(1.7*fc*B)
    # A T^2 - d T + M = 0
    disc = d*d - 4*A*M
    if disc < 0:
        return None, None
    T = (d - math.sqrt(disc))/(2*A)
    return T, M/T

# --------------------------------------------------------------- bisection
def bisect_theta(force_of_theta, target, lo=-2.0, hi=1.0, tol=1e-12):
    """T(theta) is affine and decreasing -> unique root.  Bracket widened below 0
    and above 1 only to REPORT inadmissible roots honestly."""
    flo, fhi = force_of_theta(lo)-target, force_of_theta(hi)-target
    if flo*fhi > 0:
        return None
    for _ in range(200):
        mid = 0.5*(lo+hi)
        fm = force_of_theta(mid)-target
        if flo*fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi-lo < tol:
            break
    return 0.5*(lo+hi)

# ================================================================ T1 - LONG
def t1_longitudinal(rows, d):
    """Free body: cut at midspan.  M = V*a, a = (a/d)*d, V = P_ult/2.
    T_req = M/z.  T(theta) = A_long (1-theta) sigma_s(eps_long)."""
    res = []
    for r in rows:
        if r["eps_long"] is None:
            continue
        a  = r["ad"]*d
        V  = r["Pult"]/2.0*1e3              # N
        M  = V*a                            # N.mm
        sig= steel_stress(r["eps_long"]*1e-6, "long")
        Tm = A_LONG*sig                     # N, measured tie force at theta = 0
        entry = dict(beam=r["beam"], ad=r["ad"], a_mm=a, V_kN=V/1e3, M_kNm=M/1e6,
                     eps=r["eps_long"], sigma_MPa=sig, yielded=r["eps_long"]*1e-6 > EPSY["long"],
                     T_meas_kN=Tm/1e3, theta_true=0.0)
        for name, z in (("z=0.9d", z_frac(d)),
                        ("z=cracked-elastic", z_cracked(d, r["fc"])),
                        ("z=stress-block", T_stressblock(M, d, r["fc"])[1])):
            Treq = M/z
            th = bisect_theta(lambda t: A_LONG*(1.0-t)*sig, Treq)
            entry[name] = dict(z_mm=z, z_over_d=z/d, T_req_kN=Treq/1e3,
                               theta=th, theta_pp_error=None if th is None else 100.0*th)
        res.append(entry)
    return res

# ================================================================ T1 - STIRRUP
def t1_stirrup(rows, d, alphas_deg):
    """Free body cut by the inclined shear crack.  The crack rises over the
    internal lever arm z_v = 0.9d at angle alpha to the horizontal; its horizontal
    projection is z_v*cot(alpha) and it is crossed by n = z_v*cot(alpha)/s stirrups,
    each vertical, each contributing A_v*sigma_v.  Vertical equilibrium of the free
    body with the concrete term neglected (T_c ~ 0, the study's constitutive map):

        V_s(theta) = n A_v (1-theta) sigma_v(eps_stirrup)  =  V_req = P_ult/2
    """
    zv = 0.9*d
    res = []
    for r in rows:
        if r["eps_stir"] is None:
            continue                        # corroded beams: NO stirrup gauge
        Vreq = r["Pult"]/2.0*1e3
        sig  = steel_stress(r["eps_stir"]*1e-6, "stir")
        entry = dict(beam=r["beam"], ad=r["ad"], s_mm=r["s"], V_req_kN=Vreq/1e3,
                     eps=r["eps_stir"], sigma_MPa=sig,
                     yielded=r["eps_stir"]*1e-6 > EPSY["stir"], theta_true=0.0,
                     angles={})
        for al in alphas_deg:
            n = zv/math.tan(math.radians(al))/r["s"]
            Vs0 = n*A_V*sig                 # stirrup contribution at theta = 0
            th = bisect_theta(lambda t: n*A_V*(1.0-t)*sig, Vreq)
            entry["angles"][f"{al:g}deg"] = dict(
                n_stirrups=n, Vs_theta0_kN=Vs0/1e3,
                Vs_over_Vreq=Vs0/Vreq,
                theta=th, admissible=(th is not None and 0.0 <= th <= 1.0),
                Vc_needed_kN=(Vreq-Vs0)/1e3, Vc_needed_frac=(Vreq-Vs0)/Vreq)
        # the "direct strut" limit: crack spans the whole shear span
        a = r["ad"]*d
        al_ds = math.degrees(math.atan2(zv, a))
        n = a/r["s"]
        Vs0 = n*A_V*sig
        th = bisect_theta(lambda t: n*A_V*(1.0-t)*sig, Vreq)
        entry["angles"]["direct-strut(%.1fdeg)" % al_ds] = dict(
            n_stirrups=n, Vs_theta0_kN=Vs0/1e3, Vs_over_Vreq=Vs0/Vreq,
            theta=th, admissible=(th is not None and 0.0 <= th <= 1.0),
            Vc_needed_kN=(Vreq-Vs0)/1e3, Vc_needed_frac=(Vreq-Vs0)/Vreq)
        res.append(entry)
    return res

# ======================================================================= T2
def t2(rows, d, alpha_deg=45.0):
    """Constitutive test.  Truss stirrup contribution
       V_s(theta) = (z_v cot alpha / s) A_v (1-theta) f_yv,
    concrete term V_c calibrated on the control beam of the SAME series
    (same a/d and same s):  V_c = V_control - V_s(0).
    Forward:  predict V_u and the strength reduction from the weighed theta.
    Inverse:  recover theta from the measured capacity drop, compare to weighed.
    """
    zv = 0.9*d
    ctrl = {}
    for r in rows:
        if r["loss"] is None:
            ctrl[(r["ad"], r["s"])] = r
    out = []
    for r in rows:
        if r["loss"] is None:
            continue
        c = ctrl[(r["ad"], r["s"])]
        n = zv/math.tan(math.radians(alpha_deg))/r["s"]
        Vs0 = n*A_V*FY["stir"]/1e3          # kN, control stirrup contribution
        Vc  = c["Pult"]/2.0 - Vs0           # kN, calibrated on the control
        th  = r["loss"]/100.0
        Vpred = Vc + Vs0*(1.0-th)
        Vmeas = r["Pult"]/2.0
        Vctrl = c["Pult"]/2.0
        red_pred = 100.0*(Vctrl-Vpred)/Vctrl
        red_meas = 100.0*(Vctrl-Vmeas)/Vctrl
        # inverse: theta implied by the measured capacity drop
        th_inv = bisect_theta(lambda t: Vc + Vs0*(1.0-t), Vmeas, lo=-3.0, hi=6.0)
        # decomposition of the measured loss
        dV      = Vctrl - Vmeas
        dV_steel= Vs0*th
        dV_conc = dV - dV_steel
        out.append(dict(
            beam=r["beam"], control=c["beam"], ad=r["ad"], s_mm=r["s"], mode=r["mode"],
            theta_weighed_pct=r["loss"],
            Vs0_kN=Vs0, Vc_kN=Vc, Vs0_frac_of_control=Vs0/Vctrl,
            V_ctrl_kN=Vctrl, V_meas_kN=Vmeas, V_pred_kN=Vpred,
            V_error_kN=Vpred-Vmeas, V_error_pct_of_capacity=100.0*(Vpred-Vmeas)/Vctrl,
            red_meas_pct=red_meas, red_reported_pct=r["red"], red_pred_pct=red_pred,
            red_error_pp=red_pred-red_meas,
            theta_from_capacity_pct=None if th_inv is None else 100.0*th_inv,
            theta_from_capacity_admissible=(th_inv is not None and 0.0 <= th_inv <= 1.0),
            theta_error_pp=None if th_inv is None else 100.0*th_inv - r["loss"],
            dV_total_kN=dV, dV_from_steel_kN=dV_steel, dV_from_concrete_kN=dV_conc,
            frac_of_loss_not_from_steel=None if dV == 0 else dV_conc/dV))
    return out


# ============================================== crack-angle sensitivity (T1-STIR)
def crack_angle_sensitivity(rows, d, alpha_ref=45.0, dalphas=(-10,-5,-2,2,5,10)):
    """The stirrup free body only closes if a concrete term V_c is admitted.
    Calibrate V_c on the CONTROL beam so that the identification returns exactly
    theta = 0 at the reference crack angle, then ask what theta the SAME condition
    returns when the assumed crack angle is perturbed.  Because theta_true is still
    0, every recovered value is pure geometric model error."""
    zv = 0.9*d
    out = []
    for r in rows:
        if r["eps_stir"] is None:
            continue
        Vreq = r["Pult"]/2.0*1e3
        sig  = steel_stress(r["eps_stir"]*1e-6, "stir")
        n_ref = zv/math.tan(math.radians(alpha_ref))/r["s"]
        Vs_ref = n_ref*A_V*sig
        Vc = Vreq - Vs_ref                       # calibrated so theta(alpha_ref)=0
        e = dict(beam=r["beam"], ad=r["ad"], s_mm=r["s"], alpha_ref_deg=alpha_ref,
                 Vc_calibrated_kN=Vc/1e3, Vc_frac=Vc/Vreq, pert={})
        for da in dalphas:
            al = alpha_ref + da
            n  = zv/math.tan(math.radians(al))/r["s"]
            th = bisect_theta(lambda t: Vc + n*A_V*(1.0-t)*sig, Vreq, lo=-5.0, hi=5.0)
            e["pert"][f"{da:+g}deg"] = dict(alpha_deg=al, theta_pct=None if th is None else 100.0*th)
        # pp of apparent section loss per degree of crack angle, central difference
        p2 = e["pert"]["+2deg"]["theta_pct"]; m2 = e["pert"]["-2deg"]["theta_pct"]
        e["pp_per_degree"] = (p2-m2)/4.0
        out.append(e)
    return out


# ================================== information content of a post-yield strain
def strain_information(rows, hardening_ratios=(0.0, 0.005, 0.01, 0.02)):
    """Every measured stirrup strain (2875-3829 ue) is past the source's own
    stirrup yield strain of 2600 ue.  Ask how much stress -- hence how much
    identifying information -- the strain spread actually carries."""
    eps = [r["eps_stir"] for r in rows if r["eps_stir"] is not None]
    lo, hi = min(eps), max(eps)
    out = dict(eps_min_ue=lo, eps_max_ue=hi, eps_spread_pct=100.0*(hi-lo)/lo,
               yield_strain_ue=EPSY["stir"]*1e6, all_past_yield=all(e > EPSY["stir"]*1e6 for e in eps),
               cases={})
    global EH_RATIO
    keep = EH_RATIO
    for h in hardening_ratios:
        EH_RATIO = h
        slo, shi = steel_stress(lo*1e-6, "stir"), steel_stress(hi*1e-6, "stir")
        out["cases"][f"Eh/Es={h}"] = dict(sigma_min_MPa=slo, sigma_max_MPa=shi,
                                          sigma_spread_pct=100.0*(shi-slo)/slo,
                                          theta_spread_pp=100.0*(1.0 - slo/shi))
    EH_RATIO = keep
    # same question for the LONGITUDINAL bars, which stayed elastic
    el = [r["eps_long"] for r in rows if r["eps_long"] is not None]
    out["longitudinal_elastic"] = dict(
        eps_min_ue=min(el), eps_max_ue=max(el), all_below_yield=all(e < EPSY["long"]*1e6 for e in el),
        sigma_spread_pct=100.0*(max(el)-min(el))/min(el))
    return out

# ===================================================================== main
def main():
    rows = load()
    report = dict(
        source="El-Sayed, Hussain & Shuraim (2016) JCEM 22(4):491-499",
        assumptions=dict(
            d_not_reported_by_source=True, d_used_mm=D_NOM, d_alternative_mm=D_ALT,
            Es_from_source_fy_over_yield_strain_MPa=ES,
            fy_MPa=FY, A_long_mm2=A_LONG, A_v_mm2=A_V,
            post_yield_modulus="perfectly plastic (EH_RATIO=0)"))

    for d in (D_NOM, D_ALT):
        key = f"d={d:.1f}mm"
        report.setdefault("T1_longitudinal", {})[key] = t1_longitudinal(rows, d)
        report.setdefault("T1_stirrup",      {})[key] = t1_stirrup(rows, d, [30, 35, 45])
        report.setdefault("T2",              {})[key] = t2(rows, d)
        report.setdefault("T1_stirrup_angle_sensitivity", {})[key] = crack_angle_sensitivity(rows, d)

    # ------------------------------------------------------------- summaries
    def summ_long(res):
        s = {}
        for k in ("z=0.9d", "z=cracked-elastic", "z=stress-block"):
            th = [100.0*e[k]["theta"] for e in res if e[k]["theta"] is not None]
            s[k] = dict(n=len(th), mean_pp=sum(th)/len(th),
                        min_pp=min(th), max_pp=max(th),
                        mean_abs_pp=sum(abs(x) for x in th)/len(th),
                        spread_pp=max(th)-min(th))
        # spread across lever-arm choices, per beam
        per = [max(100.0*e[k]["theta"] for k in ("z=0.9d","z=cracked-elastic","z=stress-block"))
               - min(100.0*e[k]["theta"] for k in ("z=0.9d","z=cracked-elastic","z=stress-block"))
               for e in res]
        s["lever_arm_spread_per_beam_pp"] = dict(mean=sum(per)/len(per), min=min(per), max=max(per))
        return s

    def summ_t2(res):
        e_red = [abs(r["red_error_pp"]) for r in res]
        e_th  = [r["theta_error_pp"] for r in res if r["theta_error_pp"] is not None]
        e_cap = [abs(r["V_error_pct_of_capacity"]) for r in res]
        f_nc  = [r["frac_of_loss_not_from_steel"] for r in res]
        return dict(n=len(res),
                    mean_abs_reduction_error_pp=sum(e_red)/len(e_red), max_abs_reduction_error_pp=max(e_red),
                    mean_abs_capacity_error_pct=sum(e_cap)/len(e_cap), max_abs_capacity_error_pct=max(e_cap),
                    theta_error_mean_pp=sum(e_th)/len(e_th), theta_error_min_pp=min(e_th), theta_error_max_pp=max(e_th),
                    mean_frac_of_loss_not_from_steel=sum(f_nc)/len(f_nc))

    report["strain_information"] = strain_information(rows)
    report["summary"] = dict(
        T1_longitudinal={k: summ_long(v) for k, v in report["T1_longitudinal"].items()},
        T2={k: summ_t2(v) for k, v in report["T2"].items()},
        T1_stirrup_note=("stirrup strain is reported ONLY for the five uncorroded controls; "
                         "no corroded beam carries a stirrup gauge, so section loss and strain "
                         "never co-occur on the same specimen"))

    out = os.path.join(DATA, "elsayed2016_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)

    # ------------------------------------------------------------- printout
    W = 96
    print("="*W); print("T1-LONG  longitudinal tie at midspan, theta_true = 0 (bars epoxy coated)"); print("="*W)
    print(f"{'beam':10s} {'a/d':>4s} {'V,kN':>7s} {'eps,ue':>7s} {'sig,MPa':>8s} {'Tmeas,kN':>9s}"
          f" {'th(0.9d)':>9s} {'th(elas)':>9s} {'th(blk)':>9s} {'spread':>7s}")
    for e in report["T1_longitudinal"][f"d={D_NOM:.1f}mm"]:
        ths = [100.0*e[k]["theta"] for k in ("z=0.9d","z=cracked-elastic","z=stress-block")]
        print(f"{e['beam']:10s} {e['ad']:4.0f} {e['V_kN']:7.1f} {e['eps']:7.0f} {e['sigma_MPa']:8.1f}"
              f" {e['T_meas_kN']:9.1f} {ths[0]:8.1f}% {ths[1]:8.1f}% {ths[2]:8.1f}% {max(ths)-min(ths):6.1f}pp")
    for k, v in report["summary"]["T1_longitudinal"][f"d={D_NOM:.1f}mm"].items():
        print("  ", k, {kk: (round(vv,2) if isinstance(vv,float) else vv) for kk,vv in v.items()})

    print(); print("="*W); print("T1-STIR  stirrup free body, theta_true = 0 (CONTROL beams -- the only ones gauged)"); print("="*W)
    for e in report["T1_stirrup"][f"d={D_NOM:.1f}mm"]:
        print(f"{e['beam']:10s} a/d={e['ad']:.0f} s={e['s_mm']:.0f}mm  V_req={e['V_req_kN']:.1f}kN  "
              f"eps={e['eps']:.0f}ue -> sigma={e['sigma_MPa']:.0f}MPa (yielded={e['yielded']})")
        for al, a in e["angles"].items():
            th = "no root in [-2,1]" if a["theta"] is None else f"{100*a['theta']:8.1f}%"
            print(f"     {al:>22s}  n={a['n_stirrups']:5.2f}  Vs={a['Vs_theta0_kN']:7.1f}kN "
                  f"({100*a['Vs_over_Vreq']:5.1f}% of V)  theta={th:>18s}  admissible={a['admissible']}"
                  f"  Vc needed={a['Vc_needed_kN']:7.1f}kN ({100*a['Vc_needed_frac']:.0f}%)")

    print(); print("="*W); print("T2  does the weighed mass loss predict the measured capacity?"); print("="*W)
    print(f"{'beam':10s} {'ctrl':10s} {'th_wgh':>7s} {'Vs0/V':>6s} {'Vctrl':>7s} {'Vmeas':>7s} {'Vpred':>7s}"
          f" {'errkN':>7s} {'red_m':>6s} {'red_p':>6s} {'errpp':>6s} {'th_cap':>7s} {'th_err':>7s} {'%loss!steel':>11s}")
    for r in report["T2"][f"d={D_NOM:.1f}mm"]:
        print(f"{r['beam']:10s} {r['control']:10s} {r['theta_weighed_pct']:6.1f}% {r['Vs0_frac_of_control']:6.3f}"
              f" {r['V_ctrl_kN']:7.1f} {r['V_meas_kN']:7.1f} {r['V_pred_kN']:7.1f} {r['V_error_kN']:7.1f}"
              f" {r['red_meas_pct']:5.1f}% {r['red_pred_pct']:5.1f}% {r['red_error_pp']:5.1f}"
              f" {r['theta_from_capacity_pct']:6.1f}%{'!' if not r['theta_from_capacity_admissible'] else ' '}{r['theta_error_pp']:+6.1f} {100*r['frac_of_loss_not_from_steel']:10.0f}%")
    print("  summary:", {k:(round(v,2) if isinstance(v,float) else v)
                          for k,v in report["summary"]["T2"][f"d={D_NOM:.1f}mm"].items()})
    print(); print("="*W); print("CRACK-ANGLE SENSITIVITY: Vc calibrated so theta=0 at 45 deg, then angle perturbed")
    print("="*W)
    print(f"{'beam':10s} {'Vc/V':>6s} " + " ".join(f"{k:>9s}" for k in ("-10deg","-5deg","-2deg","+2deg","+5deg","+10deg")) + f" {'pp/deg':>7s}")
    for e in report["T1_stirrup_angle_sensitivity"][f"d={D_NOM:.1f}mm"]:
        cells = " ".join(f"{e['pert'][k]['theta_pct']:8.1f}%" for k in ("-10deg","-5deg","-2deg","+2deg","+5deg","+10deg"))
        print(f"{e['beam']:10s} {e['Vc_frac']:6.3f} " + cells + f" {e['pp_per_degree']:7.2f}")
    print()
    print("d-sensitivity of the T1-LONG bias (mean recovered theta, theta_true = 0):")
    for dk, sm in report["summary"]["T1_longitudinal"].items():
        print("  ", dk, {k: round(v["mean_pp"],2) for k, v in sm.items() if k != "lever_arm_spread_per_beam_pp"})
    print("d-sensitivity of the T2 inverse (mean theta error, pp):")
    for dk, sm in report["summary"]["T2"].items():
        print("  ", dk, round(sm["theta_error_mean_pp"],2))
    si = report["strain_information"]
    print(); print("="*W); print("INFORMATION CONTENT OF THE MEASURED STIRRUP STRAIN"); print("="*W)
    print(f"  stirrup strains span {si['eps_min_ue']:.0f}-{si['eps_max_ue']:.0f} ue "
          f"({si['eps_spread_pct']:.0f}% spread); yield strain {si['yield_strain_ue']:.0f} ue; "
          f"all past yield = {si['all_past_yield']}")
    for k, v in si["cases"].items():
        print(f"    {k:14s} sigma {v['sigma_min_MPa']:7.1f} -> {v['sigma_max_MPa']:7.1f} MPa "
              f"({v['sigma_spread_pct']:5.2f}% spread)  == {v['theta_spread_pp']:5.2f} pp of theta")
    L = si["longitudinal_elastic"]
    print(f"  longitudinal bars {L['eps_min_ue']:.0f}-{L['eps_max_ue']:.0f} ue, all below yield = "
          f"{L['all_below_yield']}, so sigma spread = strain spread = {L['sigma_spread_pct']:.0f}%")
    print(); print("wrote", out)

if __name__ == "__main__":
    main()
