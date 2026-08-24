"""
DATASET 2 -- Davis, Hoult & Scott (2017), Eng. Struct. 140:473-482.
doi:10.1016/j.engstruct.2017.03.013

TEST TYPE: T2 (CONSTITUTIVE), not T1.
The article tabulates NO strain. Table 1 is the only table in the paper; all fiber-optic
strain (Figs 5-10) exists solely as raster plots. So the identifying condition cannot be
evaluated from measured strain and a theta cannot be recovered from strain. What CAN be
tested is whether the weighed section loss predicts the measured ultimate load.

Everything below is driven by Table 1 + Section 3.1/3.3 text + printed Figure 1 dimensions.
Nothing is digitised off a plot.

Model (the study's own map: bilinear steel, concrete tension neglected, T_c = 0):
    T(theta)   = T_c + A_s (1-theta) sigma_s          [rho_tie(1-theta)S -> A_s(1-theta)sigma_s]
    a(theta)   = T(theta) / (0.85 f'_c b)             [rectangular compression block]
    M(theta)   = T(theta) * (d - a(theta)/2)          [lever arm jd = d - a/2]
    P(theta)   = 4 M(theta) / L_span                  [three-point bending, M = P L / 4]

sigma_s is calibrated PER SLEEVE SERIES on that series' own control so that P(0) reproduces
the measured control capacity. That is the "fix the lever arm on the controls" step: it
absorbs strain hardening into an effective sigma_s * jd product.

Inversion: P(theta) is strictly decreasing in theta, so the measured P_u has a unique root,
found by bisection -- exactly the root-find the identification method uses, but driven by
force instead of strain.
"""
import json, math, csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

# ---------------------------------------------------------------- source values
FY_BOT   = 450.0     # MPa, "two 20M bottom bars with a yield strength of 450 MPa" (Sec 3.1)
ES       = 200000.0  # MPa, NOT reported by the source; standard value
FC       = 45.0      # MPa, "average compressive strength of 45 MPa at the time of testing"
B        = 200.0     # mm, Sec 3.1 + Fig 1(c)
D_EFF    = 250.0     # mm, "effective depth of 250 mm" + Fig 1
L_SPAN   = 2000.0    # mm, "span between the supports was 2.0 m"
L_BEAM   = 2300.0    # mm
N_BOT    = 2
AB_20M   = 300.0     # mm^2, CSA G30.18 nominal for 20M -- NOT stated in the paper
DB_20M   = 19.5      # mm, CSA G30.18 nominal -- NOT stated in the paper
AS       = N_BOT * AB_20M          # 600 mm^2
SLEEVE   = {"short": 100.0, "long": 300.0}   # mm penetration into the beam, Fig 1(a),(b)

# Table 1, verbatim
BEAMS = [
    # name,   sleeve,  barA%,  barB%,  avg%,  Pu(kN)
    ("BS-C",  "short", None,   None,   0.0,   156.9),
    ("BL-C",  "long",  None,   None,   0.0,   151.5),
    ("BS-01", "short", 1.3,    1.1,    1.2,   136.1),
    ("BL-01", "long",  1.3,    1.5,    1.4,    94.0),
    ("BS-05", "short", 4.2,    3.9,    4.0,   108.2),
    ("BL-05", "long",  4.0,    3.5,    3.8,    94.7),
]
CONTROL = {"short": 156.9, "long": 151.5}

# ---------------------------------------------------------------- forward map
def tie_force(theta, sigma_s, As=AS, Tc=0.0):
    """T(theta) = T_c + A_s (1-theta) sigma_s.  Affine, decreasing in theta."""
    return Tc + As * (1.0 - theta) * sigma_s

def capacity_kN(theta, sigma_s, As=AS):
    T  = tie_force(theta, sigma_s, As)
    a  = T / (0.85 * FC * B)
    jd = D_EFF - a / 2.0
    assert a < D_EFF, "compression block deeper than d -- model invalid"
    M  = T * jd                       # N.mm
    return 4.0 * M / L_SPAN / 1e3, jd, a   # kN, mm, mm


def M_of_T(T):
    """M = T (d - a/2), a = T/(0.85 f'c b).  Increasing in T only while a < d."""
    return T * (D_EFF - T/(0.85*FC*B)/2.0)

T_BAL = 0.85*FC*B*D_EFF          # T at which a = d; M(T) peaks here, then falls
def T_from_P(P_kN):
    """Tie force consistent with a measured three-point-bending load, on the ASCENDING
       branch a < d (bracket capped at T_BAL, otherwise the root is not unique)."""
    lo, hi = 0.0, T_BAL
    assert 4.0*M_of_T(hi)/L_SPAN/1e3 > P_kN, "load exceeds the balanced-section moment"
    for _ in range(200):
        T = 0.5*(lo+hi)
        if 4.0*M_of_T(T)/L_SPAN/1e3 < P_kN: lo = T
        else: hi = T
    return 0.5*(lo+hi)

def calibrate_sigma(P_control_kN, As=AS):
    """Bisect sigma_s so that P(theta=0) == measured control capacity."""
    lo, hi = 1.0, 3000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if capacity_kN(0.0, mid, As)[0] < P_control_kN: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def recover_theta(P_meas_kN, sigma_s, As=AS):
    """Bisection on the identifying condition P(theta) = P_measured.
       P is strictly decreasing in theta -> unique root."""
    lo, hi = -0.60, 0.99
    assert capacity_kN(lo, sigma_s, As)[0] > P_meas_kN > capacity_kN(hi, sigma_s, As)[0], "root not bracketed"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if capacity_kN(mid, sigma_s, As)[0] > P_meas_kN: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- run
sig = {k: calibrate_sigma(v) for k, v in CONTROL.items()}
print("="*104)
print("DAVIS, HOULT & SCOTT (2017)  --  T2 CONSTITUTIVE TEST  (no tabulated strain -> T1 impossible)")
print("="*104)
print(f"A_s = {AS:.0f} mm^2 (2 x 20M, CSA nominal)   f_y = {FY_BOT:.0f} MPa   f'_c = {FC:.0f} MPa"
      f"   b = {B:.0f} mm   d = {D_EFF:.0f} mm   L = {L_SPAN/1e3:.1f} m")
print()
print("Lever-arm / effective-stress calibration on the CONTROLS:")
for k in ("short", "long"):
    P, jd, a = capacity_kN(0.0, sig[k])
    print(f"  {k:5s} sleeve  control P_u = {CONTROL[k]:6.1f} kN  ->  sigma_s,eff = {sig[k]:6.1f} MPa "
          f"(= {sig[k]/FY_BOT:.2f} f_y)   a = {a:5.1f} mm   jd = {jd:5.1f} mm   jd/d = {jd/D_EFF:.3f}")
jd_y = capacity_kN(0.0, FY_BOT)[1]
print(f"  at sigma_s = f_y = 450 MPa the section gives M_y -> P_y = {capacity_kN(0.0, FY_BOT)[0]:.1f} kN, jd = {jd_y:.1f} mm")
print(f"  -> both controls exceed the yield-based capacity; the calibration is absorbing STRAIN HARDENING")
print(f"     (paper: BS-C 'reinforcement yielded at approximately 140 kN' then rose to 156.9 kN).")
print()

hdr = (f"{'beam':6s} {'sleeve':6s} {'theta_wgt':>9s} {'P_meas':>8s} {'P_pred':>8s} {'dP':>7s} "
       f"{'err%':>7s} {'theta_rec':>9s} {'err_pp':>8s} {'amplif':>7s}")
print(hdr); print("-"*len(hdr))
rows = []
for name, sl, bA, bB, avg, Pu in BEAMS:
    th   = avg / 100.0
    s    = sig[sl]
    Ppred = capacity_kN(th, s)[0]
    threc = recover_theta(Pu, s)
    err_pp = 100.0 * threc - avg
    amp    = (100.0 * threc / avg) if avg > 0 else float('nan')
    print(f"{name:6s} {sl:6s} {avg:8.1f}% {Pu:8.1f} {Ppred:8.1f} {Ppred-Pu:7.1f} "
          f"{100*(Ppred-Pu)/Pu:6.1f}% {100*threc:8.1f}% {err_pp:+8.1f} "
          f"{amp:7.1f}" if avg > 0 else
          f"{name:6s} {sl:6s} {avg:8.1f}% {Pu:8.1f} {Ppred:8.1f} {Ppred-Pu:7.1f} "
          f"{100*(Ppred-Pu)/Pu:6.1f}% {100*threc:8.1f}% {err_pp:+8.1f} {'  (calib)':>7s}")
    rows.append(dict(beam=name, sleeve=sl, theta_weighed_pct=avg, P_meas_kN=Pu,
                     P_pred_kN=round(Ppred,1), load_err_pct=round(100*(Ppred-Pu)/Pu,1),
                     theta_recovered_pct=round(100*threc,1), err_pp=round(err_pp,1),
                     amplification=(round(amp,1) if avg>0 else "")))
print()

# ------------------------------------------------- sleeve-length correction on mass loss
print("Mass loss is gravimetric over the WHOLE bar, but the sleeved ends did not corrode")
print("(Sec 3.1: 'no corrosion occurred in the bars underneath the sleeves').")
print("Local section loss over the exposed length is therefore larger:")
print(f"  {'beam':6s} {'L_corroded':>11s} {'factor':>7s} {'theta_avg':>9s} {'theta_local':>11s} {'still needed':>13s}")
for name, sl, bA, bB, avg, Pu in BEAMS:
    if avg == 0: continue
    Lc  = L_BEAM - 2*SLEEVE[sl]
    f   = L_BEAM / Lc                     # lower bound: bar ends protrude beyond the concrete
    thl = avg * f
    need = 100*recover_theta(Pu, sig[sl])
    print(f"  {name:6s} {Lc:10.0f}mm {f:7.3f} {avg:8.1f}% {thl:10.2f}% {need:12.1f}%")
print("  (factor is a LOWER bound: the bars also protrude past the beam ends under the sleeves)")
print()

# ------------------------------------------------- pitting plausibility
print("Could localized pitting explain the gap?  Ratio of the section loss the load DEMANDS")
print("to the weighed average loss:")
for name, sl, bA, bB, avg, Pu in BEAMS:
    if avg == 0: continue
    need = 100*recover_theta(Pu, sig[sl])
    tag = "uniform corrosion (Sec 4.1)" if "01" in name else "distributed + MINOR pitting (Sec 4.1)"
    print(f"  {name:6s} needs {need:5.1f}% local loss vs {avg:.1f}% weighed  -> x{need/avg:5.1f}   [{tag}]")
print()

# ------------------------------------------------- the competing explanation: bonded length
print("="*104)
print("THE NON-MONOTONICITY:  BL-01 (1.4%, 94.0 kN) carries LESS than BS-05 (4.0%, 108.2 kN).")
print("="*104)
print("Within each sleeve series the ordering is fine; the reversal is entirely a SLEEVE effect.")
print("Bonded length available between the end of the sleeve and midspan:")
bond = {}
for sl in ("short", "long"):
    Lb = L_SPAN/2.0 + (L_BEAM - L_SPAN)/2.0 - SLEEVE[sl]   # midspan -> end of sleeve
    bond[sl] = Lb
    peri = N_BOT * math.pi * DB_20M * Lb
    print(f"  {sl:5s} sleeve: L_bond = {Lb:6.0f} mm   bar surface = {peri:9.0f} mm^2")
# ACI development length for the uncorroded bar
ld = FY_BOT / (1.1 * math.sqrt(FC) * 2.5) * DB_20M
print(f"  ACI 318 development length for an uncorroded 20M at f_y=450, f'c=45: l_d = {ld:.0f} mm")
print(f"  -> BOTH configurations develop the bar when uncorroded (l_d << {bond['long']:.0f} mm), which is why")
print(f"     BS-C and BL-C reach essentially the same load (156.9 vs 151.5 kN).")
print()
print("Now invert for the average bond stress each corroded beam actually mobilised,")
print("u = T_req / (n pi d_b L_bond), with T_req = M_u / jd from the same section model:")
print(f"  {'beam':6s} {'P_u,kN':>7s} {'M_u,kNm':>8s} {'T_req,kN':>8s} {'L_bnd':>6s} {'u,MPa':>7s}")
us = []
for name, sl, bA, bB, avg, Pu in BEAMS:
    s = sig[sl]
    th = avg/100.0
    T = T_from_P(Pu)   # T_req consistent with the measured moment and the model lever arm
    u = T / (N_BOT * math.pi * DB_20M * bond[sl])
    mark = "  <- lower bound (yielded/hardened, did not fail in bond)" if avg == 0 else ""
    Mnm = M_of_T(T)/1e6
    print(f"  {name:6s} {Pu:7.1f} {Mnm:8.1f} {T/1e3:8.1f} {bond[sl]:6.0f} {u:7.2f}{mark}")
    if avg > 0: us.append((name, sl, Pu, u))
umean = sum(u for _,_,_,u in us)/len(us)
print(f"\n  mean residual bond stress over the four CORRODED beams: u = {umean:.2f} MPa "
      f"(spread {min(u for *_ ,u in us):.2f} to {max(u for *_ ,u in us):.2f}, CoV "
      f"{100*(sum((u-umean)**2 for *_ ,u in us)/len(us))**0.5/umean:.0f}%)")
print("\n  Forward-predict all four corroded capacities from that SINGLE bond stress "
      "(geometry only, no theta):")
print(f"  {'beam':6s} {'P_meas':>8s} {'P_bond':>8s} {'err%':>7s}   vs  {'P_sectionloss':>13s} {'err%':>7s}")
for name, sl, bA, bB, avg, Pu in BEAMS:
    if avg == 0: continue
    T  = umean * N_BOT * math.pi * DB_20M * bond[sl]
    Pb = 4.0*M_of_T(T)/L_SPAN/1e3
    Ps = capacity_kN(avg/100.0, sig[sl])[0]
    print(f"  {name:6s} {Pu:8.1f} {Pb:8.1f} {100*(Pb-Pu)/Pu:6.1f}%   vs  {Ps:13.1f} {100*(Ps-Pu)/Pu:6.1f}%")
    for r in rows:
        if r["beam"] == name:
            r["P_bond_model_kN"] = round(Pb,1); r["bond_err_pct"] = round(100*(Pb-Pu)/Pu,1)
print()

# ------------------------------------------------- partial T1 on the control only
print("="*104)
print("PARTIAL T1 (control only, author-stated strain event -- NOT a digitisation)")
print("="*104)
print("Sec 4.3: 'At 137 kN, the reinforcement is yielding at midspan as shown by the large strain")
print("increase in that region.'  137 kN is a recorded FOS load stage. That is a strain-derived")
print("statement: eps_s >= eps_y = f_y/E_s = %.0f microstrain, i.e. sigma_s = f_y at the cut." % (1e6*FY_BOT/ES))
P_y = 137.0
Treq = T_from_P(P_y)
th_c = 1.0 - Treq/(AS*FY_BOT)
print(f"  T_req at 137 kN (M = {137*0.5:.1f} kNm, jd = {D_EFF - Treq/(0.85*FC*B)/2.0:.1f} mm) = {Treq/1e3:.1f} kN")
print(f"  T(theta) = A_s(1-theta) f_y = {AS*FY_BOT/1e3:.1f}(1-theta) kN")
print(f"  bisection root:  theta = {100*th_c:+.1f}%   against the known theta = 0.0% (BS-C is a control)")
print(f"  -> the statics + bilinear-steel chain is off by {100*th_c:+.1f} pp with ZERO corrosion and")
print(f"     intact bond. That is the model-error floor of the chain on this specimen.")
print()

with open(os.path.join(DATA, "davis2017_t2_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["beam","sleeve","theta_weighed_pct","P_meas_kN","P_pred_kN",
                                      "load_err_pct","theta_recovered_pct","err_pp","amplification",
                                      "P_bond_model_kN","bond_err_pct"])
    w.writeheader()
    for r in rows: w.writerow(r)
print("wrote", os.path.join(DATA, "davis2017_t2_results.csv"))

# ------------------------------------------------- reconciliation of the partial T1
sig_needed = Treq / AS
print(f"  Reconciliation: theta = 0 is recovered exactly if sigma_s at that load stage is")
print(f"  {sig_needed:.0f} MPa = {sig_needed/FY_BOT:.2f} f_y.  BS-C was already on the hardening branch")
print(f"  (yield knee ~140 kN, load rose to 156.9 kN), so 1.10 f_y is physically ordinary.")
print(f"  Read the other way: insisting sigma_s = f_y displaces theta by {100*th_c:+.1f} pp.")
print()

# ------------------------------------------------- consistency checks
print("="*104)
print("CONSISTENCY CHECKS")
print("="*104)
# 1. capacity reductions vs the paper's own text
print("1. Capacity reduction recomputed from Table 1 vs the paper's own text:")
for name, sl, bA, bB, avg, Pu in BEAMS:
    if avg == 0: continue
    red = 100*(1 - Pu/CONTROL[sl])
    print(f"   {name:6s} {Pu:6.1f} / {CONTROL[sl]:6.1f} -> {red:5.1f}% reduction")
print("   paper Sec 4.2: 13% / 31% (short), 38% / 37% (long)  -> matches Table 1")
print("   paper Conclusion 3: '13% and 39%' for the short series -> the 39% is an ERROR in the")
print("   paper (Table 1 and Sec 4.2 both give 31%); Conclusion 3 also says 1.1% where Table 1")
print("   says 1.2% for BS-01. Use Table 1.")
# 2. shear
Av = 2*100.0
Vs = Av*440.0*D_EFF/150.0/1e3
Vc = 0.17*math.sqrt(FC)*B*D_EFF/1e3
print(f"\n2. Is the flexural free body the right one?  V_max = P/2 = {156.9/2:.1f} kN at the")
print(f"   highest load recorded.  Stirrup capacity V_s = A_v f_yv d/s = {Vs:.0f} kN,")
print(f"   V_c ~ 0.17 sqrt(f'c) b d = {Vc:.0f} kN, total {Vs+Vc:.0f} kN >> {156.9/2:.1f} kN.")
print(f"   a/d = {1000/D_EFF:.1f} -> slender B-region flexural member, shear does not govern.")
# 3. per-bar spread
print("\n3. Bar-to-bar spread of the weighed mass loss (Table 1), i.e. how well 'theta' is even")
print("   defined for a two-bar chord:")
for name, sl, bA, bB, avg, Pu in BEAMS:
    if bA is None: continue
    print(f"   {name:6s} bar A {bA:.1f}%  bar B {bB:.1f}%  mean {avg:.1f}%  spread "
          f"{abs(bA-bB):.1f} pp ({100*abs(bA-bB)/avg:.0f}% of the mean)")
print("\n4. Elasticity of capacity to weighed mass loss, d(P/P0)/d(theta).  The section-loss")
print("   model predicts 1.0 by construction:")
for name, sl, bA, bB, avg, Pu in BEAMS:
    if avg == 0: continue
    print(f"   {name:6s} {(1 - Pu/CONTROL[sl])/(avg/100):6.1f}")
print(f"   BL-01 -> BL-05 (1.4% -> 3.8%, 94.0 -> 94.7 kN): "
      f"{((94.0-94.7)/151.5)/((3.8-1.4)/100):6.2f}   <- NEGATIVE: within the long-sleeve series,")
print( "   capacity is flat while mass loss nearly triples. That is the fingerprint of a limit")
print( "   state that does not depend on the bar section.")
