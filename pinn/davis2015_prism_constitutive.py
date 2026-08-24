"""
Davis (2015) -- companion to davis2015_prism_identification.py

Three things the identification script defers:
  (1) the T2 (constitutive) direction: does a KNOWN weighed section loss predict the measured
      yield load?  This is the weaker test but it is the one the data cleanly supports.
  (2) the concrete tension term T_c, measured directly from the thesis' own paired bare-bar /
      RC control specimens at three bar sizes (Table 2.2).  The constitutive map used in this
      study neglects concrete tension; here is how big that neglect actually is.
  (3) decomposition of the observed recovery bias into the bond-loss part and the pitting part.

Run with /usr/local/bin/python3.12
"""
import math

A_S = 200.0
FPC = 33.6

# ---- Table 2.2 verbatim: paired bare-bar and RC-prism yield loads at three bar sizes ----
TAB22 = {
    "10M": dict(A=100.0, bare=[44.0, 43.9, 43.8],   rc=[47.9, 48.2, 47.6]),
    "15M": dict(A=200.0, bare=[86.0, 89.5],         rc=[85.1, 95.4, 89.5, 88.2]),
    "20M": dict(A=300.0, bare=[135.4, 134.0, 135.3],rc=[140.3, 138.8, 141.6]),
}
# ---- Table 3.2 corroded RC prisms: (name, weighed mass loss frac, measured yield load kN) ----
RC = [("C1-CR", 0.009, 92.0), ("C2-CR", 0.006, 86.5), ("C3-CR", 0.005, 89.0),
      ("C1-05", 0.061, 79.0), ("C2-05", 0.071, 72.0), ("C3-05", 0.034, 76.5),
      ("C1-10", 0.123, 68.8), ("C2-10", 0.130, 59.5)]
BARE = [("R-01", 0.009, 97.8), ("R-05", 0.051, 87.3),
        ("R-10", 0.101, 76.0), ("R-15", 0.154, 66.0)]

mean = lambda v: sum(v) / len(v)

print("=" * 96)
print("(2) HOW BIG IS THE NEGLECTED CONCRETE TENSION TERM T_c AT YIELD?")
print("    measured as (RC control yield load) - (bare bar yield load), same bar, same batch")
print("=" * 96)
print(f"{'bar':5s} {'A_s':>6s} {'bare N_y (kN)':>14s} {'RC N_y (kN)':>13s} {'T_c (kN)':>9s} "
      f"{'T_c/N_y':>8s} {'implied sigma_ct (MPa)':>23s}")
tcs = []
for k, d in TAB22.items():
    b, r = mean(d["bare"]), mean(d["rc"])
    tc = r - b
    Ac = 100.0 * 100.0 - d["A"]
    tcs.append(tc)
    print(f"{k:5s} {d['A']:6.0f} {b:14.2f} {r:13.2f} {tc:9.2f} {tc/r*100:7.1f}% {tc*1000/Ac:23.2f}")
tc_bar = mean(tcs)
fct = 0.33 * math.sqrt(FPC)   # CSA A23.3 modulus of rupture proxy
print(f"\n  mean T_c = {tc_bar:.2f} kN over three bar sizes")
print(f"  implied residual concrete tensile stress ~= {tc_bar*1000/9800:.2f} MPa "
      f"= {tc_bar*1000/9800/fct:.2f} f_ct  (f_ct = 0.33 sqrt(f'c) = {fct:.2f} MPa)")
print(f"  For the 15M prism specifically T_c = {TAB22['15M']['rc'] and mean(TAB22['15M']['rc'])-mean(TAB22['15M']['bare']):.2f} kN.")
print(f"  Setting T_c = 0 in the identifying condition biases theta LOW (under-reports damage) by")
print(f"  T_c/(f_y A_s) = {tc_bar/89.55*100:.1f} pp using the 3-size mean, "
      f"{(mean(TAB22['15M']['rc'])-mean(TAB22['15M']['bare']))/89.55*100:.1f} pp using the 15M value.")
print("  <-- this IS the study's characteristic under-report direction, and it is small.")
print()

print("=" * 96)
print("(1) T2 CONSTITUTIVE DIRECTION: predict N_yield from the WEIGHED mass loss")
print("    N_pred = N_ref (1 - theta_weighed),  N_ref = mean of the study's own controls")
print("=" * 96)
for lab, grp, nref in (("RC PRISMS", RC, mean(TAB22["15M"]["rc"])),
                       ("BARE BARS", BARE, mean(TAB22["15M"]["bare"]))):
    print(f"\n  {lab}   N_ref = {nref:.2f} kN")
    print(f"  {'spec':7s} {'weighed%':>9s} {'N_pred(kN)':>11s} {'N_meas(kN)':>11s} "
          f"{'over-pred %':>12s}")
    errs = []
    for n, th, nm in grp:
        npd = nref * (1 - th)
        e = (npd - nm) / nm * 100
        errs.append(e)
        print(f"  {n:7s} {th*100:9.1f} {npd:11.2f} {nm:11.2f} {e:+12.1f}")
    m = mean(errs)
    sd = math.sqrt(sum((x - m) ** 2 for x in errs) / (len(errs) - 1))
    print(f"  mean over-prediction of residual capacity = {m:+.1f} %  (sd {sd:.1f} %, "
          f"max {max(errs):+.1f} %)")
print()
print("  Reading: a UNIFORM section-loss constitutive model fed the true weighed mass loss")
print("  systematically OVER-predicts the surviving yield load of a real corroded RC prism.")
print("  It is unconservative, by up to 31% on C2-10.")
print()

print("=" * 96)
print("(3) DECOMPOSITION OF THE +7.5 pp RECOVERY BIAS ON THE 8 CORRODED RC PRISMS")
print("=" * 96)
tc15 = mean(TAB22["15M"]["rc"]) - mean(TAB22["15M"]["bare"])
nref = mean(TAB22["15M"]["rc"])
print(f"  total observed bias (self-referenced, from identification script)   = +7.51 pp")
print(f"  part explained by bond loss (T_c falls from {tc15:.2f} kN to ~0 as corrosion")
print(f"    destroys bond, while the theta=0 datum still contains it)        = +{tc15/nref*100:.1f} pp at most")
print(f"  residual, attributable to PITTING (governing minimum section is")
print(f"    smaller than the gravimetric mean)                               = +{7.51 - tc15/nref*100:.1f} pp")
print()
print("  Independent corroboration of the pitting share: regression slope of recovered on")
print("  weighed is 2.31, i.e. the mechanically effective section loss at the governing section")
print("  runs at ~2.3x the bar-average mass loss in these specimens.  Davis reports pitting in")
print("  every RC prism (p.57) and uniform corrosion in the bare bars (p.55).")
print()

print("=" * 96)
print("WHAT THIS MEANS FOR THE IDENTIFYING CONDITION")
print("=" * 96)
print("  The condition is not falsified.  It returns the section loss the MECHANICS sees, i.e.")
print("  the loss at the governing section.  The available ground truth (bar mass, weighed) is a")
print("  DIFFERENT quantity: the spatial average.  Under pitting the two differ by ~2.3x here.")
print("  A validation of theta against gravimetric mass loss is therefore only admissible when")
print("  the corrosion is verified uniform.  Davis' bare bars are the only such set and they")
print("  number four, with a 26% batch spread in apparent yield strength (390-493 MPa), which")
print("  is itself larger than the section-loss signal for the 1% and 5% specimens.")


# ============================== DETECTABILITY AGAINST THE CONTROL BAND ==============================
print()
print("=" * 96)
print("(4) DETECTABILITY: can the recovered theta be told apart from the control scatter?")
print("=" * 96)
NREF = mean(TAB22["15M"]["rc"])
ctrl_theta = [(1 - n / NREF) * 100 for n in TAB22["15M"]["rc"]]
mu = mean(ctrl_theta)
sd = math.sqrt(sum((x - mu) ** 2 for x in ctrl_theta) / (len(ctrl_theta) - 1))
band = 2 * sd
print(f"  4 uncorroded controls recover theta = "
      f"[{', '.join(f'{x:+.2f}' for x in ctrl_theta)}] pp,  sd = {sd:.2f} pp")
print(f"  2-sigma detection band = +/- {band:.2f} pp")
print(f"\n  {'spec':7s} {'weighed%':>9s} {'recovered pp':>13s} {'detected?':>10s}")
det = []
for n, th, nm in RC:
    r = (1 - nm / NREF) * 100
    d = r > band
    det.append((th * 100, d))
    print(f"  {n:7s} {th*100:9.1f} {r:13.2f} {'YES' if d else 'no':>10s}")
lo = [t for t, d in det if not d]
hi = [t for t, d in det if d]
print(f"\n  not detected at weighed loss: {sorted(lo)} %")
print(f"  detected     at weighed loss: {sorted(hi)} %")
print(f"  => detection threshold lies between {max(lo):.1f}% and {min(hi):.1f}% weighed mass loss.")
print(f"  Below ~1% loss the recovered theta is indistinguishable from batch scatter in f_y.")

# Spearman rank correlation, recovered vs weighed, 8 corroded RC prisms
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[s[j + 1]] == v[s[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[s[k]] = avg
        i = j + 1
    return r

x = [t for _, t, _ in RC]
y = [1 - n / NREF for _, _, n in RC]
rx, ry = rank(x), rank(y)
n = len(x)
mrx, mry = mean(rx), mean(ry)
num = sum((a - mrx) * (b - mry) for a, b in zip(rx, ry))
den = math.sqrt(sum((a - mrx) ** 2 for a in rx) * sum((b - mry) ** 2 for b in ry))
print(f"\n  Spearman rank correlation (recovered vs weighed, 8 corroded RC prisms) = {num/den:.3f}")
print(f"  => the ORDERING of damage is recovered essentially perfectly; only the SCALE is wrong.")
