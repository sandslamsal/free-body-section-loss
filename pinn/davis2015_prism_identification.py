"""
Davis (2015) MASc thesis, Queen's University -- external check of the statics-on-a-free-body
identification of tie section loss, run on REAL corroded reinforced concrete tension prisms
with WEIGHED (gravimetric) section loss.

WHY THIS SOURCE:  on an axially loaded tension prism the identifying condition
    T(theta) = T_c + A_s (1 - theta) sigma_s(eps)  ==  T_req
collapses to the exact statement  N_applied = T_c + A_s (1 - theta) sigma_s(eps),
because the free body is a plain cut: no lever arm, no reaction position, no compression
centroid.  Every geometric confound of the deep-beam free body disappears.

WHAT THE SOURCE ACTUALLY CONTAINS:  the thesis has FIVE tables (2.1, 2.2, 3.1, 3.2, 4.1).
NONE of them tabulates strain.  All distributed fiber-optic strain lives in figures only.
So the strong identification test (T1) using measured strain at load stages CANNOT be run
from tabulated data.  What CAN be run is the identifying condition evaluated at the one
stress state the thesis does tabulate -- yielding, where sigma_s = f_y is known:

    N_yield = T_c + f_y A_s (1 - theta)      =>   theta_hat = 1 - (N_yield - T_c) / (f_y A_s)

That is the same affine, monotone-decreasing condition, inverted in closed form (the
bisection root of the affine map, done analytically because the map is linear here).

Run with /usr/local/bin/python3.12
"""
import csv, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

A_S   = 200.0        # mm^2, nominal 15M
E_S   = 191_000.0    # MPa, Davis p.19, measured from rebar tensile tests
FY_NOM   = 400.0     # MPa, the thesis' own assumed value for its "theoretical yield load"
FY_MEAS  = 447.5     # MPa, thesis p.63 "average yield stress found the control bars"
FY_STUDY = 500.0     # MPa, this study's default constitutive value


def load():
    rows = []
    with open(os.path.join(DATA, "davis2015_tables.csv")) as f:
        for line in f:
            if line.startswith("#") or line.startswith("table,"):
                continue
            p = [x.strip() for x in line.rstrip("\n").split(",")]
            rows.append(dict(
                table=p[0], name=p[1], set=p[2],
                mass_loss=float(p[5]) / 100.0,
                N_exp=float(p[6]), N_y=float(p[7]), notes=p[8]))
    return rows


def linreg(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    slope = sxy / sxx
    icept = my - slope * mx
    r2 = (sxy ** 2) / (sxx * syy) if syy > 0 else float("nan")
    # standard error of slope
    resid = [b - (icept + slope * a) for a, b in zip(x, y)]
    s2 = sum(r ** 2 for r in resid) / (n - 2)
    se = math.sqrt(s2 / sxx)
    return slope, icept, r2, se


def recover(N_y_kN, N_ref_kN):
    """Affine identifying condition inverted.  N_ref = f_y*A_s + T_c is the theta=0 datum."""
    return 1.0 - N_y_kN / N_ref_kN


def stats(errs):
    n = len(errs)
    m = sum(errs) / n
    sd = math.sqrt(sum((e - m) ** 2 for e in errs) / (n - 1)) if n > 1 else float("nan")
    return m, sd, max(abs(e) for e in errs), sum(abs(e) for e in errs) / n


rows = load()
rc   = [r for r in rows if r["set"] == "rc"]
bare = [r for r in rows if r["set"] == "bare"]

rc_ctrl   = [r for r in rc   if r["mass_loss"] == 0]
bare_ctrl = [r for r in bare if r["mass_loss"] == 0]
N_ref_rc   = sum(r["N_y"] for r in rc_ctrl)   / len(rc_ctrl)
N_ref_bare = sum(r["N_y"] for r in bare_ctrl) / len(bare_ctrl)

print("=" * 96)
print("DAVIS (2015) -- IDENTIFYING CONDITION EVALUATED AT YIELD, REAL CORRODED BARS")
print("=" * 96)
print(f"15M bar, A_s = {A_S:.0f} mm^2, E_s = {E_S/1000:.0f} GPa (thesis p.19)")
print(f"prism 100 x 100 x 900 mm, f'c = 33.6 MPa, gravimetric section loss (weighed before/after)")
print()
print("theta=0 DATUM (self-referenced to the study's own uncorroded controls):")
print(f"  RC prisms   N_ref = mean(C1-C,C2-C,C3-C,C4-C) = "
      f"({', '.join(f'{r[chr(34)+chr(34)] if False else r[str()] if False else r[chr(78)+chr(95)+chr(121)]:.1f}' for r in rc_ctrl)}) "
      f"-> {N_ref_rc:.2f} kN  => implied f_y+T_c = {N_ref_rc*1000/A_S:.1f} MPa-equivalent")
print(f"  bare bars   N_ref = mean(R1-C,R2-C) = "
      f"({', '.join(f'{r[chr(78)+chr(95)+chr(121)]:.1f}' for r in bare_ctrl)}) "
      f"-> {N_ref_bare:.2f} kN  => implied f_y = {N_ref_bare*1000/A_S:.1f} MPa")
print()

results = {}
for label, grp, N_ref in (("RC TENSION PRISMS (Table 3.2)", rc, N_ref_rc),
                          ("BARE BARS (Table 3.1)", bare, N_ref_bare)):
    print("-" * 96)
    print(label + f"   [theta=0 datum N_ref = {N_ref:.2f} kN]")
    print("-" * 96)
    print(f"{'spec':7s} {'N_y(kN)':>8s} {'weighed%':>9s} {'recovered%':>11s} {'err(pp)':>9s} "
          f"{'f_y,app on net section (MPa)':>30s}   note")
    xs, ys, errs = [], [], []
    for r in grp:
        th_hat = recover(r["N_y"], N_ref) * 100.0
        th_true = r["mass_loss"] * 100.0
        err = th_hat - th_true
        fy_app = r["N_y"] * 1000.0 / (A_S * (1 - r["mass_loss"]))
        xs.append(th_true); ys.append(th_hat); errs.append(err)
        print(f"{r['name']:7s} {r['N_y']:8.1f} {th_true:9.1f} {th_hat:11.2f} {err:+9.2f} "
              f"{fy_app:30.1f}   {r['notes']}")
    m, sd, mx, mae = stats(errs)
    sl, ic, r2, se = linreg(xs, ys)
    print(f"\n  signed bias (mean err)      = {m:+.2f} pp"
          f"   [positive = OVER-reports damage]")
    print(f"  sd of err                   = {sd:.2f} pp")
    print(f"  MAE                         = {mae:.2f} pp     max |err| = {mx:.2f} pp")
    print(f"  regression recovered = a + b*weighed :  b = {sl:.3f} +/- {se:.3f},  "
          f"a = {ic:+.2f} pp,  R^2 = {r2:.3f}")
    print(f"  one-for-one tracking?  slope {sl:.3f} vs 1.000 -> "
          f"{'YES within 1 se' if abs(sl-1) < se else 'NO, ' + f'{abs(sl-1)/se:.1f} se from unity'}")
    # corroded-only
    cor = [r for r in grp if r["mass_loss"] > 0]
    ec = [recover(r["N_y"], N_ref) * 100 - r["mass_loss"] * 100 for r in cor]
    mc, sdc, mxc, maec = stats(ec)
    print(f"  corroded specimens only (n={len(cor)}): bias {mc:+.2f} pp, sd {sdc:.2f} pp, "
          f"MAE {maec:.2f} pp, max {mxc:.2f} pp")
    results[label] = dict(bias=m, sd=sd, mae=mae, mx=mx, slope=sl, se=se, icept=ic, r2=r2,
                          xs=xs, ys=ys, N_ref=N_ref)
    print()

# ---------------------------------------------------------------- control-only noise floor
print("=" * 96)
print("IRREDUCIBLE NOISE FLOOR -- what the four UNCORRODED RC controls alone recover")
print("=" * 96)
for r in rc_ctrl:
    print(f"  {r['name']:6s} N_y={r['N_y']:5.1f} kN -> theta_hat = {recover(r['N_y'], N_ref_rc)*100:+6.2f} pp "
          f"(truth 0.00)")
ctrl_err = [recover(r["N_y"], N_ref_rc) * 100 for r in rc_ctrl]
print(f"  spread of controls: {min(ctrl_err):+.2f} to {max(ctrl_err):+.2f} pp, "
      f"sd = {stats(ctrl_err)[1]:.2f} pp")
print(f"  => no recovery from a single prism can be trusted to better than about "
      f"+/-{max(abs(e) for e in ctrl_err):.1f} pp,")
print(f"     because nominally identical uncorroded bars yield {min(r['N_y'] for r in rc_ctrl):.1f}-"
      f"{max(r['N_y'] for r in rc_ctrl):.1f} kN "
      f"({(max(r['N_y'] for r in rc_ctrl)/min(r['N_y'] for r in rc_ctrl)-1)*100:.1f}% spread).")
print()

# ---------------------------------------------------------------- f_y sensitivity (absolute form)
print("=" * 96)
print("SENSITIVITY TO THE ASSUMED f_y  (absolute form, T_c = 0, no self-referencing)")
print("=" * 96)
print(f"{'assumed f_y':>26s} {'f_y*A_s(kN)':>12s} {'theta_hat of an UNCORRODED':>28s} {'signed bias on':>16s}")
print(f"{'':>26s} {'':>12s} {'RC control (pp)':>28s} {'8 corroded (pp)':>16s}")
for lab, fy in (("400 MPa (thesis nominal)", FY_NOM),
                ("447.5 MPa (thesis measured)", FY_MEAS),
                ("500 MPa (this study default)", FY_STUDY)):
    Nref = fy * A_S / 1000.0
    ctrl = recover(N_ref_rc, Nref) * 100
    cor = [r for r in rc if r["mass_loss"] > 0]
    b = sum(recover(r["N_y"], Nref) * 100 - r["mass_loss"] * 100 for r in cor) / len(cor)
    print(f"{lab:>26s} {Nref:12.1f} {ctrl:+28.2f} {b:+16.2f}")
print()
print("  Reading: with f_y ASSUMED BELOW the true value the condition UNDER-reports damage")
print("  (negative theta on an intact bar); with f_y assumed above it OVER-reports.  The signed")
print("  bias of the method here is set by the f_y datum, not by the statics.")
print()

# ---------------------------------------------------------------- pitting diagnosis
print("=" * 96)
print("WHY THE RC PRISMS OVER-REPORT: pitting makes MINIMUM section, not MEAN mass loss, govern")
print("=" * 96)
print(f"{'spec':7s} {'weighed%':>9s} {'f_y,app on weighed net section (MPa)':>38s}")
for r in rc:
    fy_app = r["N_y"] * 1000.0 / (A_S * (1 - r["mass_loss"]))
    print(f"{r['name']:7s} {r['mass_loss']*100:9.1f} {fy_app:38.1f}")
cor = [r for r in rc if r["mass_loss"] > 0]
fy_c = [r["N_y"] * 1000 / (A_S * (1 - r["mass_loss"])) for r in rc_ctrl]
fy_x = [r["N_y"] * 1000 / (A_S * (1 - r["mass_loss"])) for r in cor]
print(f"\n  controls  mean apparent f_y = {sum(fy_c)/len(fy_c):.1f} MPa")
print(f"  corroded  mean apparent f_y = {sum(fy_x)/len(fy_x):.1f} MPa")
print(f"  If corrosion were UNIFORM these would be equal.  They are not: the corroded prisms lose")
print(f"  {(1 - (sum(fy_x)/len(fy_x))/(sum(fy_c)/len(fy_c)))*100:.1f}% of apparent net-section strength that mass loss does not explain.")
print(f"  Thesis p.57: 'the majority of the reinforcement deterioration in the reinforced concrete")
print(f"  specimens was pitting corrosion ... caused a greater reduction in the rebar cross section")
print(f"  in certain locations than was seen for the bare rebar specimens.'")
print()

# ---------------------------------------------------------------- one T1-style spot check
print("=" * 96)
print("SINGLE T1-STYLE SPOT CHECK (author-quoted approximate figure value, NOT a table number)")
print("=" * 96)
N = 10_000.0   # N, applied load
eps = 500e-6   # thesis p.65: C2-10 nylon-fiber strain spike 'reaches approximately 500 microstrain'
sig = E_S * eps
A_req = N / sig
print(f"  C2-10 (weighed mean mass loss 13.0%), applied N = {N/1000:.0f} kN, measured local eps ~ {eps*1e6:.0f} ue")
print(f"  identifying condition with T_c = 0 at the pit:  A(1-theta) = N / (E_s eps) = {A_req:.1f} mm^2")
print(f"  -> theta_local = {(1 - A_req/A_S)*100:.1f} %   vs weighed MEAN mass loss 13.0 %")
print(f"  The condition is satisfied exactly; it returns the LOCAL section loss at the pit, which is")
print(f"  {(1-A_req/A_S)*100/13.0:.1f}x the gravimetric average.  Flagged as illustrative only: the 500 ue is the")
print(f"  thesis' own prose reading of its Figure 3.16, not a tabulated value.")
print()

# ---------------------------------------------------------------- beams
print("=" * 96)
print("TABLE 4.1 BEAMS -- NOT USABLE")
print("=" * 96)
print("  BS-01 1.1% loss -> -13% capacity;  BL-01 1.4% loss -> -38%;  BS-05 4.0% -> -31%;")
print("  BL-05 3.8% -> -37%.  BL-01 (less corroded) lost MORE capacity than BL-05.  Thesis p.100:")
print("  'end debonding due to the protective sleeves plays an important role'.  The failure mode")
print("  changed to anchorage, so capacity no longer measures tie force on a flexural free body.")
print()

# ---------------------------------------------------------------- write results
out = os.path.join(DATA, "davis2015_recovery.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["set", "specimen", "N_yield_kN", "weighed_mass_loss_pct",
                "recovered_theta_pct", "error_pp", "apparent_fy_on_net_section_MPa"])
    for label, grp, N_ref in (("rc", rc, N_ref_rc), ("bare", bare, N_ref_bare)):
        for r in grp:
            th = recover(r["N_y"], N_ref) * 100
            w.writerow([label, r["name"], f"{r['N_y']:.1f}", f"{r['mass_loss']*100:.1f}",
                        f"{th:.2f}", f"{th - r['mass_loss']*100:+.2f}",
                        f"{r['N_y']*1000/(A_S*(1-r['mass_loss'])):.1f}"])
print(f"wrote {out}")
