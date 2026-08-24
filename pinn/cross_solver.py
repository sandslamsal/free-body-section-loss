"""What model mismatch costs, as against material mismatch.

This study reads a section loss out of measured strain by reconciling a
band couple against statics, and every accuracy figure quoted for it so far
was obtained on fields the study's own solver produced. Identification and
generator therefore share the constitutive map, the element, the
discretization and the equilibrium path, which makes the figure a statement
about self-consistency before it is one about a structure. `model_error.py`
prices part of that by moving a material constant inside the shared solver,
but the element, the cracking formulation and the path stay common, so it
prices the constants and not the model.

This module supplies the missing line. It runs the identification, entirely
unchanged, on four families of fields:

  same solver     constant-strain triangles, rotating cracked membrane, the
                  family the study already reports (fields_theta.npz)
  new element     the same rotating cracked membrane on four-node
                  quadrilaterals (q4_fields_theta.npz)
  new model       the fixed smeared-crack quadrilateral of
                  fixed_crack_oracle.py at two discretizations, which
                  disagrees with the Compatible Stress Field Method about
                  crack rotation, concrete tension, the compressive law,
                  what drives compression softening, shear across a crack
                  and the element (fields_crossmodel.npz)

The identifier is the same function in every case and reads the same 50 mm
measurement grid, so the differences between rows are differences between
generators and nothing else.

Two variants accompany the unchanged result, and neither replaces it. The
first widens the trial range to negative section loss. The identification
searches non-negative values, which is right, but a model error large
enough to push the required moment above anything an intact tie can supply
then returns no root at all, and no root is not a number the error budget
can carry. The widened range turns that outcome into the signed value it
implicitly is. The second gives each family its own measured reaction
centroid instead of the 370 mm measured on the reference discretization,
because the arm is a measured quantity and holding it at another model's
value would charge a measurement error to the constitutive.

Run:  python cross_solver.py       (writes figures/cross_solver.json)
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

ORACLE = HERE.parent / "oracle"
OUT = HERE.parent / "figures" / "cross_solver.json"
BUDGET = HERE.parent / "figures" / "model_error_by_load.json"
ARM = 370.0                    # reaction centroid measured on the reference
DELTA = "3.5"                  # the station the study reports everywhere
PREPEAK = "2.0"                # a station short of the limit point of both
THETA = [0.0, 0.10, 0.20, 0.30, 0.40]
WIDE = np.linspace(-0.70, 0.70, 141)


def identify(prob, area, xy, u, lam, arm=ARM, grid=None):
    """The identification, unchanged: constant-strain kinematics on the
    measurement grid, the CSFM band couple, the root against statics."""
    cx, cy, ex, ey, gxy = element_strains(xy, u, FD.NX, FD.NY)
    root, _g, _f, m_req = FD.recover_band(prob, cx, cy, ex, ey, gxy,
                                          area, lam, arm, grid=grid)
    return root, m_req, (cx, cy, ex, ey, gxy)


def family(name, states, prob, area, arms=None):
    """Recovered value and error for one family of generated fields."""
    rows = []
    for k, (th, xy, u, lam) in enumerate(states):
        arm = ARM if arms is None else arms[k]
        root, m_req, s = identify(prob, area, xy, u, lam)
        wide, _, _ = identify(prob, area, xy, u, lam, grid=WIDE)
        own, _, _ = identify(prob, area, xy, u, lam, arm=arm, grid=WIDE)
        T, z, couple = FD.band_couple(prob, *s, area, 0.0)
        cx, cy, ex = s[0], s[1], s[2]
        strip = (cy < FD.BAND) & (np.abs(cx - FD.X_CUT) < FD.BAND_W)
        rows.append({
            "band_eps_at_cut": float(ex[strip].mean()),
            "band_eps_over_yield": float(ex[strip].mean()
                                         / (prob.mat.fy / prob.mat.Es)),
            "theta_true": th, "lam": lam, "M_req_kNm": m_req,
            "theta_rec": None if not np.isfinite(root) else float(root),
            "err_pp": None if not np.isfinite(root) else (root - th) * 100.0,
            "theta_rec_wide": None if not np.isfinite(wide) else float(wide),
            "err_pp_wide": None if not np.isfinite(wide)
            else (wide - th) * 100.0,
            "arm_own_mm": float(arm),
            "theta_rec_own_arm": None if not np.isfinite(own) else float(own),
            "err_pp_own_arm": None if not np.isfinite(own)
            else (own - th) * 100.0,
            "T_intact_kN": T, "z_mm": z, "couple_intact_kNm": couple,
            "moment_headroom": couple / m_req})
    def rng(key):
        v = [r[key] for r in rows if r[key] is not None]
        return [float(min(v)), float(max(v))] if v else None
    e = [r["err_pp_wide"] for r in rows if r["err_pp_wide"] is not None]
    ok = [(r["theta_true"], r["theta_rec_wide"]) for r in rows
          if r["theta_rec_wide"] is not None]
    fit = None
    if len(ok) >= 2:
        s_, i_ = np.polyfit(np.array([a for a, _ in ok]),
                            np.array([b for _, b in ok]), 1)
        fit = {"slope": float(s_), "intercept": float(i_)}
    return {"name": name, "rows": rows, "affine_wide": fit,
            "n_admissible": sum(1 for r in rows if r["theta_rec"] is not None),
            "n_states": len(rows),
            "err_range_pp": rng("err_pp"),
            "err_range_pp_wide": rng("err_pp_wide"),
            "err_range_pp_own_arm": rng("err_pp_own_arm"),
            "mean_abs_err_pp_wide": float(np.mean(np.abs(e))) if e else None}


def load_same_solver(delta=DELTA):
    """The reference family, and the reaction centroid each of its states
    actually has. The identification uses a fixed 370 mm, measured once on
    this discretization; the per-state value is carried so the variant that
    lets every family measure its own arm has one to use here too."""
    from arclength_oracle import build_mesh, assemble                      # noqa: E402
    from oracle_rho_sweep import deepbeam_rho, RHO_NOM                     # noqa: E402
    d = np.load(ORACLE / "fields_theta.npz")
    st, arms = [], []
    for th in THETA:
        u = d[f"u_{th:.2f}_{delta}"]
        st.append((th, d["xy"], u, float(d[f"lam_{th:.2f}_{delta}"][0])))
        prob = deepbeam_rho(RHO_NOM * (1.0 - th))
        mesh = build_mesh(prob)
        _, f_int = assemble(u.ravel(), prob, mesh)
        n = [i for i in range(mesh.n_node)
             if mesh.fixed[2 * i + 1] and mesh.xy[i, 0] < 600.0]
        r = np.array([f_int[2 * i + 1] for i in n])
        arms.append(float((mesh.xy[n, 0] * r).sum() / r.sum()))
    return st, arms


def load_q4():
    """The CSFM on quadrilaterals. Its nodes are the measurement grid, so the
    displacement vector is reshaped and read straight off."""
    import q4_oracle as Q                                                  # noqa: E402
    from arclength_oracle import Material                                  # noqa: E402
    d = np.load(ORACLE / "q4_fields_theta.npz")
    mesh = Q.build_q4(2000.0, 1000.0, 40, 20, 250.0, 200.0, 800.0e3)
    mat = Material(fc=30.0)
    st, arms = [], []
    for th in THETA:
        u = d[f"u_{th:.2f}_{DELTA}"]
        st.append((th, d["xy"], u.reshape(-1, 2),
                   float(d[f"lam_{th:.2f}_{DELTA}"][0])))
        f_int, _ = Q.assemble(u, mesh, 0.012 * (1.0 - th), mat, 300.0)
        n = [i for i in range(mesh.xy.shape[0])
             if mesh.fixed[2 * i + 1] and mesh.xy[i, 0] < 600.0]
        r = np.array([f_int[2 * i + 1] for i in n])
        arms.append(float((mesh.xy[n, 0] * r).sum() / r.sum()))
    return st, arms


def load_cross(tag, delta=DELTA, fname="fields_crossmodel.npz"):
    """The alternative-model family, with the checks that say whether its
    fields may be used: sectional closure across the cut, global reaction
    balance, and where the station sits on the model's own curve."""
    d = np.load(ORACLE / fname)
    st = [(th, d["xy"], d[f"u_{tag}_{th:.2f}_{delta}"],
           float(d[f"lam_{tag}_{th:.2f}_{delta}"][0])) for th in THETA]
    g = lambda k, th: float(d[f"chk_{k}_{tag}_{th:.2f}_{delta}"][0])  # noqa: E731
    arms = [g("x_reaction_mm", th) for th in THETA]
    checks = []
    for th in THETA:
        c = d[f"curve_{tag}_{th:.2f}_{delta}"]
        # The peak is read off a smoothed curve. A few stations along the
        # path stall short of the force tolerance and leave a spike of a few
        # per cent, and a raw argmax lands on one of those rather than on the
        # limit point: at theta = 0.40 it reported a peak at 1.8 mm on a
        # curve that is still climbing at 3.2 mm.
        k = 25
        lam_s = np.convolve(c[:, 1], np.ones(k) / k, mode="same")
        lam_s[:k], lam_s[-k:] = c[:k, 1], c[-k:, 1]
        ipk = int(np.argmax(lam_s))
        checks.append({
            "theta": th,
            "M_closure": g("M_cut_kNm", th) / g("M_statics_kNm", th),
            "V_closure": g("V_cut_kN", th) / g("V_statics_kN", th),
            "N_cut_kN": g("N_cut_kN", th),
            "global_closure": g("global_closure", th),
            "x_reaction_mm": g("x_reaction_mm", th),
            "peak_lam": float(lam_s[ipk]), "peak_delta_mm": float(c[ipk, 0]),
            "lam_over_peak": float(c[-1, 1] / lam_s[ipk]),
            "post_peak": bool(c[ipk, 0] < float(delta) - 0.05),
            "unconverged_steps": [int(v) for v in
                                  d[f"unconv_{tag}_{th:.2f}_{delta}"]]})
    return st, arms, checks


def table(fam):
    print(f"\n{fam['name']}")
    print(f"{'theta':>7}{'lambda':>9}{'M req':>9}{'couple/M':>10}"
          f"{'rec':>9}{'err pp':>9}{'rec wide':>10}{'err pp':>9}"
          f"{'arm':>7}{'err pp':>9}")
    for r in fam["rows"]:
        f2 = lambda v, p=4: "none" if v is None else f"{v:.{p}f}"   # noqa: E731
        f3 = lambda v: "  --" if v is None else f"{v:+.2f}"         # noqa: E731
        print(f"{r['theta_true']:>7.2f}{r['lam']:>9.4f}{r['M_req_kNm']:>9.1f}"
              f"{r['moment_headroom']:>10.3f}"
              f"{f2(r['theta_rec']):>9}{f3(r['err_pp']):>9}"
              f"{f2(r['theta_rec_wide']):>10}{f3(r['err_pp_wide']):>9}"
              f"{r['arm_own_mm']:>7.0f}{f3(r['err_pp_own_arm']):>9}")
    print(f"        admissible {fam['n_admissible']}/{fam['n_states']}    "
          f"wide-range error {fam['err_range_pp_wide'][0]:+.1f} to "
          f"{fam['err_range_pp_wide'][1]:+.1f} pp")


def main() -> None:
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    st, arms = load_same_solver()
    fams = [family("same solver: CST, rotating cracked membrane",
                   st, prob, area, arms=arms)]
    st, arms = load_q4()
    fams.append(family("same constitutive, new element: Q4 40x20",
                       st, prob, area, arms=arms))
    checks = {}
    for tag in ("40x20", "80x40"):
        st, arms, ck = load_cross(tag)
        checks[f"{tag} at {DELTA} mm"] = ck
        fams.append(family(f"new model: fixed crack, Q4 {tag}",
                           st, prob, area, arms=arms))

    # The alternative model reaches its limit point before the 3.5 mm
    # station, so every field above is on its descending branch while the
    # reference is not. This pair repeats the comparison at a station short
    # of both limit points, which is the only version of it in which the
    # model form is the sole difference.
    st, arms = load_same_solver(PREPEAK)
    pre = [family(f"same solver at {PREPEAK} mm", st, prob, area, arms=arms)]
    st, arms, ck = load_cross("40x20", PREPEAK, "fields_crossmodel_prepeak.npz")
    checks[f"40x20 at {PREPEAK} mm"] = ck
    pre.append(family(f"new model at {PREPEAK} mm: fixed crack, Q4 40x20",
                      st, prob, area, arms=arms))

    for f in fams + pre:
        table(f)

    print("\nadmissibility of the alternative fields, on their own stresses")
    print(f"{'family':>18}{'theta':>7}{'M close':>9}{'V close':>9}"
          f"{'N cut kN':>10}{'reactions':>11}{'x_R mm':>8}"
          f"{'peak lam':>10}{'at mm':>8}{'lam/peak':>11}")
    for tag, ck in checks.items():
        for c in ck:
            print(f"{tag:>18}{c['theta']:>7.2f}{c['M_closure']:>9.3f}"
                  f"{c['V_closure']:>9.3f}{c['N_cut_kN']:>10.1f}"
                  f"{c['global_closure']:>11.4f}{c['x_reaction_mm']:>8.0f}"
                  f"{c['peak_lam']:>10.3f}{c['peak_delta_mm']:>8.2f}"
                  f"{c['lam_over_peak']:>11.3f}")

    print("\nthe strain the identification integrates, at the cut")
    print(f"{'family':>46}{'band eps':>11}{'/ yield':>9}")
    for f in fams + pre:
        e = np.mean([r["band_eps_at_cut"] for r in f["rows"]])
        y = np.mean([r["band_eps_over_yield"] for r in f["rows"]])
        print(f"{f['name']:>46}{e * 1e3:>10.2f}e-3{y:>9.2f}")

    # ---- the budget, four lines --------------------------------------
    def shifts_against(ref_fam, others):
        base = {r["theta_true"]: r["theta_rec_wide"] for r in ref_fam["rows"]}
        out = {}
        for f in others:
            v = [abs(r["theta_rec_wide"] - base[r["theta_true"]]) * 100.0
                 for r in f["rows"]
                 if r["theta_rec_wide"] is not None
                 and base.get(r["theta_true"]) is not None]
            out[f["name"]] = {"mean_abs_shift_pp": float(np.mean(v)),
                              "max_abs_shift_pp": float(np.max(v)),
                              "n": len(v)}
        return out

    shifts = shifts_against(fams[0], fams[1:])
    shifts.update(shifts_against(pre[0], pre[1:]))
    mb = json.loads(BUDGET.read_text())["family_shift_pp"]["3.5"]
    b0 = fams[0]["err_range_pp"]

    lines = [("1. same solver, no noise: bias against truth",
              f"{b0[0]:+.1f} to {b0[1]:+.1f}"),
             ("2. material perturbation: f_c, f_y, E_s",
              f"{min(mb.values()):.1f} to {max(mb.values()):.1f}")]
    for label, f in [("3. element only: Q4, same constitutive", fams[1]),
                     ("4. model form at 3.5 mm, both at their limit point",
                      fams[2]),
                     ("   the same on 80x40, which is well past its own",
                      fams[3]),
                     ("4. model form at 2.0 mm, both still climbing",
                      pre[1])]:
        sh = shifts[f["name"]]
        lines.append((label, f"{sh['mean_abs_shift_pp']:.1f} mean, "
                             f"{sh['max_abs_shift_pp']:.1f} max"))
    print("\n" + "=" * 74)
    print("error budget, in points of section loss\n")
    print(f"{'source':<48}{'cost (pp)':>24}")
    for a, b in lines:
        print(f"{a:<48}{b:>24}")
    print("\nlines 3 and 4 are shifts against the same-solver recovery at the "
          "same station,\nnot against truth, so they price the change of "
          "generator alone. The states\nadmitting no root at all are counted "
          "through the widened trial range.")

    out = {"what": "cost of model mismatch, against material mismatch, in "
                   "points of section loss",
           "delta_mm": float(DELTA), "prepeak_delta_mm": float(PREPEAK),
           "arm_mm": ARM, "x_cut_mm": FD.X_CUT, "band_mm": FD.BAND,
           "identifier": "figdata.recover_band, unchanged",
           "generator": "oracle/fixed_crack_oracle.py, fixed smeared crack "
                        "on Q4, 2x2 Gauss",
           "shared_with_identifier": [
               "geometry, supports, bearing widths and load position",
               "smeared reinforcement as a volume fraction with perfect bond",
               "the bilinear bare-bar steel law and all its constants",
               "the concrete material constants f_c, E_c and eps_c2; only "
               "the form of the concrete laws differs",
               "plane stress, small strain, no Poisson coupling",
               "displacement control at the load patch and the load factor "
               "read from the patch reaction",
               "a Levenberg-Marquardt damped Newton driver",
               "the 50 mm measurement grid and the constant-strain "
               "kinematics the identifier forms from it",
               "the identification itself: free body, cut at x = 700 mm, "
               "100 mm strip, band couple, reaction arm",
               "NumPy and SciPy in one repository, written by one author"],
           "broken": [
               "crack orientation: rotating principal frame against a frame "
               "frozen at first cracking",
               "concrete tension: neglected against f_ct with exponential "
               "softening regularized on G_f over a crack band",
               "cracking criterion: none against Rankine",
               "compressive law: parabola-rectangle plateau against "
               "Hognestad with a descending branch and a residual plateau",
               "compression softening: k_c2(eps_1) against Vecchio and "
               "Collins 1993 Model B in the principal strain ratio",
               "shear on a crack: implied by frame rotation against an "
               "explicit retention factor",
               "element and quadrature: CST at one point against Q4 at 2x2",
               "discretization: 40x20 against 40x20 and 80x40",
               "path: no memory against frozen crack directions",
               "linearisation: analytic tangent against finite differences"],
           "wide_grid": [float(WIDE[0]), float(WIDE[-1]), int(WIDE.size)],
           "families": fams, "families_prepeak": pre,
           "shift_vs_same_solver": shifts,
           "material_budget_3.5mm_pp": mb,
           "field_checks": checks}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
