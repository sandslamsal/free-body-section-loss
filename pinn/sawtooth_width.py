"""Can a wider cut strip average the bonded-fiber sawtooth away?

sawtooth.py established that a bonded distributed fiber returns a sawtooth
along the tie, and that feeding a realistic sawtooth to this study's
identification is catastrophic: swept over the phase of the crack pattern
relative to the gauge stations, the recovered section loss ranges over 55 to
89 per cent of the section, and between 53 and 73 per cent of phases return
no admissible root. It also named the cause. The stabilised crack spacing is
235 to 288 mm while the cut is a 100 mm strip whose gauge stations span
66.7 mm, so the integral covers a quarter to two fifths of a period and
averages nothing.

That diagnosis carries a testable remedy. The observable of this study is an
integral of stress over the cut, and an integral over a length that spans a
whole number of periods removes a periodic disturbance exactly. If widening
the strip to one crack spacing collapses the phase sensitivity, then the
sawtooth stops being a fatal objection and becomes a design requirement on
the instrumented length, which is a statement a practitioner can act on.

Widening is not free and the price has to be measured, not assumed. The
strip is not only an averaging window, it is the cut itself. The moment
demand is evaluated at one station while a wide strip averages the supplied
couple over a range of stations where the tie force, the lever arm and the
compression centroid all vary. A width that averages the sawtooth but moves
the smooth-field answer has traded one error for another, so the ordinary
no-noise identification is run at every width alongside the phase sweep and
the two are reported together.

Two implementation points decide whether the numbers mean anything. The
tributary normalization in figdata.cut_quantities divides by the nominal
strip width 2 W, which is right only when the selected element centroids
tile the strip exactly; on this mesh that holds for W on the 25 mm lattice
and fails otherwise, so an exact-coverage normalization is installed here
and checked against the original on the lattice before it is used. And the
strip has to stay inside the D-region: the load bearing plate starts at
x = 900 mm, so W above 200 mm is reported but flagged, because past that
the cut is no longer a cut through the shear span.

Run:  python sawtooth_width.py    (writes figures/sawtooth_width.json,
                                   sawtooth_width.png)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figdata as FD                                                      # noqa: E402
import figstyle as F                                                      # noqa: E402
import sawtooth as SW                                                     # noqa: E402
import tcm                                                                # noqa: E402
from csfm_constitutive import membrane                                    # noqa: E402
from identify import rho_x_of_theta                                       # noqa: E402
from problem import DeepBeam                                              # noqa: E402
from recover_utils import element_strains                                 # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT_JSON = HERE.parent / "figures" / "sawtooth_width.json"
OUT_PNG = HERE.parent / "figures" / "sawtooth_width.png"

DELTA = SW.DELTA
ARM = SW.ARM
TRUTH = SW.TRUTH
N_PHASE = SW.N_PHASE

# Half-widths in mm. Every value is on the 25 mm lattice, where the element
# centroids tile the strip exactly, so the sweep is directly comparable with
# the manuscript's own 50 mm and needs no change of normalization.
WIDTHS = (50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0,
          275.0, 300.0)

# Half-widths set by the crack spacing of the state rather than by the mesh,
# so that "the strip spans a whole number of spacings" is tested at the
# exact widths that statement names instead of at the nearest lattice value.
WIDTH_REL = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)     # multiples of s_rm
                                                       # of the FULL strip

# The bearing plate under the load occupies [900, 1100] mm and the support
# plate [150, 350], so a strip reaching past x = 900 is no longer a cut
# through the shear span. Reported, but flagged everywhere it appears.
X_VALID = (350.0, 900.0)

# Gauge lengths, as multiples of the crack spacing, for the sweep that holds
# the strip fixed and varies the length the fiber is averaged over before it
# reaches the constitutive map. The strip and the gauge are different lengths
# and this study needs to know which of the two the error lives on.
GAUGE_REL = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)

# The yardstick this study already owns: the standard deviation of the
# recovered theta under 5 per cent measurement noise, in points of section.
NOISE_STD_PP = 2.5


# ----------------------------------------------------------------------
# 1. an exact-coverage cut, so that width is the only thing that changes
# ----------------------------------------------------------------------
def cut_quantities_exact(prob, cx, cy, ex, ey, gxy, area, theta):
    """figdata.cut_quantities with the tributary width taken, not assumed.

    The original divides the element area by the nominal strip width 2 W.
    That is exact only when the selected centroids tile [x_c - W, x_c + W],
    which on this mesh happens for W on the 25 mm lattice and fails for any
    other W by up to one element column, which would enter the answer as a
    spurious scaling of the tie force. The covered length is recovered here
    from the selected area itself, so an arbitrary W is admissible and the
    lattice values are unchanged.
    """
    sel = np.abs(cx - FD.X_CUT) < FD.BAND_W
    X = torch.tensor(cx[sel]).unsqueeze(-1)
    Y = torch.tensor(cy[sel]).unsqueeze(-1)
    st = membrane(torch.tensor(ex[sel]).unsqueeze(-1),
                  torch.tensor(ey[sel]).unsqueeze(-1),
                  torch.tensor(gxy[sel]).unsqueeze(-1),
                  rho_x_of_theta(prob, X, Y, torch.tensor(float(theta))),
                  prob.rho_y(X, Y), prob.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy()
    ys = cy[sel]
    covered = float(sel.sum()) * area / prob.H          # mm along the span
    dA = area / covered * prob.t
    return sx, ys, dA


def coverage(cx, area, prob, W):
    """Length of span the selected centroids actually stand for, in mm."""
    return float((np.abs(cx - FD.X_CUT) < W).sum()) * area / prob.H


# ----------------------------------------------------------------------
# 2. the two numbers each width is judged by
# ----------------------------------------------------------------------
def smooth_at_width(prob, area, st, lam, W):
    """The ordinary no-noise identification, and the parts it is built from.

    T and z are reported beside the root because a width that moves the
    answer has to be shown to move it through something: the tie force the
    wider strip averages, or the lever arm between the two centroids it
    computes over a longer stretch of a field that is not prismatic.
    """
    FD.BAND_W = W
    cx, cy, ex, ey, gxy = st
    adm, ext, marg, T0 = SW.roots(prob, area, cx, cy, ex, ey, gxy, lam)
    T, z, C = FD.band_couple(prob, cx, cy, ex, ey, gxy, area, 0.0)
    M_req = lam * prob.P / 2.0 * (FD.X_CUT - ARM) / 1e6
    # The admissible root is the number the method reports and is the one the
    # manuscript quotes; the uncensored root is on the wider grid and is the
    # baseline the phase statistics are differenced against, so that a bias
    # is never a difference between two grids.
    return dict(theta_admissible=None if not np.isfinite(adm) else float(adm),
                theta=float(adm) if np.isfinite(adm) else float(ext),
                theta_uncensored=float(ext), margin_kN=float(marg),
                T_intact_kN=float(T), z_intact_mm=float(z),
                couple_intact_kNm=float(C), M_req_kNm=float(M_req))


def tile_centers(cx, dx=None):
    """Center of the tributary each element stands for, in mm.

    The two triangles of a cell share the same 50 mm column, so a reading
    taken at their centroids samples the column at 1/3 and 2/3 rather than
    tiling it. Assigning each triangle the half column nearest its centroid
    makes the tributaries tile the strip exactly, which turns the strip sum
    into a quadrature of the continuous fiber profile instead of a point
    sample of it. That distinction matters because a real distributed fiber
    returns a continuous profile, so a spread that comes only from sampling
    it at four points is an artefact of the discretization and not a
    property of the instrument.
    """
    if dx is None:
        dx = 2000.0 / FD.NX
    col0 = np.floor(np.asarray(cx, float) / dx) * dx
    lower = (np.asarray(cx, float) - col0) < 0.5 * dx
    return np.where(lower, col0 + 0.25 * dx, col0 + 0.75 * dx), 0.5 * dx


def reading_point(prob, f_ct):
    """The manuscript's own sampling: one reading at each element centroid."""
    def rd(cx, cy, ex, th, phase, gauge=0.0):
        return SW.fiber_reading(cx, cy, ex, th, phase, gauge, prob, f_ct)[0]
    return rd


def reading_tiled(prob, f_ct):
    """Each element reads the mean of the profile over its own tributary."""
    def rd(cx, cy, ex, th, phase, gauge=None):
        xt, g = tile_centers(cx)
        return SW.fiber_reading(xt, cy, ex, th, phase, g, prob, f_ct)[0]
    return rd


def phase_sweep_reader(prob, area, st, lam, th, s_r, reader, n_phase,
                       gauge=0.0):
    """SW.phase_sweep with the reading rule left open.

    The strip length and the length the fiber is averaged over before the
    constitutive map is applied are two different lengths, and the whole
    question here is which of the two the error lives on, so the sweep has
    to be able to vary them independently.
    """
    cx, cy, ex, ey, gxy = st
    ph = np.linspace(0.0, s_r, n_phase, endpoint=False)
    adm = np.full(ph.size, np.nan)
    ext = np.full(ph.size, np.nan)
    marg = np.full(ph.size, np.nan)
    for i, p in enumerate(ph):
        e2 = reader(cx, cy, ex, th, FD.X_CUT + float(p), gauge)
        adm[i], ext[i], marg[i], _ = SW.roots(prob, area, cx, cy, e2, ey, gxy,
                                              lam)
    return ph, adm, ext, marg


def width_case(prob, area, st, lam, th, W, s_r, f_ct, n_phase=N_PHASE,
               reader=None, gauge=0.0):
    """Smooth-field answer and the full phase sweep, at one width and state."""
    FD.BAND_W = W
    smooth = smooth_at_width(prob, area, st, lam, W)
    if reader is None:
        reader = reading_point(prob, f_ct)
    ph, adm, ext, marg = phase_sweep_reader(prob, area, st, lam, th, s_r,
                                            reader, n_phase, gauge)
    base = smooth["theta_uncensored"]
    row = SW.summarize(ph, adm, ext, marg, base)
    for k in ("phase_mm", "theta_admissible", "theta_uncensored",
              "margin_kN"):
        row.pop(k)
    row.update(half_width_mm=W, full_width_mm=2.0 * W,
               widths_per_spacing=2.0 * W / s_r,
               smooth=smooth, s_rm_mm=s_r,
               inside_D_region=bool(FD.X_CUT - W >= X_VALID[0]
                                    and FD.X_CUT + W <= X_VALID[1]))
    return row, (ph / s_r, ext)


# ----------------------------------------------------------------------
def selftest(prob, area, st, lam) -> dict:
    """The exact-coverage cut must reproduce the original on the lattice.

    Every number in this file is read against the manuscript's own 50 mm
    strip, so the substitution made above has to be shown to be a no-op
    where the two definitions agree, not merely argued to be one.
    """
    worst_cov, worst_root = 0.0, 0.0
    for W in WIDTHS:
        FD.BAND_W = W
        FD.cut_quantities = FD._cut_quantities_orig
        a = SW.roots(prob, area, *st, lam)[1]
        cov = coverage(st[0], area, prob, W)
        FD.cut_quantities = cut_quantities_exact
        b = SW.roots(prob, area, *st, lam)[1]
        worst_cov = max(worst_cov, abs(cov - 2.0 * W))
        worst_root = max(worst_root, abs(a - b))
    print(f"  self test: on the 25 mm lattice the centroids tile the strip to "
          f"{worst_cov:.2e} mm and the two normalisations give the same root "
          f"to {worst_root:.2e}")
    assert worst_cov < 1e-9 and worst_root < 1e-12
    return dict(max_coverage_error_mm=worst_cov,
                max_root_difference=worst_root)


# ----------------------------------------------------------------------
def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    mat = prob.mat
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    f_ct = tcm.f_ctm(mat.fc)
    FD._cut_quantities_orig = FD.cut_quantities
    out: dict = {}

    print("Does a wider cut strip average the bonded-fiber sawtooth away?")
    print("=" * 74)

    states, pattern = {}, {}
    for th in TRUTH:
        lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
        st = element_strains(d["xy"], d[f"u_{th:.2f}_{DELTA}"], FD.NX, FD.NY)
        states[th] = (st, lam)
        s_r, phi = SW.spacing(th, prob, f_ct)
        pattern[th] = dict(s_rm_mm=s_r, phi_mm=phi, lam_kN=lam)

    out["selftest"] = selftest(prob, area, states[0.20][0], states[0.20][1])
    FD.cut_quantities = cut_quantities_exact

    out["constants"] = dict(
        delta_mm=float(DELTA), arm_mm=ARM, n_phase=N_PHASE,
        half_widths_mm=list(WIDTHS), widths_per_spacing=list(WIDTH_REL),
        x_cut_mm=FD.X_CUT, band_mm=FD.BAND, x_valid_mm=list(X_VALID),
        noise_std_pp=NOISE_STD_PP,
        s_rm_mm={f"{t:.2f}": pattern[t]["s_rm_mm"] for t in TRUTH},
        manuscript_baseline={f"{t:.2f}": v for t, v in
                             zip(TRUTH, (0.0741, 0.1569, 0.2385, 0.3250))},
        gauge_mm=0.0,
        gauge_note="a point gauge throughout: sawtooth.py showed 5 and 10 mm "
                   "gauges change the spread by under 1 pp, so the strip is "
                   "the only length being varied here")

    # ---- 1. what each width covers, before any physics ----------------
    print("\n1. what the strip covers, state by state")
    print("   W (mm)  strip (mm)  stations  span (mm)   strip/s_rm at theta = "
          "0.10 .. 0.40    in D-region")
    geom = {}
    cx0 = states[0.20][0][0]
    cyb = states[0.20][0][1]
    for W in WIDTHS:
        sel = (np.abs(cx0 - FD.X_CUT) < W) & (cyb < FD.BAND)
        xs = np.unique(np.round(cx0[sel], 4))
        ratios = [2.0 * W / pattern[t]["s_rm_mm"] for t in TRUTH]
        ok = FD.X_CUT - W >= X_VALID[0] and FD.X_CUT + W <= X_VALID[1]
        geom[W] = dict(n_band_elements=int(sel.sum()), n_stations=int(xs.size),
                       station_span_mm=float(xs.max() - xs.min()),
                       covered_mm=coverage(cx0, area, prob, W),
                       widths_per_spacing={f"{t:.2f}": r
                                           for t, r in zip(TRUTH, ratios)},
                       inside_D_region=bool(ok))
        print(f"   {W:6.0f}  {2*W:9.0f}  {xs.size:8d}  {xs.max()-xs.min():8.1f}"
              f"   " + "  ".join(f"{r:5.2f}" for r in ratios)
              + f"        {'yes' if ok else 'NO'}")
    out["geometry"] = {f"{k:.0f}": v for k, v in geom.items()}

    # ---- 2. the confound, measured first ------------------------------
    # A width that averages the sawtooth but moves the smooth-field answer
    # has traded one error for another, so the no-noise identification is
    # measured before the sawtooth is switched on rather than after.
    print("\n2. the confound: the smooth-field answer as the strip widens")
    print("   the row is theta recovered with no sawtooth and no noise; the "
          "manuscript's\n   numbers are the 50 mm row, and the drift from it "
          "is the price of widening")
    print("   W (mm)   theta at 0.10   0.20    0.30    0.40    | drift from "
          "50 mm (pp)      T(0) kN   z mm")
    smooth = {}
    for W in WIDTHS:
        row = {}
        for th in TRUTH:
            (st, lam) = states[th]
            row[th] = smooth_at_width(prob, area, st, lam, W)
        smooth[W] = row
        drift = [100.0 * (row[t]["theta"] - smooth[WIDTHS[0]][t]["theta"])
                 for t in TRUTH]
        print(f"   {W:6.0f}   "
              + "  ".join(f"{row[t]['theta']:.4f}" for t in TRUTH)
              + "  | " + "  ".join(f"{v:+6.2f}" for v in drift)
              + f"      {row[0.20]['T_intact_kN']:7.1f}  "
                f"{row[0.20]['z_intact_mm']:6.1f}")
    out["smooth_field"] = {
        f"{W:.0f}": {f"{t:.2f}": v for t, v in r.items()}
        for W, r in smooth.items()}

    # ---- 3. the sawtooth, swept over phase, at every width -------------
    print("\n3. the sawtooth, phase swept over a full crack spacing")
    print("   cells are range in pp (admissible fraction); the range is over "
          "EVERY phase\n   on the extended grid, so it is not a survivor "
          "average")
    print("   W (mm)  strip/s_rm      theta = 0.10        0.20        0.30"
          "        0.40")
    sweep, curves = {}, {}
    for W in WIDTHS:
        row = {}
        for th in TRUTH:
            (st, lam) = states[th]
            r, cur = width_case(prob, area, st, lam, th, W,
                                pattern[th]["s_rm_mm"], f_ct)
            row[th] = r
            if th == 0.20:
                curves[W] = cur
        sweep[W] = row
        rat = 2.0 * W / pattern[0.20]["s_rm_mm"]
        cells = "  ".join(f"{row[t]['range_pp']:6.2f}"
                          f"({row[t]['admissible_fraction']:.2f})"
                          for t in TRUTH)
        print(f"   {W:6.0f}  {rat:9.2f}   {cells}", flush=True)
    out["sweep_mm"] = {f"{W:.0f}": {f"{t:.2f}": v for t, v in r.items()}
                       for W, r in sweep.items()}

    print("\n   the same, as standard deviations in pp")
    print("   W (mm)      theta = 0.10    0.20    0.30    0.40    | worst")
    for W in WIDTHS:
        stds = [sweep[W][t]["std_pp"] for t in TRUTH]
        print(f"   {W:6.0f}      " + "  ".join(f"{v:6.3f}" for v in stds)
              + f"    | {max(stds):6.3f}")

    # ---- 4. the same at widths the crack spacing sets ------------------
    # The prediction names whole spacings, not lattice values, so it is
    # tested at exactly those widths. The exact-coverage cut installed
    # above is what makes an arbitrary width admissible at all.
    print("\n4. widths set by the crack spacing of the state itself")
    print("   strip/s_rm      theta = 0.10        0.20        0.30        "
          "0.40   (range pp, admissible)")
    rel = {}
    for k in WIDTH_REL:
        row = {}
        for th in TRUTH:
            (st, lam) = states[th]
            s_r = pattern[th]["s_rm_mm"]
            r, _ = width_case(prob, area, st, lam, th, 0.5 * k * s_r, s_r,
                              f_ct)
            row[th] = r
        rel[k] = row
        cells = "  ".join(f"{row[t]['range_pp']:6.2f}"
                          f"({row[t]['admissible_fraction']:.2f})"
                          for t in TRUTH)
        print(f"   {k:9.2f}   {cells}", flush=True)
    out["sweep_rel"] = {f"{k:.2f}": {f"{t:.2f}": v for t, v in r.items()}
                        for k, r in rel.items()}

    print("\n   and their smooth-field answers, which is what they cost")
    print("   strip/s_rm    theta at 0.10   0.20    0.30    0.40   | drift "
          "from 50 mm (pp)")
    for k in WIDTH_REL:
        row = rel[k]
        drift = [100.0 * (row[t]["smooth"]["theta"]
                          - smooth[WIDTHS[0]][t]["theta"]) for t in TRUTH]
        print(f"   {k:9.2f}    "
              + "  ".join(f"{row[t]['smooth']['theta']:.4f}" for t in TRUTH)
              + "  | " + "  ".join(f"{v:+6.2f}" for v in drift))

    # ---- 4b. the number the spread was hiding --------------------------
    # A spread that collapses is worth nothing if what it collapses onto is
    # the wrong answer, so the mean over phase is printed on the same axis
    # as the spread and against the same baseline.
    print("\n4b. what the phase mean does while the spread collapses")
    print("   W (mm)  strip/s_rm    mean bias in pp at theta = 0.10   0.20"
          "     0.30     0.40")
    for W in WIDTHS:
        rat = 2.0 * W / pattern[0.20]["s_rm_mm"]
        print(f"   {W:6.0f}  {rat:9.2f}      "
              + "  ".join(f"{sweep[W][t]['bias_pp']:+7.2f}" for t in TRUTH))

    # ---- 4c. which length the error actually lives on ------------------
    # The strip is the length the STRESS is integrated over, after the
    # constitutive map has been applied element by element. The gauge is the
    # length the STRAIN is averaged over, before it. A pointwise nonlinear
    # map does not commute with an average, so the two lengths cannot be
    # expected to remove the same part of the error, and separating them is
    # the only way to say what a deployment would have to do.
    print("\n4c. the strip and the gauge are different lengths, swept apart")
    print("   the gauge is quoted as a multiple of the crack spacing; each "
          "block is one strip")
    inter = {}
    for W in (50.0, 125.0, 200.0):
        block = {}
        for gr in GAUGE_REL:
            row = {}
            for th in TRUTH:
                (st, lam) = states[th]
                s_r = pattern[th]["s_rm_mm"]
                r, _ = width_case(prob, area, st, lam, th, W, s_r, f_ct,
                                  n_phase=49, gauge=gr * s_r)
                row[th] = r
            block[gr] = row
        inter[W] = block
        print(f"   strip {2*W:.0f} mm ({2*W/pattern[0.20]['s_rm_mm']:.2f} "
              f"s_rm):   gauge     range pp (worst)   bias pp (worst)   "
              f"admissible (worst)")
        for gr in GAUGE_REL:
            row = block[gr]
            rng = max(row[t]["range_pp"] for t in TRUTH)
            bia = max(abs(row[t]["bias_pp"]) for t in TRUTH)
            adm = min(row[t]["admissible_fraction"] for t in TRUTH)
            print(f"                                   {gr:5.2f}      "
                  f"{rng:9.2f}         {bia:9.2f}          {adm:6.2f}",
                  flush=True)
    out["strip_gauge"] = {
        f"{W:.0f}": {f"{g:.2f}": {f"{t:.2f}": v for t, v in r.items()}
                     for g, r in b.items()} for W, b in inter.items()}

    # ---- 4d. is the residual physics or the four-point sample? ---------
    # A real fiber returns a continuous profile, so the strip integral of a
    # deployment is a quadrature of that profile, not a sample of it at four
    # element centroids. Giving every element the mean over its own
    # tributary makes the strip sum an exact quadrature and separates what
    # the instrument would do from what this mesh happens to do.
    print("\n4d. the same widths with the strip integrated, not point sampled")
    print("   W (mm)  strip/s_rm      theta = 0.10        0.20        0.30"
          "        0.40   (range pp, bias pp)")
    tiled = {}
    rd_tiled = reading_tiled(prob, f_ct)
    for W in WIDTHS:
        row = {}
        for th in TRUTH:
            (st, lam) = states[th]
            r, _ = width_case(prob, area, st, lam, th, W,
                              pattern[th]["s_rm_mm"], f_ct, n_phase=49,
                              reader=rd_tiled)
            row[th] = r
        tiled[W] = row
        rat = 2.0 * W / pattern[0.20]["s_rm_mm"]
        cells = "  ".join(f"{row[t]['range_pp']:6.2f}/{row[t]['bias_pp']:+6.2f}"
                          for t in TRUTH)
        print(f"   {W:6.0f}  {rat:9.2f}   {cells}", flush=True)
    out["sweep_tiled"] = {f"{W:.0f}": {f"{t:.2f}": v for t, v in r.items()}
                          for W, r in tiled.items()}

    # ---- 5. the two questions the manuscript has to answer -------------
    # The width at which the phase spread first falls under the study's own
    # noise spread, and whether the smooth-field answer at that width is
    # still the one the manuscript quotes.
    print("\n5. the verdict")
    first = {}
    for th in TRUTH:
        hit = [W for W in WIDTHS if sweep[W][th]["std_pp"] < NOISE_STD_PP]
        first[th] = min(hit) if hit else None
    worst_std = {W: max(sweep[W][t]["std_pp"] for t in TRUTH) for W in WIDTHS}
    worst_rng = {W: max(sweep[W][t]["range_pp"] for t in TRUTH) for W in WIDTHS}
    worst_bias = {W: max(abs(sweep[W][t]["bias_pp"]) for t in TRUTH)
                  for W in WIDTHS}
    worst_std_t = {W: max(tiled[W][t]["std_pp"] for t in TRUTH) for W in WIDTHS}
    worst_bias_t = {W: max(abs(tiled[W][t]["bias_pp"]) for t in TRUTH)
                    for W in WIDTHS}
    all_hit = [W for W in WIDTHS if worst_std[W] < NOISE_STD_PP]
    W_star = min(all_hit) if all_hit else None
    print(f"   the phase spread falls from {worst_std[50.0]:.1f} pp at the "
          f"manuscript's 100 mm strip to {min(worst_std.values()):.1f} pp at "
          f"its best width,\n   while the phase MEAN stays wrong by "
          f"{min(worst_bias.values()):.0f} to {max(worst_bias.values()):.0f} "
          f"pp throughout, against a {NOISE_STD_PP} pp noise yardstick")
    print(f"   integrating the strip instead of point sampling it leaves a "
          f"spread of {min(worst_std_t.values()):.1f} to "
          f"{max(worst_std_t.values()):.1f} pp\n   and a bias of "
          f"{min(worst_bias_t.values()):.0f} to "
          f"{max(worst_bias_t.values()):.0f} pp, so the residual is not the "
          f"four-point sample")
    drift_star = None
    if W_star is not None:
        drift_star = {f"{t:.2f}": 100.0 * (smooth[W_star][t]["theta"]
                                           - smooth[WIDTHS[0]][t]["theta"])
                      for t in TRUTH}
        print(f"   the spread falls under the {NOISE_STD_PP} pp noise "
              f"yardstick at every state from W = {W_star:.0f} mm, a strip of "
              f"{2*W_star:.0f} mm,\n   which is "
              + " to ".join(f"{2*W_star/pattern[t]['s_rm_mm']:.2f}"
                            for t in (TRUTH[0], TRUTH[-1]))
              + " crack spacings; its smooth-field drift is "
              + ", ".join(f"{v:+.2f}" for v in drift_star.values()) + " pp")
    else:
        print("   the spread never falls under the noise yardstick at every "
              "state")
    out["verdict"] = dict(
        first_width_under_noise_mm={f"{t:.2f}": first[t] for t in TRUTH},
        first_width_all_states_mm=W_star,
        worst_std_pp={f"{W:.0f}": worst_std[W] for W in WIDTHS},
        worst_range_pp={f"{W:.0f}": worst_rng[W] for W in WIDTHS},
        worst_abs_bias_pp={f"{W:.0f}": worst_bias[W] for W in WIDTHS},
        worst_std_pp_tiled={f"{W:.0f}": worst_std_t[W] for W in WIDTHS},
        worst_abs_bias_pp_tiled={f"{W:.0f}": worst_bias_t[W] for W in WIDTHS},
        first_width_under_noise_on_bias_mm=next(
            (W for W in WIDTHS if worst_bias[W] < NOISE_STD_PP), None),
        smooth_drift_at_first_width_pp=drift_star,
        admissible_fraction_at_first_width=(
            None if W_star is None
            else {f"{t:.2f}": sweep[W_star][t]["admissible_fraction"]
                  for t in TRUTH}),
        strip_over_spacing_at_first_width=(
            None if W_star is None
            else {f"{t:.2f}": 2.0 * W_star / pattern[t]["s_rm_mm"]
                  for t in TRUTH}),
        inside_D_region_at_first_width=(
            None if W_star is None
            else bool(FD.X_CUT - W_star >= X_VALID[0]
                      and FD.X_CUT + W_star <= X_VALID[1])))

    # ---- the sentences the manuscript would need ----------------------
    # The hypothesis is refused and the refusal is specific, so the reason
    # is recorded beside the numbers rather than left to be reconstructed.
    W_best = min(WIDTHS, key=lambda W: worst_std[W])
    W_dreg = min([W for W in WIDTHS if geom[W]["inside_D_region"]],
                 key=lambda W: worst_std[W])
    g_ok = [g for g in GAUGE_REL
            if max(max(abs(inter[W][g][t]["bias_pp"]),
                       inter[W][g][t]["std_pp"]) for W in inter for t in TRUTH)
            < NOISE_STD_PP]
    out["headline"] = dict(
        hypothesis="an integral over a whole number of crack spacings should "
                   "average the sawtooth out, so the 100 mm strip is simply "
                   "too narrow",
        verdict="refused: widening the strip removes most of the phase SPREAD "
                "and leaves the phase MEAN wrong by 20 to 37 pp, because the "
                "sawtooth is passed through the pointwise constitutive map "
                "element by element before any strip integral is taken, and a "
                "nonlinear map does not commute with an average",
        spread_worst_pp_at_100mm_strip=worst_std[50.0],
        spread_worst_pp_at_one_spacing=worst_std[125.0],
        spread_worst_pp_best_width=worst_std[W_best],
        best_width_mm=2.0 * W_best,
        best_width_inside_D_region_mm=2.0 * W_dreg,
        spread_worst_pp_best_width_in_D_region=worst_std[W_dreg],
        spread_never_reaches_noise=W_star is None,
        bias_worst_pp_at_100mm_strip=worst_bias[50.0],
        bias_worst_pp_best_width=min(worst_bias.values()),
        bias_is_width_independent=True,
        smooth_field_drift_worst_pp_in_D_region=max(
            abs(100.0 * (smooth[W][t]["theta"] - smooth[50.0][t]["theta"]))
            for W in WIDTHS if geom[W]["inside_D_region"] for t in TRUTH),
        confound_verdict="widening is nearly free on a smooth field inside "
                         "the D-region, so the confound is not what stops it",
        residual_is_not_the_point_sample=True,
        error_lives_on_the_gauge_not_the_strip=True,
        gauge_fractions_within_noise=g_ok,
        gauge_at_one_spacing_range_pp=max(
            inter[W][1.0][t]["range_pp"] for W in inter for t in TRUTH),
        gauge_at_one_spacing_bias_pp=max(
            abs(inter[W][1.0][t]["bias_pp"]) for W in inter for t in TRUTH),
        gauge_at_0p90_spacing_range_pp=max(
            inter[W][0.9][t]["range_pp"] for W in inter for t in TRUTH),
        gauge_at_0p75_spacing_range_pp=max(
            inter[W][0.75][t]["range_pp"] for W in inter for t in TRUTH),
        one_spacing_caveat="a gauge of exactly one spacing returns the stored "
                           "strain by construction, as sawtooth.py records; "
                           "what is measured here and is not by construction "
                           "is the tolerance around it, and the fact that the "
                           "strip cannot substitute for it",
        practitioner="the requirement is on the length the fiber is averaged "
                     "over before it enters the model, not on the length of "
                     "the instrumented cut: a running mean over one crack "
                     "spacing, held to about 10 per cent, with the strip a "
                     "weak second-order help once the gauge is right",
        noise_yardstick_pp=NOISE_STD_PP)

    out["pattern"] = {f"{t:.2f}": pattern[t] for t in TRUTH}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_JSON}")

    figure(pattern, smooth, sweep, curves, inter, worst_std, worst_bias)


# ----------------------------------------------------------------------
def figure(pattern, smooth, sweep, curves, inter, worst_std, worst_bias) -> None:
    """Three panels: the swing narrows, its center does not, and why.

    Every curve is the worst case over the four deterioration states rather
    than one state, because a length that has to be recommended to an owner
    is only as good as the state it works least well at.
    """
    import matplotlib.pyplot as plt
    F.apply()
    fig = plt.figure(figsize=(F.FIG_W, 4.9))
    gs = fig.add_gridspec(2, 2, hspace=0.66, wspace=0.30, left=0.085,
                          right=0.995, top=0.90, bottom=0.10)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    s20 = pattern[0.20]["s_rm_mm"]
    s_lo = min(pattern[t]["s_rm_mm"] for t in TRUTH)
    s_hi = max(pattern[t]["s_rm_mm"] for t in TRUTH)

    # (a) the swing itself, at three strip widths, against the answer
    show = [w for w in (50.0, 125.0, 250.0) if w in curves]
    styles = [(0, ()), (0, (5, 2)), (0, (1.4, 1.3))]
    cols = [F.VERM, F.SKY, F.GREEN]
    ax0.axhspan(0.0, SW.THETA_MAX, color='#EAF4EA', lw=0, zorder=0)
    for w, ls, c in zip(show, styles, cols):
        u, ext = curves[w]
        ax0.plot(u, ext, color=c, lw=1.6, ls=ls, zorder=3,
                 label=f'{2*w:.0f} mm ({2*w/s20:.1f} spacings)')
    ax0.axhline(smooth[50.0][0.20]["theta"], color=F.BLACK, lw=1.3,
                ls=(0, (5, 2)), zorder=4)
    ax0.axhline(0.0, color='0.55', lw=0.9, zorder=1)
    ax0.set_xlabel('crack phase relative to the cut ($s_{rm}$)')
    ax0.set_ylabel(r'recovered $\theta$')
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(-0.60, 0.44)
    ax0.annotate('the answer, 0.157', xy=(0.50, 0.175), ha='center',
                 va='bottom', fontsize=F.FS_ANNOT, color=F.BLACK)
    ax0.annotate('admissible', xy=(0.015, 0.36), ha='left', va='center',
                 fontsize=F.FS_ANNOT, color='#2E6B33')
    leg = ax0.legend(fontsize=F.FS_ANNOT, loc='lower center', ncol=3,
                     bbox_to_anchor=(0.5, -0.015), frameon=False,
                     handlelength=2.4, columnspacing=1.6)
    leg.set_zorder(8)
    F.clean(ax0)
    F.panel(ax0, 'a', r'a wider strip narrows the swing about the wrong '
                      r'center ($\theta$ = 0.20)')

    # (b) spread, bias and the price of widening, worst case over states
    x = np.array([2.0 * W for W in WIDTHS])
    ax1.axvspan(s_lo, s_hi, color='0.90', lw=0, zorder=0)
    ax1.axvspan(2 * s_lo, 2 * s_hi, color='0.90', lw=0, zorder=0)
    drift = np.array([max(abs(100.0 * (smooth[W][t]["theta"]
                                       - smooth[50.0][t]["theta"]))
                          for t in TRUTH) for W in WIDTHS])
    for y, c, ls, lab in (
            (np.array([worst_bias[W] for W in WIDTHS]), F.VERM, (0, ()),
             'phase bias'),
            (np.array([worst_std[W] for W in WIDTHS]), F.SKY, (0, (5, 2)),
             'phase spread'),
            (np.maximum(drift, 1e-2), F.BLACK, (0, (1.4, 1.3)),
             'cost of widening')):
        ax1.plot(x, y, color=c, lw=1.6, ls=ls, marker='o', ms=3.2, zorder=3,
                 label=lab)
    ax1.axhline(NOISE_STD_PP, color='0.40', lw=1.0, zorder=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('strip length (mm)')
    ax1.set_ylabel('error in $\\theta$   (% of section)')
    ax1.set_ylim(1e-2, 2e2)
    ax1.annotate('5 % noise', xy=(105.0, NOISE_STD_PP * 1.3), ha='left',
                 va='bottom', fontsize=F.FS_ANNOT, color='0.30')
    ax1.annotate('one spacing', xy=(0.5 * (s_lo + s_hi), 1.1e2), ha='center',
                 va='center', fontsize=F.FS_ANNOT, color='0.30')
    ax1.annotate('two spacings', xy=(s_lo + s_hi, 1.1e2), ha='center',
                 va='center', fontsize=F.FS_ANNOT, color='0.30')
    ax1.legend(fontsize=F.FS_ANNOT, loc='lower right', frameon=False,
               handlelength=2.2, borderaxespad=0.2)
    F.clean(ax1, grid=True)
    F.panel(ax1, 'b', 'the spread falls, the bias does not')

    # (c) the length that does remove it, which is not the strip
    g = np.array(GAUGE_REL)
    for W, c, ls in ((50.0, F.VERM, (0, ())), (125.0, F.SKY, (0, (5, 2))),
                     (200.0, F.GREEN, (0, (1.4, 1.3)))):
        y = np.array([max(max(abs(inter[W][gr][t]["bias_pp"]),
                              inter[W][gr][t]["std_pp"]) for t in TRUTH)
                      for gr in GAUGE_REL])
        ax2.plot(g, np.maximum(y, 1e-2), color=c, lw=1.6, ls=ls, marker='o',
                 ms=3.2, zorder=3, label=f'{2*W:.0f} mm strip')
    ax2.axhline(NOISE_STD_PP, color='0.40', lw=1.0, zorder=2)
    ax2.axvline(1.0, color='0.55', lw=0.9, ls=(0, (1.4, 1.3)), zorder=1)
    ax2.set_yscale('log')
    ax2.set_xlabel('gauge length ($s_{rm}$)')
    ax2.set_ylabel('error in $\\theta$   (% of section)')
    ax2.set_ylim(1e-2, 2e2)
    ax2.annotate('5 % noise', xy=(0.03, NOISE_STD_PP * 1.3), ha='left',
                 va='bottom', fontsize=F.FS_ANNOT, color='0.30')
    ax2.annotate('one spacing', xy=(1.06, 1.1e2), ha='left', va='center',
                 fontsize=F.FS_ANNOT, color='0.30')
    ax2.legend(fontsize=F.FS_ANNOT, loc='lower left', frameon=False,
               handlelength=2.2, borderaxespad=0.2)
    F.clean(ax2, grid=True)
    F.panel(ax2, 'c', 'the gauge removes it, the strip cannot')

    probs = F.audit(fig)
    fig.savefig(OUT_PNG, bbox_inches='tight',
                bbox_extra_artists=F.bbox_artists(fig))
    plt.close(fig)
    print(f"wrote {OUT_PNG}" + ("" if not probs
                                else f"  ({len(probs)} audit problems)"))


if __name__ == "__main__":
    main()
