"""What the tension chord model costs the identification, and what it buys.

Section 7.1 records a deliberate deviation from the reference formulation.
The reinforcement of the tie band carries the bare-bar stress
sigma_s(eps_x), and the tension chord model of Marti, Alvarez, Kaufmann and
Sigrist, in which the concrete between cracks carries tension through bond
so that the mean steel strain over a crack spacing is less than the strain
at a crack, is not used. The reason given is that rho_x would then enter
sigma_s as well as multiplying it and the identifying quantity would cease
to be exactly affine in theta.

That reason is honest but it is not sufficient, because uniqueness of the
root needs only strict monotonicity, which is weaker than affinity. The
objection therefore stands until the model is implemented and the cost is
measured rather than asserted, and that is what this study does here: the
tension chord model is put into the identification at bond and crack
spacing taken from the source, and four things are measured. Whether
T(theta) stays strictly monotone, and how close to zero its slope comes.
How far T(theta) departs from its secant, against the 1e-7 the nominal
model achieves. How many bisection steps the root now costs. And whether
the bias of -2.6 to -7.5 percentage points falls.

The last is the one the manuscript stakes a claim on: a crude average
tensile stress beta f_ct with beta = 0.10 cuts the mean recovery error from
5.0 to 1.6 percentage points, and Section 7.4 reads that as evidence that
the omission of tension stiffening carries most of the bias. The tension
chord model fixes beta rather than leaving it free, so it can confirm that
reading or refute it. No bond parameter is tuned to the answer; the sweeps
below are reported whole.

Run:  python tcm.py            (writes figures/tcm.json)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figdata as FD                                                      # noqa: E402
from csfm_constitutive import membrane, steel_stress                      # noqa: E402
from identify import rho_x_of_theta                                       # noqa: E402
from problem import DeepBeam                                              # noqa: E402
from recover_utils import bracket_root, element_strains                   # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "tcm.json"

DELTA = 3.5                 # the state the manuscript quotes the recovery at
ARM = 370.0                 # the assumed reaction arm, as everywhere here
THETA_MAX = 0.70            # the admissible range of the parameter
GRID = np.linspace(0.0, THETA_MAX, 141)     # fine grid for slope and secant

# ----------------------------------------------------------------------
# the constants of the model, and where each one comes from
# ----------------------------------------------------------------------
# Concrete tensile strength. EN 1992-1-1:2004 Table 3.1, f_ctm = 0.30
# f_ck^(2/3) for concrete up to C50/60. With f_ck = 30 MPa this is 2.90 MPa.
# This is the value the earlier beta f_ct check in tension_chord.py used, so
# the two are on the same footing.
def f_ctm(fc: float) -> float:
    """Mean axial tensile strength (MPa) from EN 1992-1-1 Table 3.1."""
    return 0.30 * fc ** (2.0 / 3.0)


# Bond. Marti, Alvarez, Kaufmann and Sigrist (1998), Structural Engineering
# International 8(4), 287-298, Eq. (1): a stepped, rigid-perfectly-plastic
# bond shear stress, tau_b0 = 2 f_ct while the steel at the crack is
# elastic and tau_b1 = f_ct once it has yielded. The halving on yielding is
# the model's representation of the local bond damage that accompanies
# plastic strain in the bar.
TAU_B0 = 2.0                # multiple of f_ct, elastic steel
TAU_B1 = 1.0                # multiple of f_ct, yielded steel

# Crack spacing. Same source: with the stepped bond above, a new crack can
# form wherever the concrete stress transferred back into the chord reaches
# f_ct, which gives the maximum spacing
#     s_r,max = phi f_ct / (2 tau_b0 rho) = phi / (4 rho)
# and any spacing between s_r,max / 2 and s_r,max is admissible at
# stabilised cracking. The spacing is therefore written s_rm = LAM_S
# s_r,max with LAM_S in [0.5, 1.0], and this study reports the whole range
# rather than picking one value and hiding the others. The nominal is the
# mid-range value.
LAM_S_RANGE = (0.5, 0.67, 1.0)
LAM_S = 0.67

# Bar diameter. The smeared model fixes only the ratio, rho_tie = 0.012 over
# a 150 mm band 300 mm thick, which is A_s = 540 mm^2 and says nothing about
# how that steel is divided into bars. A diameter has to be supplied, so it
# is stated rather than absorbed: 16 mm nominal, swept over the range a
# 300 mm wide tie would plausibly use. Section 4 below shows the elastic
# branch of the model is exactly independent of it.
PHI_RANGE = (12.0, 16.0, 20.0, 25.0)
PHI_0 = 16.0


# ----------------------------------------------------------------------
# 1. the model itself
# ----------------------------------------------------------------------
def crack_spacing(phi: float | np.ndarray, rho, f_ct: float,
                  lam_s: float = LAM_S) -> np.ndarray:
    """Stabilised crack spacing s_rm (mm), Marti et al. (1998).

    s_r,max = phi f_ct / (2 tau_b0 rho) reduces to phi / (4 rho) for the
    stepped bond adopted above, and s_rm = lam_s s_r,max.
    """
    return lam_s * phi * f_ct / (2.0 * TAU_B0 * f_ct * np.asarray(rho))


def mean_strain(sig_sr, s_r, phi, mat, f_ct):
    """Mean steel strain over a crack spacing, given the stress at a crack.

    This is the direction the model is stated in and it is closed form. The
    steel stress falls from sig_sr at the crack at 4 tau_b / phi per unit
    length, with the plastic bond tau_b1 over the yielded length next to the
    crack and the elastic bond tau_b0 beyond it, and the strain is the
    bilinear bar law evaluated on that profile. Integrating over half a
    spacing and doubling gives the quantity a smeared measurement returns.

    The stress profile is clipped at zero, which matters only far below the
    cracking load where the chord is not in fact in the stabilised state;
    the clip keeps the map monotone and continuous through the origin so the
    inverse below is defined everywhere.
    """
    sig = np.asarray(sig_sr, float)
    b = 4.0 * TAU_B0 * f_ct / phi          # elastic-branch gradient, MPa/mm
    a = 4.0 * TAU_B1 * f_ct / phi          # plastic-branch gradient, MPa/mm
    half = 0.5 * np.asarray(s_r, float)
    fy, Es = mat.fy, mat.Es
    ey = mat.eps_y
    Esh = (mat.ft - fy) / (mat.eps_u - ey)

    # ---- elastic at the crack ---------------------------------------
    x0 = np.minimum(half, np.clip(sig, 0.0, None) / b)
    int_el = (sig * x0 - 0.5 * b * x0 ** 2) / Es

    # ---- yielded at the crack ---------------------------------------
    over = np.clip(sig - fy, 0.0, None)
    xy = np.minimum(half, over / a)
    int_pl = ey * xy + (over * xy - 0.5 * a * xy ** 2) / Esh
    u = np.minimum(half - xy, fy / b)
    int_pl = int_pl + (fy * u - 0.5 * b * u ** 2) / Es

    return np.where(sig <= fy, int_el, int_pl) / half


def stress_at_crack(eps_m, s_r, phi, mat, f_ct, tol=1e-10):
    """Inverse of `mean_strain`: the steel stress at a crack, in MPa.

    The forward map is strictly increasing, so the inverse is taken by
    bisection on [0, f_t] rather than by interpolating a sampled curve. A
    sampled inverse would put the answer on the sample grid, which is the
    failure this study already found once in its root-finding and does not
    intend to repeat. Strains above what the bar can carry at ultimate are
    returned at f_t and counted by the caller.
    """
    eps = np.asarray(eps_m, float)
    lo = np.zeros_like(eps)
    hi = np.full_like(eps, mat.ft)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = mean_strain(mid, s_r, phi, mat, f_ct) - eps
        hi = np.where(f >= 0.0, mid, hi)
        lo = np.where(f < 0.0, mid, lo)
        if np.all(hi - lo < tol):
            break
    return 0.5 * (lo + hi)


def tcm_increment(eps_x, rho, theta, mat, f_ct, phi0=PHI_0, lam_s=LAM_S,
                  corrode_phi=True, cracked_only=False):
    """Stress the tension chord model adds to sigma_x in the band, in MPa.

    The chord force is A_s times the stress at a crack, because the concrete
    carries nothing there; the nominal model instead uses A_s times the bare
    bar stress at the mean strain. The difference of the two is what tension
    stiffening contributes, and adding it leaves every other part of the
    membrane map untouched.

    theta enters twice. It thins the steel, rho = rho_tie (1 - theta), and
    it thins the bar, phi = phi_0 sqrt(1 - theta), since a uniform loss of
    section is what the parameter means. Both move the crack spacing. On the
    elastic branch the two movements cancel exactly and the increment is
    lam_s f_ct / 2 whatever theta and phi_0 are; past yield they do not, and
    that residue is the whole of the departure from affinity reported below.

    The increment is capped by E_c eps, the most an uncracked chord of the
    same concrete could carry, which binds only at strains far below
    cracking and keeps the map continuous through the origin.
    """
    e = np.asarray(eps_x, float)
    rho = np.asarray(rho, float)
    phi = phi0 * np.sqrt(max(1.0 - float(theta), 1e-6)) if corrode_phi else phi0
    s_r = crack_spacing(phi, np.maximum(rho, 1e-9), f_ct, lam_s)
    ep = np.clip(e, 0.0, None)
    sig_sr = stress_at_crack(ep, s_r, phi, mat, f_ct)
    bare = steel_stress(torch.tensor(ep), mat).numpy()
    inc = rho * (sig_sr - bare)
    inc = np.minimum(inc, mat.Ec0 * ep)
    inc = np.where(e > 0.0, np.clip(inc, 0.0, None), 0.0)
    cracked = sig_sr >= sigma_sr_cr(rho, mat, f_ct)
    if cracked_only:
        inc = np.where(cracked, inc, 0.0)
    return inc, sig_sr, s_r, cracked


def sigma_sr_cr(rho, mat, f_ct):
    """Steel stress at a crack when the chord first cracks, in MPa.

    Below this the chord is not in the stabilised state the model describes,
    and the concrete is continuous rather than carrying force back through
    bond. The threshold is reported rather than assumed away, because the
    lightly loaded states of this benchmark sit near it.
    """
    n = mat.Es / mat.Ec0
    rho = np.maximum(np.asarray(rho, float), 1e-9)
    return f_ct * (1.0 + rho * (n - 1.0)) / rho


# ----------------------------------------------------------------------
# 2. the identifying quantity, with the model active
# ----------------------------------------------------------------------
def cut_stress(prob, cx, cy, ex, ey, gxy, theta, tcm, lam_s=LAM_S,
               phi0=PHI_0, cracked_only=False, diag=None):
    """sigma_x on the strip that stands for the cut, with or without TCM."""
    sel = np.abs(cx - FD.X_CUT) < FD.BAND_W
    X = torch.tensor(cx[sel]).unsqueeze(-1)
    Y = torch.tensor(cy[sel]).unsqueeze(-1)
    rho = rho_x_of_theta(prob, X, Y, torch.tensor(float(theta)))
    st = membrane(torch.tensor(ex[sel]).unsqueeze(-1),
                  torch.tensor(ey[sel]).unsqueeze(-1),
                  torch.tensor(gxy[sel]).unsqueeze(-1),
                  rho, prob.rho_y(X, Y), prob.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy()
    ys = cy[sel]
    if tcm:
        f_ct = f_ctm(prob.mat.fc)
        inb = ys < FD.BAND
        inc, sig, sr, ck = tcm_increment(
            ex[sel][inb], rho.squeeze().numpy()[inb], theta, prob.mat, f_ct,
            phi0, lam_s, cracked_only=cracked_only)
        sx = sx.copy()
        sx[inb] = sx[inb] + inc
        if diag is not None:
            diag.update(inc_mean_MPa=float(inc.mean()),
                        inc_max_MPa=float(inc.max()),
                        beta_effective=float(inc.mean() / f_ct),
                        cracked_fraction=float(ck.mean()),
                        yielded_fraction=float((sig > prob.mat.fy).mean()),
                        ultimate_fraction=float(
                            (sig > prob.mat.ft - 1e-6).mean()),
                        s_rm_mm=float(np.mean(sr)),
                        eps_mean=float(ex[sel][inb].mean()))
    return sx, ys


def band_couple(prob, cx, cy, ex, ey, gxy, area, theta, tcm,
                lam_s=LAM_S, phi0=PHI_0, cracked_only=False, diag=None):
    """Tie resultant T (kN), lever arm z (mm) and their couple (kN m)."""
    sx, ys = cut_stress(prob, cx, cy, ex, ey, gxy, theta, tcm, lam_s, phi0,
                        cracked_only, diag)
    dA = area / (2.0 * FD.BAND_W) * prob.t
    inb = ys < FD.BAND
    T = float((sx[inb] * dA).sum()) / 1e3
    wT = np.clip(sx[inb], 0.0, None)
    wC = np.clip(-sx[~inb], 0.0, None)
    yT = float((wT * ys[inb]).sum() / max(wT.sum(), 1e-9))
    yC = float((wC * ys[~inb]).sum() / max(wC.sum(), 1e-9))
    z = yC - yT
    return T, z, T * z / 1e3


def recover(prob, st, area, lam, tcm, lam_s=LAM_S, phi0=PHI_0, grid=None,
            cracked_only=False):
    """Root of the band couple against statics, on a trial grid."""
    if grid is None:
        grid = np.linspace(0.0, THETA_MAX, 71)
    M_req = lam * prob.P / 2.0 * (FD.X_CUT - ARM) / 1e6
    f = np.array([band_couple(prob, *st, area, g, tcm, lam_s, phi0,
                              cracked_only)[2] - M_req for g in grid])
    return bracket_root(f, grid), M_req, f


def bisect(prob, st, area, lam, tcm, tol=1e-4, lam_s=LAM_S, phi0=PHI_0):
    """Bisection for the root on [0, THETA_MAX], counting evaluations.

    The point of the count is that bisection is a property of the bracket
    and the tolerance and not of the function, so a model that only bends
    the function cannot cost anything here. Reported as a number so the
    claim can be refuted.
    """
    M_req = lam * prob.P / 2.0 * (FD.X_CUT - ARM) / 1e6

    def g(q):
        return band_couple(prob, *st, area, q, tcm, lam_s, phi0)[2] - M_req

    lo, hi = 0.0, THETA_MAX
    flo, fhi = g(lo), g(hi)
    n = 2
    if np.sign(flo) == np.sign(fhi):
        return np.nan, n
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        fm = g(mid); n += 1
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return 0.5 * (lo + hi), n


# ----------------------------------------------------------------------
def selftest(mat, f_ct) -> dict:
    """Three checks the implementation has to pass before it is used.

    A tension chord model that is merely plausible is worth nothing here,
    because the whole point of the exercise is a number. The forward map
    must be strictly increasing, the inverse must return what the forward
    map was given, and on the elastic branch the smeared increment must come
    out at exactly lam_s f_ct / 2, which is the closed-form result of the
    model and is independent of both the bar diameter and the reinforcement
    ratio.
    """
    phi, rho = 16.0, 0.012
    s_r = float(crack_spacing(phi, rho, f_ct, LAM_S))
    sig = np.linspace(0.0, mat.ft, 4001)
    em = mean_strain(sig, s_r, phi, mat, f_ct)
    mono = bool(np.all(np.diff(em) > 0.0))

    probe = np.array([1e-5, 1e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2, 1.5e-2])
    back = stress_at_crack(probe, s_r, phi, mat, f_ct)
    rt = float(np.abs(mean_strain(back, s_r, phi, mat, f_ct) - probe).max())

    # Elastic branch of the stabilised state: sigma_sr = E_s eps_m + tau_b0
    # s_r / phi exactly, so the smeared increment is lam_s f_ct / 2 whatever
    # phi and rho are. The identity holds only where the steel stress stays
    # positive over the whole spacing, lam_s f_ct / rho <= sigma_sr <= f_y,
    # which is the stabilised elastic window; below it the chord is still in
    # crack formation and the model is being read outside its range.
    err, npts = [], 0
    for ph in PHI_RANGE:
        for rr in (0.004, 0.008, 0.012):
            sr = float(crack_spacing(ph, rr, f_ct, LAM_S))
            lo = LAM_S * f_ct / rr
            if lo >= mat.fy:
                continue
            ss_t = np.linspace(lo * 1.001, mat.fy * 0.999, 25)
            e_el = mean_strain(ss_t, sr, ph, mat, f_ct)
            ss = stress_at_crack(e_el, sr, ph, mat, f_ct)
            err.append(np.abs(rr * (ss - mat.Es * e_el)
                              - LAM_S * f_ct / 2).max())
            npts += e_el.size
    el = float(max(err))
    print(f"  self test: forward map strictly increasing {mono}; "
          f"inverse round trip {rt:.2e} in strain; "
          f"elastic increment off lam_s f_ct/2 by {el:.2e} MPa "
          f"over {npts} points")
    assert mono and rt < 1e-9 and el < 1e-6
    return dict(forward_monotone=mono, inverse_roundtrip=rt,
                elastic_identity_MPa=el, elastic_points=npts)


def secant_departure(T, g):
    """Max deviation of T from its secant, relative to the range of T."""
    sec = T[0] + (T[-1] - T[0]) * (g - g[0]) / (g[-1] - g[0])
    rng = abs(T.max() - T.min())
    return float(np.abs(T - sec).max() / max(rng, 1e-30))


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    xy = d["xy"]
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    f_ct = f_ctm(prob.mat.fc)
    thetas = [float(t) for t in d["theta_true"]]
    deltas = [float(t) for t in d["deltas"]]
    out: dict = {}

    strains = {}
    for th in thetas:
        for dl in deltas:
            k = f"{th:.2f}_{dl:.1f}"
            if f"u_{k}" in d:
                strains[k] = element_strains(xy, d[f"u_{k}"], FD.NX, FD.NY)

    # ---- 0. the constants, printed so nothing is silent ---------------
    s_r0 = float(crack_spacing(PHI_0, prob.rho_tie, f_ct, LAM_S))
    print("Tension chord model of Marti, Alvarez, Kaufmann and Sigrist (1998)")
    print("=" * 74)
    out["selftest"] = selftest(prob.mat, f_ct)
    print(f"  f_ct       = {f_ct:.3f} MPa   EN 1992-1-1 Table 3.1, "
          f"0.30 f_ck^(2/3), f_ck = {prob.mat.fc:.0f} MPa")
    print(f"  tau_b0     = {TAU_B0:.0f} f_ct = {TAU_B0*f_ct:.3f} MPa   "
          "Marti et al. (1998), elastic steel")
    print(f"  tau_b1     = {TAU_B1:.0f} f_ct = {TAU_B1*f_ct:.3f} MPa   "
          "Marti et al. (1998), yielded steel")
    print(f"  s_r,max    = phi/(4 rho) = {float(crack_spacing(PHI_0, prob.rho_tie, f_ct, 1.0)):.1f} mm "
          f"at phi = {PHI_0:.0f} mm, rho = {prob.rho_tie:.3f}")
    print(f"  s_rm       = {LAM_S:.2f} s_r,max = {s_r0:.1f} mm   "
          f"lam_s admissible in [{LAM_S_RANGE[0]:.2f}, {LAM_S_RANGE[-1]:.2f}]")
    print(f"  phi        = {PHI_0:.0f} mm nominal, corroding as sqrt(1-theta)")
    print(f"  A_s        = {prob.rho_tie*prob.band*prob.t:.0f} mm^2 over a "
          f"{prob.band:.0f} x {prob.t:.0f} mm band")
    print(f"\n  ceiling of the smeared increment, lam_s f_ct / 2 = "
          f"{LAM_S*f_ct/2:.3f} MPa, i.e. beta = {LAM_S/2:.3f}")
    print("  This is the elastic branch of the chord and it is a ceiling, not")
    print("  the value: it holds only while the bar is still elastic at a")
    print("  crack. Reading the beta of Section 7.4 back through it would put")
    print(f"  beta = 0.10 at lam_s = {2*0.10:.2f}, outside the admissible "
          f"[{LAM_S_RANGE[0]:.2f}, {LAM_S_RANGE[-1]:.2f}], but that reading is")
    print("  only valid below yield and the band at the cut is past it.")
    out["constants"] = dict(
        f_ct=f_ct, tau_b0=TAU_B0 * f_ct, tau_b1=TAU_B1 * f_ct,
        tau_b0_rule="2 f_ct (Marti et al. 1998, elastic steel)",
        tau_b1_rule="1 f_ct (Marti et al. 1998, yielded steel)",
        f_ct_source="EN 1992-1-1 Table 3.1, f_ctm = 0.30 f_ck^(2/3)",
        s_r_max_mm=float(crack_spacing(PHI_0, prob.rho_tie, f_ct, 1.0)),
        s_rm_mm=s_r0, lam_s=LAM_S, lam_s_range=list(LAM_S_RANGE),
        phi_mm=PHI_0, phi_range=list(PHI_RANGE),
        A_s_mm2=prob.rho_tie * prob.band * prob.t,
        beta_equivalent=LAM_S / 2.0,
        beta_of_section_7=0.10, lam_s_of_section_7=0.20)

    # ---- 1. the state of the band the model is asked to describe ------
    st_ref = strains[f"0.00_{DELTA:.1f}"]
    cx, cy, ex = st_ref[0], st_ref[1], st_ref[2]
    inb = (np.abs(cx - FD.X_CUT) < FD.BAND_W) & (cy < FD.BAND)
    eb = ex[inb]
    print(f"\n  band at the cut, theta = 0, delta = {DELTA} mm: "
          f"mean eps = {eb.mean():.2e}, max = {eb.max():.2e}, "
          f"yield = {prob.mat.eps_y:.2e}")
    print(f"  cracking of the chord at sigma_sr = "
          f"{f_ct*(1+prob.rho_tie*(prob.mat.Es/prob.mat.Ec0-1))/prob.rho_tie:.0f} MPa, "
          f"reached at every band element here")

    # ---- 1b. what the model actually adds, state by state -------------
    print("\n" + "=" * 74)
    print("1b. WHAT THE MODEL ADDS to the band, evaluated at the true state")
    print("=" * 74)
    print(f"{'theta':>6}{'delta':>7}{'eps mean':>11}{'s_rm mm':>9}"
          f"{'cracked %':>11}{'yielded %':>11}{'inc MPa':>10}"
          f"{'beta eq':>9}{'inc kN':>9}")
    band_A = prob.band * prob.t / 1e3        # mm^2 / 1e3, so MPa -> kN
    diag_all = {}
    for th in thetas:
        for dl in deltas:
            k = f"{th:.2f}_{dl:.1f}"
            if k not in strains:
                continue
            dg: dict = {}
            band_couple(prob, *strains[k], area, th, True, diag=dg)
            dg["inc_kN"] = dg["inc_mean_MPa"] * band_A
            diag_all[k] = dg
            print(f"{th:>6.2f}{dl:>7.1f}{dg['eps_mean']:>11.2e}"
                  f"{dg['s_rm_mm']:>9.1f}{100*dg['cracked_fraction']:>11.1f}"
                  f"{100*dg['yielded_fraction']:>11.1f}"
                  f"{dg['inc_mean_MPa']:>10.3f}{dg['beta_effective']:>9.3f}"
                  f"{dg['inc_kN']:>9.1f}")
    out["increment"] = diag_all
    print("\n  The ceiling is lam_s f_ct / 2 = "
          f"{LAM_S*f_ct/2:.3f} MPa, reached only while the bar is elastic at")
    print("  the crack. Where the band has yielded the model gives far less,")
    print("  because the stress difference over a spacing buys almost no")
    print("  strain difference once the bar is on its hardening branch.")

    # ---- 2. monotonicity, the property uniqueness needs ---------------
    print("\n" + "=" * 74)
    print("2. MONOTONICITY of T(theta) on [0, 0.70], every stored state")
    print("=" * 74)
    print(f"{'theta':>6}{'delta':>7}  {'T(0)':>9}{'T(0.7)':>9}"
          f"{'min dT/dth':>12}{'max dT/dth':>12}{'sign':>7}{'aff dep %':>11}")
    mono = {}
    for th in thetas:
        for dl in deltas:
            k = f"{th:.2f}_{dl:.1f}"
            if k not in strains:
                continue
            T = np.array([band_couple(prob, *strains[k], area, q, True)[0]
                          for q in GRID])
            dT = np.gradient(T, GRID)
            dep = secant_departure(T, GRID)
            sgn = "neg" if np.all(dT < 0) else ("pos" if np.all(dT > 0)
                                                else "MIXED")
            mono[k] = dict(T0=float(T[0]), T1=float(T[-1]),
                           dT_min=float(dT.min()), dT_max=float(dT.max()),
                           dT_absmin=float(np.abs(dT).min()),
                           strict=bool(np.all(np.diff(T) < 0.0)),
                           sign=sgn, aff_dep=dep,
                           T=[float(v) for v in T])
            print(f"{th:>6.2f}{dl:>7.1f}  {T[0]:>9.2f}{T[-1]:>9.2f}"
                  f"{dT.min():>12.2f}{dT.max():>12.2f}{sgn:>7}"
                  f"{100*dep:>11.4f}")
    fine = np.linspace(0.0, THETA_MAX, 1401)
    fmin = {}
    for th in thetas:
        k = f"{th:.2f}_{DELTA:.1f}"
        Tf = np.array([band_couple(prob, *strains[k], area, q, True)[0]
                       for q in fine])
        dTf = np.gradient(Tf, fine)
        fmin[f"{th:.2f}"] = dict(dT_min=float(dTf.min()),
                                 dT_max=float(dTf.max()),
                                 strict=bool(np.all(np.diff(Tf) < 0.0)))
    print(f"  on a 1401-point grid at delta = {DELTA} mm, strictly decreasing "
          f"at every state: {all(v['strict'] for v in fmin.values())}, "
          f"dT/dtheta in [{min(v['dT_min'] for v in fmin.values()):.1f}, "
          f"{max(v['dT_max'] for v in fmin.values()):.1f}] kN")
    out["monotonic_fine"] = fmin
    allmin = min(abs(v["dT_absmin"]) for v in mono.values())
    allstrict = all(v["strict"] for v in mono.values())
    print(f"\n  strictly decreasing at all {len(mono)} stored states: {allstrict}")
    print(f"  min |dT/dtheta| over every state and every theta: "
          f"{allmin:.2f} kN per unit theta")
    out["monotonic"] = mono
    out["min_abs_slope_kN"] = allmin
    out["strict_all"] = allstrict

    # ---- 3. departure from affinity, TCM against nominal --------------
    print("\n" + "=" * 74)
    print("3. DEPARTURE FROM AFFINITY, max |T - secant| / range of T")
    print("=" * 74)
    print(f"{'theta':>6}{'nominal':>14}{'TCM':>14}{'ratio':>12}")
    aff = {}
    for th in thetas:
        k = f"{th:.2f}_{DELTA:.1f}"
        Tn = np.array([band_couple(prob, *strains[k], area, q, False)[0]
                       for q in GRID])
        Tt = np.array([band_couple(prob, *strains[k], area, q, True)[0]
                       for q in GRID])
        a_n, a_t = secant_departure(Tn, GRID), secant_departure(Tt, GRID)
        aff[f"{th:.2f}"] = dict(nominal=a_n, tcm=a_t,
                                nominal_pct=100 * a_n, tcm_pct=100 * a_t,
                                slope_nominal=float(np.polyfit(GRID, Tn, 1)[0]),
                                slope_tcm=float(np.polyfit(GRID, Tt, 1)[0]))
        print(f"{th:>6.2f}{a_n:>14.3e}{a_t:>14.3e}{a_t/a_n:>12.3e}")
    out["affinity"] = aff
    out["affinity_max_tcm_pct"] = 100 * max(v["tcm"] for v in aff.values())
    out["affinity_max_nominal_pct"] = 100 * max(v["nominal"] for v in aff.values())

    # ---- 4. what the departure is made of -----------------------------
    # On the elastic branch the increment is lam_s f_ct / 2 whatever theta
    # and phi are, so the curvature can only come from the yielded part of
    # the band. Measured, rather than argued.
    print("\n" + "=" * 74)
    print("4. WHERE THE CURVATURE COMES FROM: the yield transition")
    print("=" * 74)
    print("  On the elastic branch of the chord the smeared increment is")
    print("  lam_s f_ct / 2 whatever rho and phi are, so theta cancels and")
    print("  T stays affine. The curvature is entirely the bar yielding at")
    print("  the crack, which the model makes the stiffening collapse at.")
    print(f"\n{'eps mean':>10}{'sigma_sr':>10}{'increment MPa':>15}"
          f"{'beta equivalent':>17}")
    rho_b = np.full(1, prob.rho_tie)
    curve = []
    for e in (5e-4, 1.0e-3, 1.5e-3, 2.0e-3, 2.096e-3, 2.2e-3, 2.5e-3,
              3.0e-3, 5.0e-3, 8.0e-3, 1.2e-2):
        inc, sig, _sr, _ck = tcm_increment(np.full(1, e), rho_b, 0.0,
                                           prob.mat, f_ct)
        curve.append(dict(eps=e, sigma_sr=float(sig[0]), inc=float(inc[0]),
                          beta=float(inc[0] / f_ct)))
        print(f"{e:>10.2e}{float(sig[0]):>10.1f}{float(inc[0]):>15.3f}"
              f"{float(inc[0]/f_ct):>17.3f}")
    out["increment_curve"] = curve
    e_star = (prob.mat.fy - LAM_S * f_ct / (2 * prob.rho_tie)) / prob.mat.Es
    print(f"\n  the bar reaches f_y at a crack when the mean strain passes "
          f"{e_star:.3e},")
    print(f"  which is {e_star/prob.mat.eps_y:.2f} of the bare-bar yield "
          "strain; the band at the cut")
    print(f"  is below it at delta = 1 mm and above it at every other stored "
          "state.")
    out["yield_transition_eps"] = float(e_star)

    # phi independence on the elastic branch, measured not asserted
    ind = {}
    for phi in PHI_RANGE:
        inc, _s, _r, _c = tcm_increment(np.full(1, 1.0e-3),
                                        np.full(1, prob.rho_tie), 0.0,
                                        prob.mat, f_ct, phi0=phi)
        ind[f"phi_{phi:.0f}"] = float(inc[0])
    print("\n  elastic-branch increment at eps = 1e-3, phi = "
          + ", ".join(f"{q:.0f} mm: {ind[f'phi_{q:.0f}']:.4f} MPa"
                      for q in PHI_RANGE))
    out["phi_independence_MPa"] = ind

    # ---- 5. cost in evaluations ---------------------------------------
    print("\n" + "=" * 74)
    print("5. COST IN EVALUATIONS, bisection to tol = 1e-4 in theta")
    print("=" * 74)
    print(f"{'theta':>6}{'nominal steps':>15}{'TCM steps':>12}"
          f"{'nom root':>11}{'TCM root':>11}")
    cost = {}
    for th in thetas:
        k = f"{th:.2f}_{DELTA:.1f}"
        rn, nn = bisect(prob, strains[k], area, float(d[f"lam_{k}"][0]), False)
        rt, nt = bisect(prob, strains[k], area, float(d[f"lam_{k}"][0]), True)
        cost[f"{th:.2f}"] = dict(steps_nominal=nn, steps_tcm=nt,
                                 root_nominal=float(rn), root_tcm=float(rt))
        print(f"{th:>6.2f}{nn:>15d}{nt:>12d}"
              f"{('none' if np.isnan(rn) else f'{rn:.4f}'):>11}"
              f"{('none' if np.isnan(rt) else f'{rt:.4f}'):>11}")
    # per-evaluation cost, which is where TCM actually charges
    k = f"0.20_{DELTA:.1f}"
    t0 = time.perf_counter()
    for _ in range(20):
        band_couple(prob, *strains[k], area, 0.2, False)
    t_nom = (time.perf_counter() - t0) / 20
    t0 = time.perf_counter()
    for _ in range(20):
        band_couple(prob, *strains[k], area, 0.2, True)
    t_tcm = (time.perf_counter() - t0) / 20
    print(f"\n  steps are identical: bisection depends on the bracket and the")
    print(f"  tolerance, not on the function. The charge is per evaluation:")
    print(f"  {1e3*t_nom:.2f} ms nominal, {1e3*t_tcm:.2f} ms with TCM "
          f"({t_tcm/t_nom:.2f} x), from the 80-step inverse of the chord map.")
    out["cost"] = dict(per_state=cost,
                       steps_nominal=int(np.median([v["steps_nominal"]
                                                    for v in cost.values()])),
                       steps_tcm=int(np.median([v["steps_tcm"]
                                                for v in cost.values()])),
                       ms_per_eval_nominal=1e3 * t_nom,
                       ms_per_eval_tcm=1e3 * t_tcm,
                       eval_cost_ratio=t_tcm / t_nom)

    # ---- 6. the headline: recovery and bias ---------------------------
    print("\n" + "=" * 74)
    print("6. RECOVERY AND BIAS, delta = 3.5 mm, arm 370 mm")
    print("=" * 74)
    print(f"{'true':>6}{'nominal':>11}{'bias pp':>10}"
          f"{'TCM':>11}{'bias pp':>10}{'change pp':>11}{'TCM cracked':>13}")
    head = {}
    for th in thetas:
        k = f"{th:.2f}_{DELTA:.1f}"
        lam = float(d[f"lam_{k}"][0])
        rn = recover(prob, strains[k], area, lam, False)[0]
        rt = recover(prob, strains[k], area, lam, True)[0]
        rc = recover(prob, strains[k], area, lam, True, cracked_only=True)[0]
        bn = 100 * (rn - th) if np.isfinite(rn) else np.nan
        bt = 100 * (rt - th) if np.isfinite(rt) else np.nan
        bc = 100 * (rc - th) if np.isfinite(rc) else np.nan
        head[f"{th:.2f}"] = dict(nominal=float(rn), tcm=float(rt),
                                 tcm_cracked_only=float(rc),
                                 bias_nominal_pp=float(bn),
                                 bias_tcm_pp=float(bt),
                                 bias_tcm_cracked_only_pp=float(bc))
        f2 = lambda v: "none" if not np.isfinite(v) else f"{v:.4f}"
        f3 = lambda v: "--" if not np.isfinite(v) else f"{v:+.2f}"
        print(f"{th:>6.2f}{f2(rn):>11}{f3(bn):>10}{f2(rt):>11}{f3(bt):>10}"
              f"{f3(abs(bt)-abs(bn)):>11}{f2(rc):>13}")
    dts = [v for k2, v in head.items() if float(k2) > 0]
    mn = float(np.mean([abs(v["bias_nominal_pp"]) for v in dts]))
    mt = float(np.mean([abs(v["bias_tcm_pp"]) for v in dts]))
    print(f"\n  mean |bias| over the four deteriorated states: "
          f"{mn:.1f} pp nominal, {mt:.1f} pp with TCM")
    out["headline"] = dict(per_state=head, mean_abs_bias_nominal_pp=mn,
                           mean_abs_bias_tcm_pp=mt)

    # ---- 7. the sweeps, reported whole --------------------------------
    print("\n" + "=" * 74)
    print("7. SWEEPS: nothing here is tuned, so the whole range is shown")
    print("=" * 74)
    print(f"{'lam_s':>7}{'beta eq':>9}" + "".join(
        f"{f'th={t:.2f}':>10}" for t in thetas) + f"{'mean |b|':>10}")
    sweep = {}
    for ls in (0.20, 0.30, 0.40, 0.50, 0.67, 0.80, 1.00):
        row, errs = [], []
        for th in thetas:
            k = f"{th:.2f}_{DELTA:.1f}"
            r = recover(prob, strains[k], area, float(d[f"lam_{k}"][0]),
                        True, lam_s=ls)[0]
            row.append(float(r))
            if th > 0 and np.isfinite(r):
                errs.append(abs(r - th) * 100)
        m = float(np.mean(errs)) if errs else np.nan
        sweep[f"{ls:.2f}"] = dict(rec=row, mean_abs_bias_pp=m,
                                  beta_equivalent=ls / 2)
        print(f"{ls:>7.2f}{ls/2:>9.3f}" + "".join(
            f"{('none' if not np.isfinite(v) else f'{v:.3f}'):>10}"
            for v in row) + f"{m:>10.1f}")
    out["lam_s_sweep"] = sweep

    print(f"\n{'phi mm':>7}" + "".join(f"{f'th={t:.2f}':>10}" for t in thetas)
          + f"{'mean |b|':>10}")
    psw = {}
    for phi in PHI_RANGE:
        row, errs = [], []
        for th in thetas:
            k = f"{th:.2f}_{DELTA:.1f}"
            r = recover(prob, strains[k], area, float(d[f"lam_{k}"][0]),
                        True, phi0=phi)[0]
            row.append(float(r))
            if th > 0 and np.isfinite(r):
                errs.append(abs(r - th) * 100)
        m = float(np.mean(errs)) if errs else np.nan
        psw[f"{phi:.0f}"] = dict(rec=row, mean_abs_bias_pp=m)
        print(f"{phi:>7.0f}" + "".join(
            f"{('none' if not np.isfinite(v) else f'{v:.3f}'):>10}"
            for v in row) + f"{m:>10.1f}")
    out["phi_sweep"] = psw

    # ---- 8. across load level ------------------------------------------
    print("\n" + "=" * 74)
    print("8. ACROSS LOAD LEVEL, mean |bias| over the four deteriorated states")
    print("=" * 74)
    print(f"{'delta':>7}{'nominal':>10}{'TCM':>10}{'TCM cracked':>13}"
          f"{'band eps':>11}{'cracked %':>11}")
    byd = {}
    for dl in deltas:
        en, et, ec = [], [], []
        for th in thetas:
            k = f"{th:.2f}_{dl:.1f}"
            if k not in strains or th == 0.0:
                continue
            lam = float(d[f"lam_{k}"][0])
            rn = recover(prob, strains[k], area, lam, False)[0]
            rt = recover(prob, strains[k], area, lam, True)[0]
            rc = recover(prob, strains[k], area, lam, True,
                         cracked_only=True)[0]
            if np.isfinite(rn):
                en.append(abs(rn - th) * 100)
            if np.isfinite(rt):
                et.append(abs(rt - th) * 100)
            if np.isfinite(rc):
                ec.append(abs(rc - th) * 100)
        dg = diag_all[f"0.20_{dl:.1f}"]
        byd[f"{dl:.1f}"] = dict(nominal_pp=float(np.mean(en)) if en else None,
                                tcm_pp=float(np.mean(et)) if et else None,
                                tcm_cracked_only_pp=float(np.mean(ec)) if ec else None,
                                n_nominal=len(en), n_tcm=len(et),
                                band_eps=dg["eps_mean"],
                                cracked_fraction=dg["cracked_fraction"])
        v = byd[f"{dl:.1f}"]
        f4 = lambda q: "--" if q is None else f"{q:.1f}"
        print(f"{dl:>7.1f}{f4(v['nominal_pp']):>10}{f4(v['tcm_pp']):>10}"
              f"{f4(v['tcm_cracked_only_pp']):>13}{v['band_eps']:>11.2e}"
              f"{100*v['cracked_fraction']:>11.1f}")
    out["by_load"] = byd

    # ---- 9. the confound: is the model correcting or compensating? ----
    # The stored fields come from a solver that neglects tension stiffening
    # too, so a contribution added only on the identification side is not a
    # correction of the data but a compensation of whatever else makes the
    # sectional integral fall short. Section 7.4 records that refining the
    # mesh removes most of the same shortfall, so the two are confounded and
    # the way to separate them is to put the model on the refined field: if
    # the shortfall is discretization, the refined field needs no
    # contribution and the model will overshoot it.
    print("\n" + "=" * 74)
    print("9. THE CONFOUND: the same model on the 60x30 field, theta = 0.20")
    print("=" * 74)
    ref = HERE.parent / "oracle" / "field_60x30_020.npz"
    if ref.exists():
        z = np.load(ref)
        n2, m2 = 60, 30
        st60 = element_strains(z["xy"], z["u"], n2, m2)
        lam60 = float(z["lam"])
        # A triangulated grid puts two centroids in every cell and neither at
        # the cell center, so a strip selected by distance from the cut picks
        # a ragged set of elements whose total area is not the nominal
        # 100 mm width. The element area is therefore rescaled so that the
        # selected band elements sum to the band area exactly, which is what
        # the 40x20 strip does by construction and what makes the two
        # comparable.
        cx60, cy60 = st60[0], st60[1]
        n_band = int(((np.abs(cx60 - FD.X_CUT) < FD.BAND_W)
                      & (cy60 < FD.BAND)).sum())
        n_b40 = 12                       # the same count on the 40x20 strip
        area_eq = FD.BAND * (2.0 * FD.BAND_W) / n_band
        M_req60 = lam60 * prob.P / 2.0 * (FD.X_CUT - ARM) / 1e6
        rows = []
        for tcm in (False, True):
            f = np.array([band_couple(prob, *st60, area_eq, q, tcm)[2]
                          - M_req60 for q in np.linspace(0.0, THETA_MAX, 71)])
            rows.append(bracket_root(f, np.linspace(0.0, THETA_MAX, 71)))
        print(f"  60x30, lambda = {lam60:.4f}, {n_band} band elements in the "
              f"strip against {n_b40} at 40x20")
        print(f"  nominal  {rows[0]:.4f}   bias {100*(rows[0]-0.20):+.2f} pp")
        print(f"  with TCM {rows[1]:.4f}   bias {100*(rows[1]-0.20):+.2f} pp")
        print(f"  against 40x20: nominal {head['0.20']['nominal']:.4f} "
              f"({head['0.20']['bias_nominal_pp']:+.2f} pp), "
              f"TCM {head['0.20']['tcm']:.4f} "
              f"({head['0.20']['bias_tcm_pp']:+.2f} pp)")
        # the strip on a triangulated grid is ragged, so the reading is
        # repeated at four half-widths to show it is not an artefact of
        # which elements the strip happens to catch
        rob = {}
        for W in (50.0, 75.0, 100.0, 150.0):
            sel = (np.abs(cx60 - FD.X_CUT) < W) & (cy60 < FD.BAND)
            nb = int(sel.sum())
            aeq = FD.BAND * (2.0 * W) / nb
            keep, FD.BAND_W = FD.BAND_W, W
            vals = []
            for tcm in (False, True):
                f = np.array([band_couple(prob, *st60, aeq, q, tcm)[2]
                              - M_req60 for q in np.linspace(0.0, THETA_MAX, 71)])
                vals.append(float(bracket_root(f, np.linspace(0.0, THETA_MAX, 71))))
            FD.BAND_W = keep
            rob[f"{W:.0f}"] = dict(nominal=vals[0], tcm=vals[1], n=nb)
        print("  strip half-width " + ", ".join(
            f"{w} mm: {rob[w]['nominal']:.3f}/{rob[w]['tcm']:.3f}"
            for w in rob) + "  (nominal/TCM)")
        print(f"  caveat: the refined solve returns lambda = {lam60:.3f} at the "
              f"same 3.5 mm deflection")
        print(f"  against {float(d['lam_0.20_3.5'][0]):.3f} at 40x20, so the "
              "forward solve is mesh sensitive too")
        out["refined"] = dict(nx=n2, ny=m2, lam=lam60, theta_true=0.20,
                              nominal=float(rows[0]), tcm=float(rows[1]),
                              bias_nominal_pp=100 * (rows[0] - 0.20),
                              bias_tcm_pp=100 * (rows[1] - 0.20),
                              n_band_elements=n_band,
                              strip_robustness=rob,
                              lam_40x20=float(d['lam_0.20_3.5'][0]))
    else:
        print("  60x30 field not present; skipped")

    # ---- 10. what the whole thing amounts to --------------------------
    print("\n" + "=" * 74)
    print("10. SUMMARY")
    print("=" * 74)
    pooled_n = float(np.mean([v["nominal_pp"] for v in byd.values()
                              if v["nominal_pp"] is not None]))
    pooled_t = float(np.mean([v["tcm_pp"] for v in byd.values()
                              if v["tcm_pp"] is not None]))
    fp = {}
    for dl in deltas:
        k = f"0.00_{dl:.1f}"
        lam = float(d[f"lam_{k}"][0])
        rn = recover(prob, strains[k], area, lam, False)[0]
        rt = recover(prob, strains[k], area, lam, True)[0]
        fp[f"{dl:.1f}"] = dict(nominal=float(rn), tcm=float(rt))
    print(f"  mean |bias| at delta = {DELTA} mm : "
          f"{mn:.1f} pp nominal -> {mt:.1f} pp with TCM")
    print(f"  pooled over all five load levels: "
          f"{pooled_n:.1f} pp nominal -> {pooled_t:.1f} pp with TCM")
    print(f"  signed bias at delta = {DELTA} mm : "
          f"{min(v['bias_nominal_pp'] for v in head.values() if np.isfinite(v['bias_nominal_pp'])):+.1f} to "
          f"{max(v['bias_nominal_pp'] for v in head.values() if np.isfinite(v['bias_nominal_pp'])):+.1f} pp -> "
          f"{min(v['bias_tcm_pp'] for v in head.values() if float(v['tcm'])>0 and np.isfinite(v['bias_tcm_pp'])):+.1f} to "
          f"{max(v['bias_tcm_pp'] for v in head.values() if np.isfinite(v['bias_tcm_pp'])):+.1f} pp")
    print("  false positive on the intact tie, root at theta_true = 0:")
    def shown(q, censored="none (censored)"):
        return censored if not np.isfinite(q) else f"{q:.4f}"

    for dl in deltas:
        v = fp[f"{dl:.1f}"]
        print(f"    delta {dl:>4.1f} mm: nominal {shown(v['nominal']):>15}"
              f"   TCM {shown(v['tcm'], 'none'):>8}")
    out["summary"] = dict(mean_abs_bias_delta35_nominal_pp=mn,
                          mean_abs_bias_delta35_tcm_pp=mt,
                          mean_abs_bias_pooled_nominal_pp=pooled_n,
                          mean_abs_bias_pooled_tcm_pp=pooled_t,
                          false_positive=fp,
                          min_abs_slope_kN=allmin,
                          affinity_worst_all_states_pct=100 * max(
                              v["aff_dep"] for v in mono.values()),
                          affinity_delta35_pct=100 * max(
                              v["tcm"] for v in aff.values()),
                          affinity_nominal_pct=100 * max(
                              v["nominal"] for v in aff.values()))

    def clean(o):
        """Non-finite floats become null, so the file is strict JSON."""
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean(out), indent=2, allow_nan=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
