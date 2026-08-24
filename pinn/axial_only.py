"""Whether the identification survives on axial strain alone.

The deployment this study recommends is distributed fiber bonded along the
reinforcement, and that instrument returns the strain component along the
bar and nothing else. Algorithm 2 evaluates the cracked-membrane map at the
full in-plane tensor, so as written it asks for two components the
recommended instrument does not deliver. Either the algorithm is asking for
more than it needs, or the recommendation is asking for the wrong
instrument, and which one is a matter of measurement rather than of taste.

The premise to be tested is that the tie side of the identifying condition
is an eps_x quantity. rho_x enters the membrane map only through the smeared
steel term, and that term is rho_x times a stress that depends on eps_x
alone, so if steel carries essentially the whole band tension then the
transverse components should not matter there. The compression side is the
part that is easy to overlook: the lever arm comes from the centroid of the
compressive sigma_x over the full depth of the cut, and nothing guarantees
that quantity is an eps_x quantity too.

This study therefore measures the steel share first, checks how far the tie
resultant moves when the transverse components are discarded, and only then
runs the identification with the transverse components measured, zeroed, and
supplied by the forward model, over the full depth of the cut and over the
tie band alone. Everything is read from the stored reference states; no new
solve is run.

Run:  python axial_only.py       (writes figures/axial_only.json)
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

from csfm_constitutive import membrane                                     # noqa: E402
from figdata import BAND, BAND_W, NX, NY, X_CUT, band_couple, recover_band  # noqa: E402
from identify import rho_x_of_theta                                        # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import bracket_root, element_strains                    # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "axial_only.json"

DELTA = 3.5                 # the state the manuscript quotes the recovery at
ARM = 370.0                 # the assumed reaction arm, as everywhere here
RISING = (1.0, 2.0, 3.5)    # stored states, load factor increasing on this run
GRID = np.linspace(0.0, 0.70, 71)   # the trial grid recover_band uses


# ----------------------------------------------------------------------
# 1. how the band tension divides, and what eps_x alone already fixes
# ----------------------------------------------------------------------
def band_split(prob, cx, cy, ex, ey, gxy, area, theta):
    """Steel and concrete parts of the band tension resultant, in kN.

    rho_x enters the membrane map only through the smeared steel term, so
    evaluating the same strain twice, once at the true ratio and once at
    zero, splits the resultant exactly rather than approximately.
    """
    sel = (np.abs(cx - X_CUT) < BAND_W) & (cy < BAND)
    X = torch.tensor(cx[sel]).unsqueeze(-1)
    Y = torch.tensor(cy[sel]).unsqueeze(-1)
    args = (torch.tensor(ex[sel]).unsqueeze(-1),
            torch.tensor(ey[sel]).unsqueeze(-1),
            torch.tensor(gxy[sel]).unsqueeze(-1))
    rho = rho_x_of_theta(prob, X, Y, torch.tensor(float(theta)))
    dA = area / (2.0 * BAND_W) * prob.t
    tot = membrane(*args, rho, prob.rho_y(X, Y), prob.mat, soften=True)
    con = membrane(*args, torch.zeros_like(rho), prob.rho_y(X, Y), prob.mat,
                   soften=True)
    T = float((tot["sigma_x"].squeeze().numpy() * dA).sum()) / 1e3
    Tc = float((con["sigma_x"].squeeze().numpy() * dA).sum()) / 1e3
    return T, T - Tc, Tc


def compression_resultant(prob, cx, cy, ex, ey, gxy, area, theta):
    """Compressive part of sigma_x above the band, in kN, and its centroid."""
    sel = (np.abs(cx - X_CUT) < BAND_W) & (cy >= BAND)
    X = torch.tensor(cx[sel]).unsqueeze(-1)
    Y = torch.tensor(cy[sel]).unsqueeze(-1)
    st = membrane(torch.tensor(ex[sel]).unsqueeze(-1),
                  torch.tensor(ey[sel]).unsqueeze(-1),
                  torch.tensor(gxy[sel]).unsqueeze(-1),
                  rho_x_of_theta(prob, X, Y, torch.tensor(float(theta))),
                  prob.rho_y(X, Y), prob.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy()
    dA = area / (2.0 * BAND_W) * prob.t
    w = np.clip(-sx, 0.0, None)
    C = float((w * dA).sum()) / 1e3
    yC = float((w * cy[sel]).sum() / max(w.sum(), 1e-9))
    return C, yC


# ----------------------------------------------------------------------
# the forward model, standing in for what the fiber cannot reach
# ----------------------------------------------------------------------
class ModelFields:
    """Strain the forward model predicts at a trial state and a given load.

    A model asked to supply what the instrument cannot must be evaluated at
    the load the structure is under, not at its deflection: the two states
    differ once the tie has lost section, and the load is what a monitoring
    campaign records. No new solve is permitted here, so the model is a
    two-point interpolation of the stored states, first in load factor along
    each deterioration branch and then across branches. It is exact at every
    stored state, so `at(theta_true, lam_measured)` reproduces the
    measurement itself, which is the check that the surrogate is a model of
    this member and not of something else.
    """

    def __init__(self, d, xy):
        self.th = np.array([float(t) for t in d["theta_true"]])
        self.lam, self.s = {}, {}
        for t in self.th:
            self.lam[t] = np.array(
                [float(d[f"lam_{t:.2f}_{x:.1f}"][0]) for x in RISING])
            assert np.all(np.diff(self.lam[t]) > 0), self.lam[t]
            self.s[t] = [element_strains(xy, d[f"u_{t:.2f}_{x:.1f}"], NX, NY)[2:]
                         for x in RISING]

    def _at_load(self, t, lam):
        L, S = self.lam[t], self.s[t]
        i = int(np.clip(np.searchsorted(L, lam) - 1, 0, len(RISING) - 2))
        w = float(np.clip((lam - L[i]) / (L[i + 1] - L[i]), 0.0, 1.0))
        return tuple(a + w * (b - a) for a, b in zip(S[i], S[i + 1]))

    def at(self, theta, lam):
        """Strain at a trial deterioration and a measured load factor."""
        q = float(np.clip(theta, self.th[0], self.th[-1]))
        j = int(np.clip(np.searchsorted(self.th, q) - 1, 0, self.th.size - 2))
        w = (q - self.th[j]) / (self.th[j + 1] - self.th[j])
        A = self._at_load(self.th[j], lam)
        B = self._at_load(self.th[j + 1], lam)
        return tuple(a + w * (b - a) for a, b in zip(A, B))


# ----------------------------------------------------------------------
# 2 and 3. the variants: which components are known, and where
# ----------------------------------------------------------------------
VARIANTS = (
    ("a_full",        "(a) full tensor, full cut depth", False),
    ("b_axial_zero",  "(b) eps_x only, transverse zeroed, full depth", False),
    ("b_axial_band",  "(b') eps_x only, tie band only", False),
    ("full_band",     "     full tensor, tie band only", False),
    ("c_model_exact", "(c) eps_x full depth, transverse from model at trial", True),
    ("c_model_asbuilt", "(c') eps_x full depth, transverse from as-built model", False),
    ("d_band_model",  "(d) eps_x in band, model over the compression side", True),
    ("d_band_asbuilt", "(d') eps_x in band, as-built model over compression", False),
)


def strain_set(name, cy, meas, model):
    """The (eps_x, eps_y, gamma_xy) triple a given instrument hands in.

    A zero here stands for an unmeasured quantity, not for a measured zero.
    That distinction is the whole point of the band-only rows: an element
    the fiber never reaches contributes no stress at all, and the
    compression centroid is then formed over whatever is left.
    """
    ex, ey, gxy = meas
    mx, my, mg = model
    inb = cy < BAND
    z = np.zeros_like(ex)
    if name == "a_full":
        return ex, ey, gxy
    if name == "b_axial_zero":
        return ex, z, z
    if name == "b_axial_band":
        return np.where(inb, ex, 0.0), z, z
    if name == "full_band":
        return (np.where(inb, ex, 0.0), np.where(inb, ey, 0.0),
                np.where(inb, gxy, 0.0))
    if name in ("c_model_exact", "c_model_asbuilt"):
        return ex, my, mg
    if name in ("d_band_model", "d_band_asbuilt"):
        return (np.where(inb, ex, mx), np.where(inb, 0.0, my),
                np.where(inb, 0.0, mg))
    raise KeyError(name)


def recover(prob, cx, cy, meas, area, lam, name, mdl, in_loop):
    """Recovered theta for one variant, on the code path recover_band uses.

    The model-in-the-loop variants cannot call recover_band, because the
    strain handed to the constitutive map changes with the trial value. The
    trial grid, the statics target and the root rule are the ones
    recover_band uses, so the two paths differ only in that one respect,
    which is verified by running variant (a) through both.
    """
    if not in_loop:
        m = mdl.at(0.0, lam) if mdl is not None else (None, None, None)
        e = strain_set(name, cy, meas, m)
        return recover_band(prob, cx, cy, *e, area, lam, ARM)[0]
    M_req = (lam * prob.P / 2.0) * (X_CUT - ARM) / 1e6
    f = np.array([band_couple(prob, cx, cy,
                              *strain_set(name, cy, meas, mdl.at(q, lam)),
                              area, q)[2] - M_req for q in GRID])
    return bracket_root(f, GRID)


# ----------------------------------------------------------------------
def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    xy = d["xy"]
    area = (prob.L / NX) * (prob.H / NY) / 2.0
    thetas = [float(t) for t in d["theta_true"]]
    deltas = [float(t) for t in d["deltas"]]
    mdl = ModelFields(d, xy)
    out: dict = {"delta_mm": DELTA, "arm_mm": ARM, "cut_x_mm": X_CUT,
                 "band_mm": BAND, "theta_true": thetas, "deltas": deltas,
                 "variant_labels": {n: lab for n, lab, _ in VARIANTS}}

    def meas_at(th, dl=DELTA):
        return element_strains(xy, d[f"u_{th:.2f}_{dl:.1f}"], NX, NY)

    # the surrogate has to be the model of this member, not of another one
    c0, cy0, *m0 = meas_at(0.20)
    ck = mdl.at(0.20, float(d["lam_0.20_3.5"][0]))
    err = max(float(np.abs(a - b).max()) for a, b in zip(m0, ck))
    print(f"model surrogate reproduces the stored state to {err:.2e} strain")
    out["surrogate_check"] = err

    # ---- 1. how the band tension divides between the two materials ----
    print(f"\n1. band tension at the cut x = {X_CUT:.0f} mm, "
          f"delta = {DELTA} mm\n")
    print(f"{'theta':>7}{'T (kN)':>10}{'steel':>10}{'concrete':>10}"
          f"{'steel share':>13}{'T, eps_x only':>15}{'error':>9}"
          f"{'dT/dtheta':>12}")
    split = {}
    for th in thetas:
        cx, cy, ex, ey, gxy = meas_at(th)
        T, Ts, Tc = band_split(prob, cx, cy, ex, ey, gxy, area, th)
        z = np.zeros_like(ex)
        Tz = band_split(prob, cx, cy, ex, z, z, area, th)[0]
        h = 1e-4
        dT = (band_split(prob, cx, cy, ex, ey, gxy, area, th + h)[0]
              - band_split(prob, cx, cy, ex, ey, gxy, area, th - h)[0]) / (2 * h)
        dTz = (band_split(prob, cx, cy, ex, z, z, area, th + h)[0]
               - band_split(prob, cx, cy, ex, z, z, area, th - h)[0]) / (2 * h)
        split[f"{th:.2f}"] = dict(T=T, T_steel=Ts, T_concrete=Tc,
                                  steel_share=Ts / T, T_axial_only=Tz,
                                  rel_error=(Tz - T) / T,
                                  dT_dtheta=dT, dT_dtheta_axial_only=dTz)
        print(f"{th:>7.2f}{T:>10.1f}{Ts:>10.1f}{Tc:>10.1f}"
              f"{100*Ts/T:>12.1f}%{Tz:>15.1f}{100*(Tz-T)/T:>8.2f}%"
              f"{dT:>12.1f}")
    out["band_split"] = split
    dd = max(abs(v["dT_dtheta"] - v["dT_dtheta_axial_only"])
             for v in split.values())
    print(f"\n   dT/dtheta is unchanged by discarding the transverse "
          f"components to {dd:.2e} kN")

    grid = {}
    for th in thetas:
        for dl in deltas:
            if f"u_{th:.2f}_{dl:.1f}" not in d.files:
                continue
            cx, cy, ex, ey, gxy = meas_at(th, dl)
            T, Ts, Tc = band_split(prob, cx, cy, ex, ey, gxy, area, th)
            grid[f"{th:.2f}_{dl:.1f}"] = dict(T=T, steel_share=Ts / T)
    out["band_split_grid"] = grid
    lo = min(grid.items(), key=lambda kv: kv[1]["steel_share"])
    hi = max(grid.items(), key=lambda kv: kv[1]["steel_share"])
    print(f"   over every stored state the steel share runs from "
          f"{lo[1]['steel_share']:.3f} (theta,delta = {lo[0]}) to "
          f"{hi[1]['steel_share']:.3f} ({hi[0]})")

    # ---- 1b. the compression side, which is not an eps_x quantity -----
    print(f"\n2. the compression side of the same cut\n")
    print(f"{'theta':>7}{'C, full':>10}{'C, eps_x only':>15}"
          f"{'eps_x < 0 anywhere on the cut':>32}")
    comp = {}
    for th in thetas:
        cx, cy, ex, ey, gxy = meas_at(th)
        z = np.zeros_like(ex)
        C, yC = compression_resultant(prob, cx, cy, ex, ey, gxy, area, th)
        Cz, yCz = compression_resultant(prob, cx, cy, ex, z, z, area, th)
        sel = np.abs(cx - X_CUT) < BAND_W
        frac = float(np.mean(ex[sel] < 0.0))
        comp[f"{th:.2f}"] = dict(C=C, yC=yC, C_axial_only=Cz, yC_axial_only=yCz,
                                 frac_eps_x_compressive=frac,
                                 min_eps_x=float(ex[sel].min()))
        print(f"{th:>7.2f}{C:>10.1f}{Cz:>15.1f}"
              f"{100*frac:>31.0f}%")
    out["compression_side"] = comp

    # ---- 3. the identification, variant by variant --------------------
    print(f"\n3. recovered theta at delta = {DELTA} mm, no noise\n")
    print(f"{'true':>6}" + "".join(f"{n:>22}" for n, _, _ in VARIANTS))
    rec: dict = {n: {} for n, _, _ in VARIANTS}
    for th in thetas:
        lm = float(d[f"lam_{th:.2f}_{DELTA}"][0])
        cx, cy, ex, ey, gxy = meas_at(th)
        row = f"{th:>6.2f}"
        for name, _, loop in VARIANTS:
            use = mdl if "model" in name or "asbuilt" in name else None
            if name.endswith("asbuilt"):
                m = mdl.at(0.0, lm)
                r = recover_band(prob, cx, cy,
                                 *strain_set(name, cy, (ex, ey, gxy), m),
                                 area, lm, ARM)[0]
            else:
                r = recover(prob, cx, cy, (ex, ey, gxy), area, lm, name,
                            use, loop)
            rec[name][f"{th:.2f}"] = None if np.isnan(r) else float(r)
            row += ("       no root       " if np.isnan(r)
                    else f"{r:>14.4f}{100*(r-th):>+8.2f}")
        print(row)
    print(f"{'':>6}" + "".join(f"{'value  pp error':>22}" for _ in VARIANTS))
    out["recovery"] = rec

    # ---- 3b. the lever arm each variant ends up using -----------------
    # The arm is where the transverse components actually act, so it is
    # reported next to the resultant they leave alone.
    print(f"\n4. lever arm at the true theta (mm), by variant\n")
    arm = {}
    for th in thetas:
        lm = float(d[f"lam_{th:.2f}_{DELTA}"][0])
        cx, cy, ex, ey, gxy = meas_at(th)
        zs = {}
        for name, _, loop in VARIANTS:
            m = mdl.at(0.0 if name.endswith("asbuilt") else th, lm)
            T, zz, M = band_couple(prob, cx, cy,
                                   *strain_set(name, cy, (ex, ey, gxy), m),
                                   area, th)
            zs[name] = dict(T=T, z=zz, couple=M)
        arm[f"{th:.2f}"] = zs
    print(f"  {'variant':<52}" + "".join(f"{t:>8.2f}" for t in thetas))
    for name, lab, _ in VARIANTS:
        print(f"  {lab:<52}" + "".join(f"{arm[f'{t:.2f}'][name]['z']:>8.1f}"
                                       for t in thetas))
    print(f"  {'tie resultant T (kN), full tensor':<52}"
          + "".join(f"{arm[f'{t:.2f}']['a_full']['T']:>8.1f}" for t in thetas))
    print(f"  {'tie resultant T (kN), eps_x alone':<52}"
          + "".join(f"{arm[f'{t:.2f}']['b_axial_zero']['T']:>8.1f}"
                    for t in thetas))
    out["couple_at_true_theta"] = arm

    # ---- 3c. is the finding a property of the chosen cut? -------------
    # The compression side vanishes from eps_x because the horizontal
    # compression on a D-region cut is the projection of an inclined strut.
    # If that were an accident of one station, moving the cut would fix it,
    # so every station is checked rather than the one the method uses.
    print("\n4b. axial shortening anywhere on the cut, station by station\n")
    print(f"{'x (mm)':>8}{'min eps_x':>13}{'depth with eps_x < 0':>23}")
    stn = {}
    cx, cy, ex, ey, gxy = meas_at(0.20)
    for xs in (150.0, 250.0, 400.0, 550.0, 700.0, 850.0, 975.0):
        sel = np.abs(cx - xs) < BAND_W
        frac = float(np.mean(ex[sel] < 0.0))
        stn[f"{xs:.0f}"] = dict(min_eps_x=float(ex[sel].min()),
                                frac_compressive=frac)
        print(f"{xs:>8.0f}{ex[sel].min():>13.2e}{100*frac:>22.0f}%")
    out["station_scan_theta_020_delta_3p5"] = stn

    # ---- 5. does the ranking hold across the load path? ---------------
    print(f"\n5. recovered theta across the load path\n")
    keep = ("a_full", "b_axial_zero", "c_model_exact", "d_band_model")
    print(f"{'true':>6}{'delta':>7}" + "".join(f"{k.split('_')[0]:>11}"
                                               for k in keep))
    across = {}
    for th in thetas:
        for dl in deltas:
            if f"u_{th:.2f}_{dl:.1f}" not in d.files:
                continue
            lm = float(d[f"lam_{th:.2f}_{dl:.1f}"][0])
            cx, cy, ex, ey, gxy = meas_at(th, dl)
            vals = {}
            for name in keep:
                loop = dict((n, l) for n, _, l in VARIANTS)[name]
                r = recover(prob, cx, cy, (ex, ey, gxy), area, lm, name,
                            mdl, loop)
                vals[name] = None if np.isnan(r) else float(r)
            across[f"{th:.2f}_{dl:.1f}"] = vals
            print(f"{th:>6.2f}{dl:>7.1f}" + "".join(
                "        nan" if vals[k] is None else f"{vals[k]:>11.4f}"
                for k in keep))
    out["recovery_across_load_path"] = across

    # ---- 6. the affine calibration each variant implies ---------------
    print("\n6. affine calibration, fitted on the four deteriorated states\n")
    fits = {}
    xs = np.array([t for t in thetas if t > 0])
    for name, lab, _ in VARIANTS:
        ys = np.array([np.nan if rec[name][f"{t:.2f}"] is None
                       else rec[name][f"{t:.2f}"] for t in xs])
        if np.all(np.isfinite(ys)):
            s, i = np.polyfit(xs, ys, 1)
            bias = (ys - xs) * 100.0
            fits[name] = dict(slope=float(s), intercept=float(i),
                              bias_pp_min=float(bias.min()),
                              bias_pp_max=float(bias.max()))
            print(f"  {lab:<52}{s:.4f} theta {i:+.4f}   bias "
                  f"{bias.min():+.1f} to {bias.max():+.1f} pp")
        else:
            fits[name] = None
            print(f"  {lab:<52}no root: the parameter is not identified")
    out["affine_fit"] = fits

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
