"""Two-parameter identifiability: one couple per cut, one parameter per half.

The one-parameter recovery reads a single band couple on a single cut and
asks a single number of it. Whether the method scales past one parameter
is a question about sensitivity structure rather than about noise: if two
parameters press on the same observable, the inverse problem is ill
conditioned no matter how clean the measurement is. This study splits the
tie band into two independently deteriorating halves, theta1 scaling the
band for x < 1000 mm and theta2 for x >= 1000 mm, and reads one couple per
half: cut 1 at x = 700 in the left shear span, cut 2 at x = 1300 treated
by mirror symmetry. The band elements entering each couple lie entirely
inside one half, so the sensitivity matrix should come out diagonal and
the two-parameter problem should inherit the one-parameter conditioning.
That expectation is measured here, not assumed: the cross sensitivities
are reported as computed, whatever they are.

Two consequences of asymmetry are handled explicitly. First, with
theta1 != theta2 the member is asymmetric, so the two reactions are not
lambda * P / 2 each; each is read off the solved field as the assembled
internal force minus the external load at the fixed vertical degrees of
freedom of its support. Second, each reaction acts at its contact
centroid rather than at the plate center: a = 370 mm for the left cut
and, by symmetry, a = 1630 mm for the right.

Run:  python two_parameter.py
      (verifies the machinery on the uniform theta = 0.20 field first,
       solves the asymmetric field if ../oracle/field_asym.npz is absent,
       then prints the table and writes ../figures/two_parameter.json)
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

from arclength_oracle import (Material, Problem, build_mesh,               # noqa: E402
                              membrane as membrane_np)
from arclength_oracle_crisfield import newton_displacement_control        # noqa: E402
from csfm_constitutive import membrane                                     # noqa: E402
from oracle_rho_sweep import RHO_NOM                                       # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains, bracket_root                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
ASYM = HERE.parent / "oracle" / "field_asym.npz"
OUT = HERE.parent / "figures" / "two_parameter.json"

NX, NY = 40, 20
BAND = 150.0               # tie-band depth (mm)
BAND_W = 50.0              # half-width of the strip that stands for a cut
X_SPLIT = 1000.0           # theta1 acts at x < X_SPLIT, theta2 at x >= X_SPLIT
CUTS = (700.0, 1300.0)     # cut i pairs with theta_i by construction
ARMS = (370.0, 1630.0)     # reaction contact centroids, left and right
THETA_TRUE = (0.30, 0.10)
DELTA = 3.5                # load level (mm prescribed patch deflection)
N_STEPS = 28               # 8 steps per mm, as in generate_fields.py
H_FD = 0.02                # central-difference step in theta
THETA_MAX = 0.70


# ----------------------------------------------------------------------
# the two-parameter reinforcement field
# ----------------------------------------------------------------------
def rho_x_two(prob: DeepBeam, x, y, th1: float, th2: float):
    """Smeared horizontal ratio with the band halves scaled independently.

    Mirrors identify.rho_x_of_theta except that the deterioration is
    theta1 left of X_SPLIT and theta2 right of it. The mesh floor is a
    property of the detailing and is left alone.
    """
    in_band = (y < prob.band).to(x.dtype)
    left = (x < X_SPLIT).to(x.dtype)
    theta = th1 * left + th2 * (1.0 - left)
    rho_tie = prob.rho_tie * (1.0 - theta)
    return prob.rho_min + (rho_tie - prob.rho_min) * in_band


def deepbeam_two(th1: float, th2: float, P_ref: float = 800.0e3) -> Problem:
    """oracle_rho_sweep.deepbeam_rho with the tie split at midspan.

    build_mesh bakes per-element (rx, ry) into mesh.tris at build time by
    evaluating prob.rho_x at each cell centroid, so making rho_x depend on
    x here is all it takes for the solver to see the asymmetric member.
    No cell centroid sits on x = 1000, so the split is unambiguous.
    """
    L, H, t = 2000.0, 1000.0, 300.0
    a, bearing = 250.0, 200.0
    rho_stirrup, rho_min = 0.0015, 0.0010

    def rho_x(x: float, y: float) -> float:
        if y >= BAND:
            return rho_min
        return RHO_NOM * (1.0 - (th1 if x < X_SPLIT else th2))

    def rho_y(x: float, y: float) -> float:
        return rho_min + rho_stirrup

    return Problem(L=L, H=H, thickness=t, nx=NX, ny=NY,
                   rho_x=rho_x, rho_y=rho_y,
                   x_load=L / 2.0, bearing=bearing, P_ref=P_ref,
                   supports=((a, True, True), (L - a, False, True)),
                   mat=Material(fc=30.0))


# ----------------------------------------------------------------------
# reactions from the solved field
# ----------------------------------------------------------------------
def internal_forces_two(u: np.ndarray, prob: Problem, mesh,
                        th1: float, th2: float) -> np.ndarray:
    """Assembled internal force vector at a trial (theta1, theta2).

    recover_nodal.internal_forces with the band ratio depending on the
    element centroid x as well as y. The assembly is the solver's own, so
    at the true pair F_int balances lambda * F_ref to solver tolerance.
    """
    F = np.zeros(mesh.ndof)
    t = prob.thickness
    rho_min = 0.0010
    for e, (nodes, rx, ry) in enumerate(mesh.tris):
        B = mesh.B[e]
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        eps = B @ u[dofs]
        xc = mesh.xy[list(nodes), 0].mean()
        yc = mesh.xy[list(nodes), 1].mean()
        if yc < BAND:
            rx_t = RHO_NOM * (1.0 - (th1 if xc < X_SPLIT else th2))
        else:
            rx_t = rho_min
        sig, _, _ = membrane_np(eps[0], eps[1], eps[2], rx_t, ry, prob.mat)
        F[dofs] += (B.T @ np.asarray(sig).ravel()) * mesh.area[e] * t
    return F


def support_reactions(u: np.ndarray, prob: Problem, mesh, lam: float,
                      th1: float, th2: float) -> tuple[float, float]:
    """Vertical reactions in kN, read off the solved field.

    R = F_int - lambda * F_ref summed over the fixed vertical degrees of
    freedom of each support. mesh.fixed is a boolean mask over dofs, not
    an index list, and mesh.load_dofs holds node indices; both conventions
    have bitten before and are respected here.
    """
    Rv = internal_forces_two(u, prob, mesh, th1, th2) - lam * mesh.F_ref
    fixed = np.asarray(mesh.fixed, dtype=bool)
    r_left = r_right = 0.0
    for n in range(mesh.n_node):
        if not fixed[2 * n + 1]:
            continue
        if mesh.xy[n, 0] < 600.0:
            r_left += Rv[2 * n + 1]
        elif mesh.xy[n, 0] > 1400.0:
            r_right += Rv[2 * n + 1]
    return r_left / 1e3, r_right / 1e3


# ----------------------------------------------------------------------
# the observable: band couple on a cut, two-parameter version
# ----------------------------------------------------------------------
def band_couple_two(prob: DeepBeam, cx, cy, ex, ey, gxy, area,
                    x_cut: float, th1: float, th2: float):
    """Tie resultant, lever arm and their couple at one cut (figdata's
    band_couple with the two-parameter reinforcement field).

    The band supplies T; axial equilibrium of the cut supplies the fact
    that the compression resultant equals it; the arm is the distance
    between the two centroids. The parameters enter only through the
    constitutive split of the band stress into steel and concrete shares.
    """
    sel = np.abs(cx - x_cut) < BAND_W
    X = torch.tensor(cx[sel]).unsqueeze(-1)
    Y = torch.tensor(cy[sel]).unsqueeze(-1)
    st = membrane(torch.tensor(ex[sel]).unsqueeze(-1),
                  torch.tensor(ey[sel]).unsqueeze(-1),
                  torch.tensor(gxy[sel]).unsqueeze(-1),
                  rho_x_two(prob, X, Y, th1, th2),
                  prob.rho_y(X, Y), prob.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy()
    ys = cy[sel]
    dA = area / (2.0 * BAND_W) * prob.t
    inb = ys < BAND
    T = float((sx[inb] * dA).sum()) / 1e3                            # kN
    wT = np.clip(sx[inb], 0.0, None)
    yT = float((wT * ys[inb]).sum() / max(wT.sum(), 1e-9))
    wC = np.clip(-sx[~inb], 0.0, None)
    yC = float((wC * ys[~inb]).sum() / max(wC.sum(), 1e-9))
    z = yC - yT
    return T, z, T * z / 1e3                                         # kN m


def moment_demands(R_left_kN: float, R_right_kN: float) -> np.ndarray:
    """Statics moment each cut must transmit, in kN m, both positive.

    Cut 1: the free body left of x = 700 carries only the left reaction,
    acting at its contact centroid a = 370. Cut 2 is the mirror: the free
    body right of x = 1300 carries only the right reaction at a = 1630,
    so the demand is R_right * (a_right - x_cut), positive like the left.
    """
    return np.array([R_left_kN * (CUTS[0] - ARMS[0]) / 1e3,
                     R_right_kN * (ARMS[1] - CUTS[1]) / 1e3])


# ----------------------------------------------------------------------
# sensitivity, recovery, posterior
# ----------------------------------------------------------------------
class CoupleSystem:
    """C(theta1, theta2) on a fixed measured field, and its Jacobian."""

    def __init__(self, prob: DeepBeam, xy: np.ndarray, u: np.ndarray):
        self.prob = prob
        self.cx, self.cy, self.ex, self.ey, self.gxy = \
            element_strains(xy, u, NX, NY)
        self.area = (prob.L / NX) * (prob.H / NY) / 2.0

    def couples(self, th: np.ndarray) -> np.ndarray:
        return np.array([band_couple_two(self.prob, self.cx, self.cy,
                                         self.ex, self.ey, self.gxy,
                                         self.area, c, th[0], th[1])[2]
                         for c in CUTS])

    def jacobian(self, th: np.ndarray, h: float = H_FD) -> np.ndarray:
        J = np.zeros((2, 2))
        for j in range(2):
            tp, tm = th.copy(), th.copy()
            tp[j] += h
            tm[j] -= h
            J[:, j] = (self.couples(tp) - self.couples(tm)) / (2.0 * h)
        return J

    def recover_grid(self, M_req: np.ndarray) -> np.ndarray:
        """Per-cut sign-change roots on a theta grid (figdata's method).

        Exact decoupling makes each cut a one-dimensional problem in its
        own parameter; the grid scan is the robustness cross-check for
        the Newton iteration, not the primary solver.
        """
        grid = np.linspace(0.0, THETA_MAX, 71)
        roots = np.full(2, np.nan)
        for i in range(2):
            f = []
            for g in grid:
                th = np.array([g, g])
                f.append(band_couple_two(self.prob, self.cx, self.cy,
                                         self.ex, self.ey, self.gxy,
                                         self.area, CUTS[i], th[0], th[1])[2]
                         - M_req[i])
            f = np.array(f)
            s = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
            if len(s):
                roots[i] = bracket_root(f, grid)
        return roots

    def recover_newton(self, M_req: np.ndarray,
                       th0=(0.20, 0.20), tol: float = 1e-8,
                       max_iter: int = 60) -> tuple[np.ndarray, float]:
        th = np.array(th0, dtype=float)
        for _ in range(max_iter):
            r = self.couples(th) - M_req
            if np.max(np.abs(r)) < tol:
                break
            J = self.jacobian(th)
            th = np.clip(th - np.linalg.solve(J, r), 0.0, THETA_MAX)
        return th, float(np.max(np.abs(self.couples(th) - M_req)))


# ----------------------------------------------------------------------
def solve_asymmetric() -> None:
    """Generate the asymmetric reference field and save it with its
    reactions. A few minutes of consistent-tangent Newton."""
    th1, th2 = THETA_TRUE
    prob = deepbeam_two(th1, th2)
    mesh = build_mesh(prob)
    print(f"solving asymmetric field theta=({th1:.2f}, {th2:.2f}), "
          f"delta={DELTA} mm, {N_STEPS} steps ...", flush=True)
    t0 = time.time()
    hist = newton_displacement_control(prob, mesh, delta_max=DELTA,
                                       n_steps=N_STEPS, verbose=True)
    last = hist[-1]
    if not last.converged:
        raise RuntimeError(
            f"asymmetric solve did not converge: resid={last.resid:.2e} "
            f"at delta={last.delta:.2f} mm; not saving the field")
    r_l, r_r = support_reactions(np.asarray(last.u), prob, mesh,
                                 last.lam, th1, th2)
    np.savez(ASYM,
             u=np.asarray(last.u).reshape(-1, 2),
             lam=np.array([last.lam]),
             delta=np.array([last.delta]),
             resid=np.array([last.resid]),
             theta_true=np.array(THETA_TRUE),
             R_left_kN=np.array([r_l]),
             R_right_kN=np.array([r_r]),
             xy=mesh.xy)
    print(f"  lam={last.lam:.4f}  resid={last.resid:.2e}  "
          f"R=({r_l:.1f}, {r_r:.1f}) kN  [{time.time() - t0:.0f} s]",
          flush=True)


def verify_uniform() -> dict:
    """Step 0: the two-cut machinery on the known uniform field.

    theta1 = theta2 free on the uniform theta = 0.20, delta = 3.5 field;
    both cuts must land near 0.16, the one-parameter result at arm 370.
    """
    d = np.load(FIELDS)
    prob = DeepBeam()
    u = d["u_0.20_3.5"]
    lam = float(d["lam_0.20_3.5"][0])
    p2 = deepbeam_two(0.20, 0.20)
    mesh = build_mesh(p2)
    r_l, r_r = support_reactions(u.ravel(), p2, mesh, lam, 0.20, 0.20)
    sys_ = CoupleSystem(prob, d["xy"], u)
    M_req = moment_demands(r_l, r_r)
    rec = sys_.recover_grid(M_req)
    half = lam * prob.P / 2.0 / 1e3
    print("\n== verification on the uniform theta = 0.20 field ==")
    print(f"  lam = {lam:.4f}   R_left = {r_l:.1f} kN   R_right = {r_r:.1f} kN"
          f"   (lam P/2 = {half:.1f} kN)")
    print(f"  M_req = ({M_req[0]:.2f}, {M_req[1]:.2f}) kN m")
    print(f"  recovered theta = ({rec[0]:.3f}, {rec[1]:.3f})   "
          f"expected about 0.16 at both cuts")
    if not (np.all(np.isfinite(rec)) and np.all(np.abs(rec - 0.16) < 0.02)):
        raise RuntimeError(
            f"machinery check failed: recovered {rec} against the known "
            f"one-parameter result of about 0.16; fix before generating")
    return {"lam": lam, "R_left_kN": r_l, "R_right_kN": r_r,
            "R_halfP_kN": half,
            "M_req_kNm": M_req.tolist(), "recovered": rec.tolist()}


def main() -> None:
    check = verify_uniform()

    if not ASYM.exists():
        solve_asymmetric()
    d = np.load(ASYM)
    th_true = np.array(THETA_TRUE)
    prob = DeepBeam()
    u = d["u"]
    lam = float(d["lam"][0])
    r_l, r_r = float(d["R_left_kN"][0]), float(d["R_right_kN"][0])
    print("\n== asymmetric field theta = (0.30, 0.10), delta = 3.5 mm ==")
    print(f"  lam = {lam:.4f}   resid = {float(d['resid'][0]):.2e}")
    print(f"  R_left = {r_l:.1f} kN   R_right = {r_r:.1f} kN   "
          f"sum = {r_l + r_r:.1f} kN   lam P = {lam * prob.P / 1e3:.1f} kN")

    sys_ = CoupleSystem(prob, d["xy"], u)

    # ---- sensitivity matrix at the true pair -------------------------
    J = sys_.jacobian(th_true)
    Jn = J / np.linalg.norm(J, axis=1, keepdims=True)
    cond = float(np.linalg.cond(Jn))
    off = np.array([abs(J[0, 1]) / abs(J[0, 0]),
                    abs(J[1, 0]) / abs(J[1, 1])])
    print("\n== sensitivity J_ij = dC_i / dtheta_j (kN m per unit theta) ==")
    print(f"  J = [[{J[0, 0]:+9.3f}, {J[0, 1]:+9.3f}],")
    print(f"       [{J[1, 0]:+9.3f}, {J[1, 1]:+9.3f}]]")
    print(f"  row-normalized condition number = {cond:.4f}")
    print(f"  off-diagonal ratios |J12|/|J11| = {off[0]:.2e},  "
          f"|J21|/|J22| = {off[1]:.2e}")
    print(f"  measured dC1/dtheta2 = {J[0, 1]:+.3e} kN m  "
          f"(cut-1 band elements all at x < 1000, so ~0 was expected)")

    # ---- two-cut recovery --------------------------------------------
    M_req = moment_demands(r_l, r_r)
    th_hat, res = sys_.recover_newton(M_req)
    th_grid = sys_.recover_grid(M_req)
    err_pp = (th_hat - th_true) * 100.0
    print("\n== two-cut recovery ==")
    print(f"  M_req = ({M_req[0]:.2f}, {M_req[1]:.2f}) kN m")
    print(f"  {'':>10}{'true':>8}{'newton':>9}{'grid':>8}{'err (pp)':>10}")
    for i in (0, 1):
        print(f"  theta_{i + 1:<4}{th_true[i]:>8.3f}{th_hat[i]:>9.3f}"
              f"{th_grid[i]:>8.3f}{err_pp[i]:>+10.1f}")
    print(f"  newton residual = {res:.2e} kN m")

    # ---- one-cut null direction ----------------------------------------
    J1 = J[0:1, :]                                    # cut 1 only, 1 x 2
    _, _, Vt = np.linalg.svd(J1)
    null = Vt[-1]
    if null[1] < 0:
        null = -null
    print("\n== one-cut case (cut 1 only) ==")
    print(f"  sensitivity row [dC1/dtheta1, dC1/dtheta2] = "
          f"[{J1[0, 0]:+.3f}, {J1[0, 1]:+.3f}] kN m")
    print(f"  null direction = ({null[0]:+.4f}, {null[1]:+.4f}): "
          f"theta2 is invisible, because every band element entering the "
          f"cut-1 couple lies at x < 1000, outside theta2's support.")

    # ---- posterior correlation ----------------------------------------
    sigma_C = 0.025 * abs(J[0, 0])
    cov = sigma_C ** 2 * np.linalg.inv(J.T @ J)
    corr = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    sig_th = np.sqrt(np.diag(cov))
    print("\n== posterior, couple noise sigma_C = 0.025 |J11| per cut ==")
    print(f"  sigma_C = {sigma_C:.3f} kN m")
    print(f"  sigma_theta = ({sig_th[0]:.4f}, {sig_th[1]:.4f})")
    print(f"  correlation(theta1_hat, theta2_hat) = {corr:+.4f}")

    out = {
        "uniform_check": check,
        "theta_true": th_true.tolist(),
        "delta_mm": DELTA,
        "lam": lam,
        "R_left_kN": r_l, "R_right_kN": r_r,
        "M_req_kNm": M_req.tolist(),
        "J_kNm": J.tolist(),
        "condition_number_rownorm": cond,
        "offdiag_ratios": off.tolist(),
        "theta_hat_newton": th_hat.tolist(),
        "theta_hat_grid": th_grid.tolist(),
        "error_pp": err_pp.tolist(),
        "one_cut_null_direction": null.tolist(),
        "sigma_C_kNm": sigma_C,
        "sigma_theta": sig_th.tolist(),
        "correlation": corr,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
