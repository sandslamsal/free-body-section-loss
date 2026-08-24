"""The band-strain identification, rerun on the quadrilateral fields.

Section 6.3 attributes the residual bias to the discretization, on the
evidence that the quadrilateral halves the gap between the measured tension
resultant and statics. That is a prediction about the recovered parameter,
and this script tests it: the same reconciliation, the same cut, the same
grid of states, on fields generated with four-node quadrilaterals at
2 by 2 Gauss instead of constant-strain triangles.

Run:  python q4_recovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from recover_utils import bracket_root

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import q4_oracle as Q                                                      # noqa: E402
from arclength_oracle import Material, membrane                            # noqa: E402

FIELDS = HERE.parent / "oracle" / "q4_fields_theta.npz"
CST = HERE.parent / "oracle" / "fields_theta.npz"

L, H, T_THK = 2000.0, 1000.0, 300.0
NX, NY = 40, 20
BAND, RHO_NOM, RHO_MIN, RHO_Y = 150.0, 0.012, 0.0010, 0.0025
X_CUT, P_REF, A_NOM = 700.0, 800.0e3, 250.0


def cell_state(mesh, u, theta, mat):
    """Strain and horizontal stress at every element center."""
    cx, cy, sx = [], [], []
    for q in mesh.quads:
        xe = mesh.xy[q]
        dofs = np.array([d for n in q for d in (2 * n, 2 * n + 1)])
        dNxy, _ = Q.shape_derivs(xe, 0.0, 0.0)
        eps = Q.b_matrix(dNxy) @ u[dofs]
        yc = xe[:, 1].mean()
        rx = RHO_NOM * (1.0 - theta) if yc < BAND else RHO_MIN
        sig, _, _ = membrane(eps[0], eps[1], eps[2], rx, RHO_Y, mat)
        cx.append(xe[:, 0].mean()); cy.append(yc)
        sx.append(float(np.asarray(sig).ravel()[0]))
    return np.array(cx), np.array(cy), np.array(sx)


def reaction_centroid(mesh, u, lam, mat, theta):
    """Where the bearing reaction acts, on this discretization."""
    F, _ = Q.assemble(u, mesh, RHO_NOM * (1.0 - theta), mat, T_THK)
    R = F - lam * mesh.F_ref
    fx = np.asarray(mesh.fixed, bool)
    xs, rs = [], []
    for n in range(mesh.xy.shape[0]):
        if fx[2 * n + 1] and mesh.xy[n, 0] < 600.0:
            xs.append(mesh.xy[n, 0]); rs.append(R[2 * n + 1])
    xs, rs = np.array(xs), np.array(rs)
    return float((xs * rs).sum() / rs.sum())


def couple(mesh, u, theta, mat):
    """Tie resultant, arm and couple on the cut, from band strain."""
    cx, cy, sx = cell_state(mesh, u, theta, mat)
    sel = np.abs(cx - X_CUT) < (L / NX)
    s, y = sx[sel], cy[sel]
    dA = (H / NY) * T_THK / 2.0
    inb = y < BAND
    Tf = float((s[inb] * dA).sum())
    wT = np.clip(s[inb], 0.0, None)
    wC = np.clip(-s[~inb], 0.0, None)
    yT = float((wT * y[inb]).sum() / max(wT.sum(), 1e-9))
    yC = float((wC * y[~inb]).sum() / max(wC.sum(), 1e-9))
    return Tf / 1e3, (yC - yT), Tf * (yC - yT) / 1e6        # kN, mm, kN m


def main() -> None:
    d = np.load(FIELDS)
    mat = Material(fc=30.0)
    mesh = Q.build_q4(L, H, NX, NY, A_NOM, 200.0, P_REF)
    thetas = [float(t) for t in d["theta_true"]]
    deltas = [float(t) for t in d["deltas"]]
    grid = np.linspace(0.0, 0.70, 71)

    print("Band-strain recovery on quadrilateral fields\n")
    print(f"{'theta':>7}{'delta':>7}{'a (mm)':>9}{'T (kN)':>9}{'z (mm)':>9}"
          f"{'couple':>9}{'needed':>9}{'gap %':>8}{'rec':>8}{'err pp':>8}")
    rows = {}
    for th in thetas:
        for dl in deltas:
            k = f"u_{th:.2f}_{dl:.1f}"
            if k not in d.files:
                continue
            u = d[k].ravel()
            lam = float(np.atleast_1d(d[f"lam_{th:.2f}_{dl:.1f}"])[0])
            a = reaction_centroid(mesh, u, lam, mat, th)
            M_req = lam * P_REF / 2.0 * (X_CUT - a) / 1e6
            Tf, z, C = couple(mesh, u, th, mat)
            f = np.array([couple(mesh, u, g, mat)[2] - M_req for g in grid])
            rec = bracket_root(f, grid)
            gap = 100.0 * (C - M_req) / M_req
            rows[(th, dl)] = rec
            print(f"{th:>7.2f}{dl:>7.1f}{a:>9.1f}{Tf:>9.1f}{z:>9.1f}"
                  f"{C:>9.1f}{M_req:>9.1f}{gap:>8.1f}"
                  f"{('none' if np.isnan(rec) else f'{rec:.3f}'):>8}"
                  f"{('--' if np.isnan(rec) else f'{100*(rec-th):+.1f}'):>8}")

    ok = [(k, v) for k, v in rows.items() if np.isfinite(v) and k[0] > 0]
    if ok:
        bias = np.array([abs(v - k[0]) * 100 for k, v in ok])
        print(f"\nquadrilateral: mean |bias| {bias.mean():.1f} pp, "
              f"range {bias.min():.1f} to {bias.max():.1f} pp over "
              f"{len(ok)} states")
    print("triangle, same states, from Table 2: 2.0 to 7.0 pp")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------
# the exact form of Instance A, on this discretization
# ----------------------------------------------------------------------
def band_imbalance(mesh, u, theta, lam, mat, x_cut=X_CUT):
    """Error in the tie force transmitted across a cut, in kN.

    The quadrilateral counterpart of the triangle version: same one-sided
    band sum, same exclusion of constrained degrees of freedom, evaluated
    on the assembly that generated the field.
    """
    F, _ = Q.assemble(u, mesh, RHO_NOM * (1.0 - theta), mat, T_THK)
    R = F - lam * mesh.F_ref
    fx = np.asarray(mesh.fixed, bool)
    tot = 0.0
    for n in range(mesh.xy.shape[0]):
        if (mesh.xy[n, 1] < BAND and mesh.xy[n, 0] < x_cut
                and not fx[2 * n]):
            tot += R[2 * n]
    return tot / 1e3


def exact_instance_a() -> None:
    """Recover theta from the assembled nodal forces, quadrilateral fields."""
    d = np.load(FIELDS)
    mat = Material(fc=30.0)
    mesh = Q.build_q4(L, H, NX, NY, A_NOM, 200.0, P_REF)
    grid = np.linspace(0.0, 0.70, 71)
    print("\n\nExact form (assembled nodal forces) on quadrilateral fields\n")
    print(f"{'theta':>7}{'delta':>7}{'recovered':>11}{'error pp':>10}")
    errs = []
    for th in [float(t) for t in d["theta_true"]]:
        for dl in [float(t) for t in d["deltas"]]:
            k = f"u_{th:.2f}_{dl:.1f}"
            if k not in d.files:
                continue
            u = d[k].ravel()
            lam = float(np.atleast_1d(d[f"lam_{th:.2f}_{dl:.1f}"])[0])
            f = np.array([band_imbalance(mesh, u, g, lam, mat) for g in grid])
            s = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
            if not len(s):
                print(f"{th:>7.2f}{dl:>7.1f}{'none':>11}{'--':>10}")
                continue
            r = bracket_root(f, grid)
            if not np.isfinite(r):
                continue
            errs.append(abs(r - th) * 100)
            print(f"{th:>7.2f}{dl:>7.1f}{r:>11.4f}{100*(r-th):>+10.3f}")
    if errs:
        print(f"\nmean |error| {np.mean(errs):.3f} pp over {len(errs)} states")
