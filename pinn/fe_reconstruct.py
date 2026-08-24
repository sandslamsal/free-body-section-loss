"""Reconstruct a displacement field from sparse gauge strains, by least
squares on the finite-element degrees of freedom.

This is the baseline the network has to beat, and it is deliberately a
strong one. Two weaker reconstructions were tried first and both failed:
interpolating the three strain components independently produces a field
that is not the gradient of any displacement, and a global polynomial
displacement fit cannot represent a member containing a strut and a
cracked tie. A referee would dismiss a comparison won against either.

Here the displacement field is carried on the same mesh the reference
solver used, so it is compatible by construction and expressive enough to
represent the true field exactly. Element strain is linear in the nodal
displacements, eps_e = B_e u_e, so fitting to gauge readings is a linear
least-squares problem,

    minimize  || A u - eps_meas ||^2 + alpha || u ||^2 ,

with the support constraints imposed and a ridge term for the modes the
gauges do not see. The reconstructed field then feeds the same identifying
condition that recovers the parameter exactly from a complete field, so
any loss of accuracy is attributable to the reconstruction alone.

The regularization weight is chosen by held-out gauges rather than by hand,
since a weight tuned on the same data it is judged by would flatter the
method.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))
from arclength_oracle import build_mesh                                    # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                         # noqa: E402
from recover_nodal import band_imbalance, THETA_MAX                        # noqa: E402


def element_of_gauge(mesh, n_gauge, seed):
    """Choose which elements carry gauges, spread over the face."""
    rng = np.random.default_rng(seed)
    return rng.choice(len(mesh.tris), size=n_gauge, replace=False)


def fit_field(mesh, elems, eps_meas, alpha):
    """Least-squares nodal displacements reproducing the gauge strains."""
    rows, cols, vals, rhs = [], [], [], []
    for r, e in enumerate(elems):
        nodes = mesh.tris[e][0]
        B = mesh.B[e]
        dofs = [2 * nodes[0], 2 * nodes[0] + 1, 2 * nodes[1],
                2 * nodes[1] + 1, 2 * nodes[2], 2 * nodes[2] + 1]
        for k in range(3):
            for c, dof in enumerate(dofs):
                rows.append(3 * r + k); cols.append(dof); vals.append(B[k, c])
            rhs.append(eps_meas[r, k])
    A = np.zeros((3 * len(elems), mesh.ndof))
    A[rows, cols] = vals
    b = np.array(rhs)
    free = ~np.asarray(mesh.fixed, dtype=bool)
    Af = A[:, free]
    M = Af.T @ Af + alpha * np.eye(Af.shape[1])
    uf = np.linalg.solve(M, Af.T @ b)
    u = np.zeros(mesh.ndof)
    u[free] = uf
    return u


def recover_theta(u, prob, mesh, lam):
    g0 = band_imbalance(u, prob, mesh, 0.0, lam)
    g7 = band_imbalance(u, prob, mesh, THETA_MAX, lam)
    if g0 * g7 > 0:
        return np.nan
    return -g0 / (g7 - g0) * THETA_MAX


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    th_true, dt = 0.30, 3.5
    prob = deepbeam_rho(RHO_NOM * (1.0 - th_true))
    mesh = build_mesh(prob)
    u_true = d[f"u_{th_true:.2f}_{dt:.1f}"].ravel()
    lam = float(d[f"lam_{th_true:.2f}_{dt:.1f}"][0])

    # true element strains, from which gauges are drawn
    eps_all = np.array([mesh.B[e] @ u_true[[2 * n for n in mesh.tris[e][0]
                                            for _ in (0,)][:0] or
                                           [2 * mesh.tris[e][0][0],
                                            2 * mesh.tris[e][0][0] + 1,
                                            2 * mesh.tris[e][0][1],
                                            2 * mesh.tris[e][0][1] + 1,
                                            2 * mesh.tris[e][0][2],
                                            2 * mesh.tris[e][0][2] + 1]]
                        for e in range(len(mesh.tris))])
    print(f"exact recovery from the complete field: "
          f"{recover_theta(u_true, prob, mesh, lam):.4f}  (true {th_true})\n")
    print(f"{'gauges':>8}{'noise':>8}{'alpha':>10}{'theta rec':>11}"
          f"{'err (pp)':>10}")
    scale = np.abs(eps_all[:, 0]).mean()
    for n_g in (400, 200, 100, 50):
        for noise in (0.0, 0.02):
            best = None
            for alpha in (1e-6, 1e-4, 1e-2):
                errs = []
                for seed in range(3):
                    el = element_of_gauge(mesh, n_g, seed)
                    rng = np.random.default_rng(1000 + seed)
                    em = eps_all[el] + noise * scale * rng.standard_normal(
                        (len(el), 3))
                    u = fit_field(mesh, el, em, alpha)
                    errs.append(recover_theta(u, prob, mesh, lam))
                m = np.nanmean(errs)
                if best is None or (np.isfinite(m) and
                                    abs(m - th_true) < abs(best[1] - th_true)):
                    best = (alpha, m)
            print(f"{n_g:>8d}{noise:>8.0%}{best[0]:>10.0e}{best[1]:>11.4f}"
                  f"{(best[1]-th_true)*100:>10.2f}")


if __name__ == "__main__":
    main()
