"""Identification against assembled internal forces.

An earlier formulation cut the member and integrated element stresses
along the cut. Diagnostics showed that this cannot work on a
constant-strain discretization: the shear recovered the reaction to within
four per cent, but the net axial force, which must vanish on any vertical
cut of a member carrying no horizontal load, came out at 23 to 86 kN, and
the transmitted moment carried a constant offset of about 62 kN m. The
reason is that constant-strain triangles converge equilibrium in the
assembled nodal sense the solver drives to a small residual, and that is
not the same as pointwise sectional equilibrium; the axial resultant is a
cancellation of some 900 kN of tension against as much compression, so a
few per cent of element error survives as a large apparent force.

The remedy is to reconcile against the quantity the solver actually
converged. At the true deterioration the internal force vector balances the
applied load,

    F_int(u, theta_true) = F_ext ,

to solver tolerance. Evaluating the same assembly at a trial theta breaks
that balance, and the imbalance is supported only where theta acts, namely
the nodes of the tie band. The identifying condition is therefore

    g(theta) = sum over band nodes LEFT OF A CUT of
               [ F_int(u, theta) - F_ext ]_x = 0 ,

the restriction to one side being essential. Summed over the whole band
the condition is identically zero for every theta: a wrong reinforcement
ratio misstates the tie force by some amount, and the resulting nodal
imbalance appears as equal and opposite end forces which cancel in the
total. Truncating at a cut retains one of them, and the sum is then the
error in the tie force transmitted across that cut,

which is exact on the discretization rather than approximate, and inherits
the properties already established: F_int depends on theta only through the
factor rho_x(theta) multiplying the steel stress, so g is affine in theta
and its root is unique.

A further benefit is that the assembly used here is the one that generated
the fields, so the two per cent discrepancy between the two constitutive
implementations does not enter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oracle"))
from arclength_oracle import build_mesh, membrane                          # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                         # noqa: E402

THETA_MAX = 0.70


def internal_forces(u: np.ndarray, prob, mesh, theta: float) -> np.ndarray:
    """Assembled internal force vector at a trial deterioration.

    Mirrors the solver's own assembly, with the tie-band reinforcement
    ratio replaced by its deteriorated value. Everything else, including
    the element geometry and the constitutive routine, is the solver's.
    """
    F = np.zeros(mesh.ndof)
    t = prob.thickness
    band, rho_min = 150.0, 0.0010
    for e, (nodes, rx, ry) in enumerate(mesh.tris):
        B = mesh.B[e]
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        eps = B @ u[dofs]
        yc = mesh.xy[list(nodes), 1].mean()
        rx_t = RHO_NOM * (1.0 - theta) if yc < band else rho_min
        sig, _, _ = membrane(eps[0], eps[1], eps[2], rx_t, ry, prob.mat)
        F[dofs] += (B.T @ np.asarray(sig).ravel()) * mesh.area[e] * t
    return F


def band_imbalance(u, prob, mesh, theta, lam, x_cut=700.0) -> float:
    """Error in the tie force transmitted across a cut, in kN.

    Summed over band nodes on one side of x_cut. The one-sided restriction
    is what makes the functional see theta at all: over the whole band the
    imbalance sums to zero identically, since the misstated tie force
    appears as equal and opposite end actions.
    """
    F_int = internal_forces(u, prob, mesh, theta)
    F_ext = lam * mesh.F_ref
    R = F_int - F_ext
    # Constrained degrees of freedom are excluded. At a fixed DOF the
    # quantity F_int - F_ext is the support reaction rather than a
    # residual, and it depends on theta, so including it adds an
    # unbalanced force to the sum and biases the recovered value.
    # mesh.fixed is a boolean mask over degrees of freedom, not a list of
    # indices. The support bearing occupies nodes at y = 0 between
    # x = 150 and 350, which lies inside the summation region, so its
    # horizontal reaction would otherwise be counted as though it were a
    # residual and would bias every recovered value.
    fixed = np.asarray(mesh.fixed, dtype=bool)
    nodes = [n for n in range(mesh.n_node)
             if mesh.xy[n, 1] < 150.0 and mesh.xy[n, 0] < x_cut
             and not fixed[2 * n]]
    return float(sum(R[2 * n] for n in nodes)) / 1e3


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    print("Recovery against assembled internal forces\n")
    print(f"{'delta':>6}{'theta true':>11}{'g(0)':>10}{'g(0.7)':>10}"
          f"{'theta rec':>11}{'error':>9}")
    errs = {}
    for dt in d["deltas"]:
        e_list = []
        for th in d["theta_true"]:
            key = f"{th:.2f}_{dt:.1f}"
            if f"u_{key}" not in d:
                continue
            u = d[f"u_{key}"].ravel()
            lam = float(d[f"lam_{key}"][0])
            prob = deepbeam_rho(RHO_NOM * (1.0 - float(th)))
            mesh = build_mesh(prob)
            g0 = band_imbalance(u, prob, mesh, 0.0, lam)
            g7 = band_imbalance(u, prob, mesh, THETA_MAX, lam)
            if g0 * g7 > 0:
                print(f"{dt:>6.1f}{th:>11.2f}{g0:>10.2f}{g7:>10.2f}"
                      f"{'none':>11}{'--':>9}")
                continue
            th_rec = -g0 / (g7 - g0) * THETA_MAX      # affine, so exact
            e = th_rec - float(th)
            e_list.append(abs(e))
            print(f"{dt:>6.1f}{th:>11.2f}{g0:>10.2f}{g7:>10.2f}"
                  f"{th_rec:>11.3f}{e:>+9.3f}")
        errs[float(dt)] = e_list
    print(f"\n{'delta (mm)':>12}{'recovered':>11}{'mean |err| (pp)':>18}")
    for dt, e in errs.items():
        m = f"{np.mean(e)*100:.2f}" if e else "--"
        print(f"{dt:>12.1f}{len(e):>11d}{m:>18}")


if __name__ == "__main__":
    main()
