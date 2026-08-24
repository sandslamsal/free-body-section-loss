"""Is there a physics weight that fits the data AND equilibrates?

Two extremes are known and both give unusable fields. With no prior the
gauge misfit reaches 2.4e-4 while the equilibrium residual rises to 8017,
above its value at initialisation, and the identifying quantity comes out
twenty times too large with the wrong sign. With a prior at 3e-3 the
residual falls to 121 but the gauge misfit stalls at 3.4, three orders
above what the same network reaches unregularised.

Before building a staged scheme to reconcile them, it is worth spending ten
minutes establishing whether any fixed weight already does. If the sweep
shows a weight where both terms are acceptable, the remedy is a weight. If
every weight trades one for the other, the tension is structural and a
schedule is the right answer. If nothing works at any weight, neither is,
and the formulation needs rethinking.

Runs are deliberately short and the collocation count reduced: this is a
diagnostic, and the question is where the curve turns, not the best
attainable value at any point on it.
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
import net_reconstruct as NR                                               # noqa: E402
from recover_nodal import band_imbalance                                   # noqa: E402


def main() -> None:
    d = np.load(HERE.parent / "oracle" / "fields_theta.npz")
    th, dt = 0.30, 3.5
    prob = deepbeam_rho(RHO_NOM * (1.0 - th))
    mesh = build_mesh(prob)
    u_true = d[f"u_{th:.2f}_{dt:.1f}"].ravel()
    lam = float(d[f"lam_{th:.2f}_{dt:.1f}"][0])

    def dofs(e):
        n = mesh.tris[e][0]
        return [2*n[0], 2*n[0]+1, 2*n[1], 2*n[1]+1, 2*n[2], 2*n[2]+1]
    eps = np.array([mesh.B[e] @ u_true[dofs(e)] for e in range(len(mesh.tris))])
    cen = np.array([mesh.xy[list(mesh.tris[e][0])].mean(axis=0)
                    for e in range(len(mesh.tris))])
    yc = cen[:, 1]
    g = np.random.default_rng(0)
    el = np.concatenate([g.choice(np.where(yc < 150)[0], 150, replace=False),
                         g.choice(np.where(yc >= 150)[0], 150, replace=False)])

    g0t = band_imbalance(u_true, prob, mesh, 0.0, lam)
    g7t = band_imbalance(u_true, prob, mesh, 0.70, lam)
    print(f"target: g(0) = {g0t:.1f} kN,  g(0.7) = {g7t:.1f} kN,  "
          f"root at theta = {th}\n")
    print(f"{'w_phys':>9}{'g(0)':>10}{'g(0.7)':>10}{'theta':>9}{'err pp':>9}")
    for w in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
        net = NR.train(cen[el, 0], cen[el, 1], eps[el], prob, mesh,
                       iters=2500, w_phys=w, seed=0)
        u = NR.nodal_field(net, mesh)
        a = band_imbalance(u, prob, mesh, 0.0, lam)
        b = band_imbalance(u, prob, mesh, 0.70, lam)
        if a * b < 0:
            t = -a / (b - a) * 0.70
            print(f"{w:>9.0e}{a:>10.1f}{b:>10.1f}{t:>9.3f}{(t-th)*100:>9.2f}")
        else:
            print(f"{w:>9.0e}{a:>10.1f}{b:>10.1f}{'no root':>9}{'--':>9}")


if __name__ == "__main__":
    main()
