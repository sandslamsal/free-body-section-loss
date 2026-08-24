"""Does the network representation earn its place?

Both candidates reconstruct a DISPLACEMENT field, so both are compatible by
construction and both feed the identifying condition that recovers the
parameter exactly from a complete field. The comparison therefore isolates
what the network representation contributes, not what compatibility
contributes, which an earlier comparison against an incompatible strain
interpolation could not do.

  finite-element fit : nodal displacements by regularized least squares on
                       the reference mesh
  network            : a Fourier-feature displacement field trained on the
                       same readings

Gauges are weighted towards the tie band. That is where the identifying
information lives, and it is also where a fiber would physically be bonded,
so a layout ignoring it would be both uninformative and unrealistic.

Reported for each: the recovered section loss, and whether the identifying
function brackets a root at all, since a failure to bracket is a different
kind of failure from an inaccurate answer and should not be averaged with
it.
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
from fe_reconstruct import fit_field                                       # noqa: E402
from net_reconstruct import train, nodal_field, recover                    # noqa: E402

BAND = 150.0


def layout(mesh, n_band, n_rest, seed):
    """Gauge elements: a share on the band, the remainder over the face."""
    yc = np.array([mesh.xy[list(mesh.tris[e][0]), 1].mean()
                   for e in range(len(mesh.tris))])
    b = np.where(yc < BAND)[0]
    r = np.where(yc >= BAND)[0]
    g = np.random.default_rng(seed)
    return np.concatenate([g.choice(b, min(n_band, len(b)), replace=False),
                           g.choice(r, min(n_rest, len(r)), replace=False)])


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
    sc = np.abs(eps[:, 0]).mean()
    print(f"exact from the complete field: {recover(u_true, prob, mesh, lam):.4f}"
          f"   (true {th})\n")
    print(f"{'band':>6}{'rest':>6}{'noise':>7} | {'FE fit':>18} | "
          f"{'network':>18}")
    for nb, nr, noise in ((120, 120, 0.0), (120, 120, 0.02),
                          (60, 60, 0.0), (60, 60, 0.02)):
        fe, nn_ = [], []
        for seed in (0, 1, 2):
            el = layout(mesh, nb, nr, seed)
            g = np.random.default_rng(500 + seed)
            ge = eps[el] + noise * sc * g.standard_normal((len(el), 3))
            u_fe = fit_field(mesh, el, ge, 1e-6)
            fe.append(recover(u_fe, prob, mesh, lam))
            net = train(cen[el, 0], cen[el, 1], ge, prob, mesh,
                        iters=6000, seed=seed)
            nn_.append(recover(nodal_field(net, mesh), prob, mesh, lam))
        def fmt(v):
            v = np.array(v, float)
            ok = np.isfinite(v).sum()
            return (f"{np.nanmean(v):.3f} ({ok}/3)" if ok else "no bracket")
        print(f"{nb:>6}{nr:>6}{noise:>7.0%} | {fmt(fe):>18} | {fmt(nn_):>18}",
              flush=True)


if __name__ == "__main__":
    main()
