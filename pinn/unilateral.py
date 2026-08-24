"""Where does the reaction act if the bearing cannot pull?

The reference solver prescribes the vertical displacement at every node
under the bearing plate, which is a bilateral constraint: those nodes carry
tension as readily as compression. A roller or an elastomeric bearing cannot
do that. It separates instead, and the contact shrinks to the part of the
plate that stays in compression.

The distinction matters here because the reaction centroid is the largest
single sensitivity in the identification, at 0.25 percentage points of
section loss per millimeter. On the bilateral solution the three outer nodes
carry 123 kN of uplift, and that negative first moment carries the centroid
to 370 mm, beyond the inner edge of the plate at 350. If the uplift is an
artefact of the constraint rather than a property of the structure, so is
part of that offset.

This script re-solves with a compression-only support by active set: solve,
release any support node whose reaction is tensile, repeat until the set
stops changing.

Run:  python unilateral.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

from arclength_oracle import build_mesh                                     # noqa: E402
from arclength_oracle_crisfield import newton_displacement_control          # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                          # noqa: E402

DELTA = 3.5


def support_nodes(mesh):
    return [n for n in range(mesh.n_node)
            if mesh.fixed[2 * n + 1] and mesh.xy[n, 0] < 600.0]


def centroid(xs, rs):
    return float((xs * rs).sum() / rs.sum())


def main() -> None:
    for th in (0.0, 0.20, 0.40):
        prob = deepbeam_rho(RHO_NOM * (1.0 - th))
        mesh = build_mesh(prob)
        released: set[int] = set()
        for sweep in range(6):
            m = build_mesh(prob)
            for n in released:
                m.fixed[2 * n + 1] = False
            hist = newton_displacement_control(prob, m, delta_max=DELTA,
                                               n_steps=max(6, int(DELTA * 8)),
                                               verbose=False)
            last = hist[-1]
            u = np.asarray(last.u).ravel()
            from recover_nodal import internal_forces
            R = internal_forces(u, prob, m, th) - last.lam * m.F_ref
            act = [n for n in support_nodes(mesh) if n not in released]
            xs = np.array([m.xy[n, 0] for n in act])
            rs = np.array([R[2 * n + 1] / 1e3 for n in act])
            # a support that is pulling down is not in contact
            tensile = [n for n, r in zip(act, rs) if r < -1.0]
            if not tensile:
                break
            released |= set(tensile)
        keep = np.array([m.xy[n, 0] for n in act])
        vals = np.array([R[2 * n + 1] / 1e3 for n in act])
        print(f"theta = {th:.2f}, {sweep + 1} active-set sweeps, "
              f"{len(released)} node(s) released")
        for x, r in zip(keep, vals):
            print(f"    x = {x:5.0f} mm   {r:+8.1f} kN")
        print(f"    total {vals.sum():.1f} kN, centroid "
              f"{centroid(keep, vals):.1f} mm   "
              f"(bilateral gave 370.3 at theta = 0.20)\n", flush=True)


if __name__ == "__main__":
    main()
