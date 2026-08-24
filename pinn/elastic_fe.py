"""Trusted linear-elastic plane-stress FE solver -- a reference, not a PINN.

A standard constant-strain-triangle finite-element solver for the deep-beam
D-region, with exactly the elastic constitutive law of
`csfm_constitutive.elastic_plane_stress` (isotropic concrete E_c, nu = 0.2,
plus smeared reinforcement rho*E_s on the normal directions). It is textbook
code, independent of every PINN file, so it can be used to answer one
question cleanly: can the PINN reproduce linear elasticity at all?

Run:  python elastic_fe.py   -> writes elastic_fe_deepbeam.json
"""
from __future__ import annotations

import json

import numpy as np

from problem import DeepBeam


def solve(prob: DeepBeam, nx: int = 40, ny: int = 20) -> dict:
    L, H, t = prob.L, prob.H, prob.t
    dx, dy = L / nx, H / ny
    nnx, nny = nx + 1, ny + 1
    n_node = nnx * nny
    ndof = 2 * n_node

    def nid(i, j):
        return j * nnx + i

    xy = np.array([[i * dx, j * dy] for j in range(nny) for i in range(nnx)],
                  dtype=float)

    nu = 0.2
    Ec = prob.mat.Ec0
    Es = prob.mat.Es
    coef = Ec / (1.0 - nu * nu)
    Dc = coef * np.array([[1.0, nu, 0.0],
                          [nu, 1.0, 0.0],
                          [0.0, 0.0, 0.5 * (1.0 - nu)]])

    # ---- triangles: 2 CST per cell --------------------------------------
    tris = []
    for j in range(ny):
        for i in range(nx):
            a, b = nid(i, j), nid(i + 1, j)
            c, d = nid(i + 1, j + 1), nid(i, j + 1)
            yc = (j + 0.5) * dy
            rx = prob.rho_tie if yc < prob.band else prob.rho_min
            ry = prob.rho_min + prob.rho_stirrup
            tris.append(([a, b, c], rx, ry))
            tris.append(([a, c, d], rx, ry))

    K = np.zeros((ndof, ndof))
    for nodes, rx, ry in tris:
        p = xy[nodes]
        b1, b2, b3 = p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]
        c1, c2, c3 = p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]
        det = p[0, 0] * b1 + p[1, 0] * b2 + p[2, 0] * b3
        area = abs(det) / 2.0
        inv = 1.0 / det
        B = np.array([
            [b1 * inv, 0, b2 * inv, 0, b3 * inv, 0],
            [0, c1 * inv, 0, c2 * inv, 0, c3 * inv],
            [c1 * inv, b1 * inv, c2 * inv, b2 * inv, c3 * inv, b3 * inv],
        ])
        D = Dc.copy()
        D[0, 0] += rx * Es           # smeared steel, x bars
        D[1, 1] += ry * Es           # smeared steel, y bars
        ke = (B.T @ D @ B) * area * t
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        K[np.ix_(dofs, dofs)] += ke

    # ---- load: total P over the bearing patch on the top edge -----------
    F = np.zeros(ndof)
    half = prob.bearing / 2.0
    top = [nid(i, ny) for i in range(nnx)
           if abs(i * dx - prob.x_load) <= half + 1e-6]
    for n in top:
        F[2 * n + 1] -= prob.P / len(top)

    # ---- supports: fix u_y at both, u_x at the left support -------------
    fixed = []
    for k, xc in enumerate(prob.x_supp):
        for n in range(n_node):
            if abs(xy[n, 0] - xc) <= half + 10 and xy[n, 1] < 1e-6:
                fixed.append(2 * n + 1)
                if k == 0:
                    fixed.append(2 * n)
    for g in set(fixed):
        K[g, :] = 0.0
        K[:, g] = 0.0
        K[g, g] = 1.0
        F[g] = 0.0

    u = np.linalg.solve(K, F)

    # ---- recover element stresses at centroids --------------------------
    cx, cy, sx, sy, txy = [], [], [], [], []
    for nodes, rx, ry in tris:
        p = xy[nodes]
        b1, b2, b3 = p[1, 1] - p[2, 1], p[2, 1] - p[0, 1], p[0, 1] - p[1, 1]
        c1, c2, c3 = p[2, 0] - p[1, 0], p[0, 0] - p[2, 0], p[1, 0] - p[0, 0]
        det = p[0, 0] * b1 + p[1, 0] * b2 + p[2, 0] * b3
        inv = 1.0 / det
        B = np.array([
            [b1 * inv, 0, b2 * inv, 0, b3 * inv, 0],
            [0, c1 * inv, 0, c2 * inv, 0, c3 * inv],
            [c1 * inv, b1 * inv, c2 * inv, b2 * inv, c3 * inv, b3 * inv],
        ])
        dofs = np.array([2 * nodes[0], 2 * nodes[0] + 1,
                         2 * nodes[1], 2 * nodes[1] + 1,
                         2 * nodes[2], 2 * nodes[2] + 1])
        eps = B @ u[dofs]
        D = Dc.copy()
        D[0, 0] += rx * Es
        D[1, 1] += ry * Es
        s = D @ eps
        cx.append(float(p[:, 0].mean())); cy.append(float(p[:, 1].mean()))
        sx.append(float(s[0])); sy.append(float(s[1])); txy.append(float(s[2]))
    return {"cx": cx, "cy": cy, "sx": sx, "sy": sy, "txy": txy,
            "max_uy": float(-u[1::2].min())}


def main() -> None:
    prob = DeepBeam()
    r = solve(prob)
    s2 = [0.5 * (a + b) - np.hypot(0.5 * (a - b), c)
          for a, b, c in zip(r["sx"], r["sy"], r["txy"])]
    json.dump(r, open("elastic_fe_deepbeam.json", "w"))
    print(f"elastic FE: {len(r['cx'])} elements, "
          f"min sigma_2 = {min(s2):.2f} MPa, max deflection = "
          f"{r['max_uy']:.3f} mm  -> elastic_fe_deepbeam.json")


if __name__ == "__main__":
    main()
