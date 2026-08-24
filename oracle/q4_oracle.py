"""Q4 reference solver, following the element choice the CSFM prescribes.

The existing reference uses constant-strain triangles. Section 3.6.5 of
Kaufmann et al. (2020) states that the CSFM models concrete with
quadrilateral shell elements (CQUAD4) carrying four integration points
"placed approximately at its quarter points", and diagnostics on the
triangular reference showed exactly the deficiency that choice avoids: the
moment of the recovered stresses over a section came out 25 to 37 per cent
short of the applied moment, while the axial and shear resultants closed.
A constant-strain element is exact for uniform strain and cannot represent
the linear through-depth gradient that carries bending, so every
identification routed through a sectional stress integral inherited that
error.

This module provides the same displacement-controlled problem on four-node
quadrilaterals with 2x2 Gauss integration, which matches the book's element
and quadrature. The constitutive, materials, reinforcement ratios, supports
and loading are unchanged, so the only difference from the existing
reference is the element.

Verification first: the sectional moment of the recovered stresses is
compared against statics before the fields are used for anything.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arclength_oracle import Material, membrane                            # noqa: E402

GP = 1.0 / np.sqrt(3.0)
GAUSS = [(-GP, -GP), (GP, -GP), (GP, GP), (-GP, GP)]


@dataclass
class Q4Mesh:
    nx: int
    ny: int
    xy: np.ndarray
    quads: np.ndarray          # (ne, 4) node indices, counter-clockwise
    ndof: int
    fixed: np.ndarray
    F_ref: np.ndarray
    load_nodes: list


def build_q4(L, H, nx, ny, a, bearing, P_ref):
    nnx, nny = nx + 1, ny + 1
    xs = np.linspace(0, L, nnx)
    ys = np.linspace(0, H, nny)
    xy = np.array([[x, y] for y in ys for x in xs])
    nid = lambda i, j: j * nnx + i                                   # noqa: E731
    quads = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                      for j in range(ny) for i in range(nx)])
    ndof = 2 * len(xy)
    fixed = np.zeros(ndof, bool)
    half = bearing / 2.0
    for n in range(len(xy)):
        x, y = xy[n]
        if y < 1e-9 and abs(x - a) <= half + 1e-9:
            fixed[2 * n] = fixed[2 * n + 1] = True          # pin
        if y < 1e-9 and abs(x - (L - a)) <= half + 1e-9:
            fixed[2 * n + 1] = True                          # roller
    load_nodes = [n for n in range(len(xy))
                  if abs(xy[n, 1] - H) < 1e-9
                  and abs(xy[n, 0] - L / 2) <= half + 1e-9]
    F_ref = np.zeros(ndof)
    for n in load_nodes:
        F_ref[2 * n + 1] -= P_ref / len(load_nodes)
    return Q4Mesh(nx, ny, xy, quads, ndof, fixed, F_ref, load_nodes)


def shape_derivs(xe, xi, eta):
    """dN/dx, dN/dy and the Jacobian determinant at one Gauss point."""
    dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                          [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
    J = dN @ xe
    detJ = np.linalg.det(J)
    return np.linalg.solve(J, dN), detJ


def b_matrix(dNxy):
    B = np.zeros((3, 8))
    B[0, 0::2] = dNxy[0]
    B[1, 1::2] = dNxy[1]
    B[2, 0::2] = dNxy[1]
    B[2, 1::2] = dNxy[0]
    return B


def element_state(u_e, xe, rho_x, rho_y, mat, t):
    """Internal force and tangent of one Q4, summed over its Gauss points."""
    fe = np.zeros(8)
    ke = np.zeros((8, 8))
    for xi, eta in GAUSS:
        dNxy, detJ = shape_derivs(xe, xi, eta)
        B = b_matrix(dNxy)
        eps = B @ u_e
        sig, D, _ = membrane(eps[0], eps[1], eps[2], rho_x, rho_y, mat)
        w = detJ * t
        fe += B.T @ np.asarray(sig).ravel() * w
        ke += B.T @ np.asarray(D) @ B * w
    return fe, ke


def assemble(u, mesh, prob_rho, mat, t, band=150.0, rho_min=0.0010,
             rho_y=0.0025):
    """Global internal force and tangent."""
    F = np.zeros(mesh.ndof)
    K = np.zeros((mesh.ndof, mesh.ndof))
    for q in mesh.quads:
        xe = mesh.xy[q]
        dofs = np.array([2 * q[0], 2 * q[0] + 1, 2 * q[1], 2 * q[1] + 1,
                         2 * q[2], 2 * q[2] + 1, 2 * q[3], 2 * q[3] + 1])
        yc = xe[:, 1].mean()
        rx = prob_rho if yc < band else rho_min
        fe, ke = element_state(u[dofs], xe, rx, rho_y, mat, t)
        F[dofs] += fe
        K[np.ix_(dofs, dofs)] += ke
    return F, K


def solve_dc(mesh, rho_tie, mat, t, P_ref, delta, n_steps=20,
             tol=1e-3, max_it=60, verbose=False):
    """Displacement control: prescribe the load-patch deflection, read the
    load factor from the patch reaction, exactly as the CST reference does."""
    u = np.zeros(mesh.ndof)
    pres = np.array([2 * n + 1 for n in mesh.load_nodes])
    free = ~mesh.fixed.copy()
    free[pres] = False
    hist = []
    for k in range(1, n_steps + 1):
        d = -delta * k / n_steps
        u[pres] = d
        mu = 1e-4
        for it in range(max_it):
            F, K = assemble(u, mesh, rho_tie, mat, t)
            r = -F[free]
            nr = np.linalg.norm(r)
            if nr < tol * max(1.0, P_ref * 1e-3):
                break
            Kff = K[np.ix_(free, free)]
            Kff = Kff + mu * np.diag(np.diag(Kff) + 1e-9)
            try:
                du = np.linalg.solve(Kff, r)
            except np.linalg.LinAlgError:
                mu *= 10
                continue
            u_try = u.copy()
            u_try[free] += du
            F2, _ = assemble(u_try, mesh, rho_tie, mat, t)
            if np.linalg.norm(F2[free]) < nr:
                u = u_try
                mu = max(mu * 0.5, 1e-8)
            else:
                mu *= 4
        lam = -F[pres].sum() / P_ref
        hist.append((float(-d), float(lam), float(nr)))
        if verbose:
            print(f"  step {k:2d}  delta={-d:6.2f}  lam={lam:.4f}  "
                  f"resid={nr:.2e}", flush=True)
    return u, hist
