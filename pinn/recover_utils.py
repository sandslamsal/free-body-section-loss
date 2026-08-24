"""Shared kinematics: strains from a CST displacement field."""
from __future__ import annotations
import numpy as np


def element_strains(xy, u, nx=40, ny=20):
    """Constant strain per CST element, with element centroids.

    The reference mesh is a rectangular grid split into two triangles per
    cell, so each element has a single strain state obtained from the
    shape-function derivatives. Returned alongside the centroids so that
    any integration over a region can be done by selection.
    """
    nnx = nx + 1
    cx, cy, ex, ey, gxy = [], [], [], [], []
    for j in range(ny):
        for i in range(nx):
            n00 = j * nnx + i
            for tri in ((n00, n00 + 1, n00 + nnx + 1),
                        (n00, n00 + nnx + 1, n00 + nnx)):
                p = xy[list(tri)]
                dsp = u[list(tri)]
                x1, y1 = p[0]; x2, y2 = p[1]; x3, y3 = p[2]
                A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if abs(A2) < 1e-12:
                    continue
                b = np.array([y2 - y3, y3 - y1, y1 - y2]) / A2
                c = np.array([x3 - x2, x1 - x3, x2 - x1]) / A2
                ux, uy = dsp[:, 0], dsp[:, 1]
                ex.append(float(b @ ux)); ey.append(float(c @ uy))
                gxy.append(float(c @ ux + b @ uy))
                cx.append(float(p[:, 0].mean())); cy.append(float(p[:, 1].mean()))
    return (np.array(cx), np.array(cy), np.array(ex),
            np.array(ey), np.array(gxy))


def bracket_root(f, x):
    """First sign change of f on the grid x, by exact linear interpolation.

    The identifying function falls through its target, so the bracketing
    pair runs from positive to negative. numpy's interp requires the
    sample points to increase and silently returns the right-hand node
    when they do not, which snaps every root onto the trial grid and
    biases it toward the coarser side. The two-point formula below is
    direction independent and is what this study uses everywhere a root
    is taken from a bracket.
    """
    f = np.asarray(f, float)
    x = np.asarray(x, float)
    s = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    if not len(s):
        return np.nan
    i = int(s[0])
    df = f[i + 1] - f[i]
    if df == 0.0:
        return float(x[i])
    return float(x[i] - f[i] * (x[i + 1] - x[i]) / df)
