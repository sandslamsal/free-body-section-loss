"""Validate the mixed continuum PINN against the continuum CSFM solver.

Loads the oracle field exported by scripts/p2_oracle.ts and the trained mixed
PINN, evaluates the PINN stress field at the oracle's element centroids, and
reports field-error metrics plus a side-by-side compression-field comparison.

Run (after pinn_mixed.py and scripts/p2_oracle.ts):
    python validate.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import DisplacementPINN
from pinn_mixed import StressPINN, stress
from problem import DeepBeam

torch.set_default_dtype(torch.float32)


def principal2(sx, sy, txy):
    """Minor principal stress (compression, negative)."""
    av = 0.5 * (sx + sy)
    rad = np.sqrt(((sx - sy) / 2) ** 2 + txy ** 2)
    return av - rad


def main() -> None:
    prob = DeepBeam()
    oracle = json.load(open("oracle_deepbeam.json"))
    el = oracle["elements"]
    cx = np.array([e["c"][0] for e in el])
    cy = np.array([e["c"][1] for e in el])
    o_sx = np.array([e["s"][0] for e in el])
    o_sy = np.array([e["s"][1] for e in el])
    o_txy = np.array([e["s"][2] for e in el])

    ckpt = torch.load("runs/deepbeam_mixed.pt", map_location="cpu")
    s_net = StressPINN(width=96, depth=6)
    s_net.load_state_dict(ckpt["s"])
    s_net.eval()

    x = torch.tensor(cx, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(cy, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        psx, psy, ptxy = stress(s_net, prob, x, y)
    psx = psx.numpy().ravel(); psy = psy.numpy().ravel(); ptxy = ptxy.numpy().ravel()

    o_s2 = principal2(o_sx, o_sy, o_txy)
    p_s2 = principal2(psx, psy, ptxy)

    def rel_rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9))

    print("=== mixed PINN vs continuum CSFM solver (deep beam) ===")
    print(f"  n elements                {len(el)}")
    print(f"  sigma_x  rel-RMS error    {rel_rms(psx, o_sx):.3f}")
    print(f"  sigma_y  rel-RMS error    {rel_rms(psy, o_sy):.3f}")
    print(f"  tau_xy   rel-RMS error    {rel_rms(ptxy, o_txy):.3f}")
    print(f"  sigma_2  rel-RMS error    {rel_rms(p_s2, o_s2):.3f}")
    print(f"  min sigma_2  oracle/PINN  {o_s2.min():.2f} / {p_s2.min():.2f} MPa")

    # side-by-side compression-field contour
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    vmin = min(o_s2.min(), p_s2.min())
    for ax, s2, ttl in ((axes[0], o_s2, "continuum CSFM solver"),
                        (axes[1], p_s2, "mixed continuum PINN")):
        tri = ax.tricontourf(cx, cy, s2, levels=24, cmap="RdBu",
                             vmin=vmin, vmax=0.0)
        fig.colorbar(tri, ax=ax, label=r"$\sigma_2$ (MPa)")
        for xc in prob.x_supp:
            ax.plot([xc], [0], "k^", ms=8)
        ax.plot([prob.x_load], [prob.H], "rv", ms=8)
        ax.set_aspect("equal")
        ax.set_title(ttl)
        ax.set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.suptitle("Compression field: PINN vs CSFM oracle, deep beam P = 800 kN")
    fig.tight_layout()
    fig.savefig("validation_field.png", dpi=200)
    plt.close(fig)
    print("  wrote validation_field.png")


if __name__ == "__main__":
    main()
