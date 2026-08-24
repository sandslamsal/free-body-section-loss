"""Fields at theta = 0.20, delta = 3.5 on a ladder of meshes.

The referee asks what the discretisation-driven part of the displaced
minimiser does under refinement, and two levels cannot show a trend. The
reference solve is 40x20 and a 60x30 solve already costs about an hour, so
the ladder is extended downward as well as upward: the cheap coarse levels
are solved here, and each is cached separately so a level that proves too
expensive can be abandoned without losing the ones already done.

Run:  python generate_mesh_levels.py NX NY
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arclength_oracle import build_mesh                                   # noqa: E402
from arclength_oracle_crisfield import newton_displacement_control        # noqa: E402
from oracle_rho_sweep import deepbeam_rho, RHO_NOM                        # noqa: E402

THETA, DELTA = 0.20, 3.5
OUTDIR = HERE / "mesh_levels"


def solve(nx: int, ny: int):
    out = OUTDIR / f"field_{nx}x{ny}_020.npz"
    if out.exists():
        print(f"{nx}x{ny} already cached", flush=True)
        return
    p = dataclasses.replace(deepbeam_rho(RHO_NOM * (1.0 - THETA)), nx=nx, ny=ny)
    mesh = build_mesh(p)
    print(f"solving {nx}x{ny} ...", flush=True)
    t0 = time.time()
    hist = newton_displacement_control(p, mesh, delta_max=DELTA,
                                       n_steps=max(6, int(DELTA * 8)),
                                       verbose=False)
    secs = time.time() - t0
    last = hist[-1]
    OUTDIR.mkdir(exist_ok=True)
    np.savez_compressed(out, xy=mesh.xy, u=np.asarray(last.u).reshape(-1, 2),
                        lam=float(last.lam), seconds=secs,
                        resid=float(last.resid))
    print(f"{nx}x{ny}: {secs:.0f} s, lam = {float(last.lam):.4f}, "
          f"resid = {float(last.resid):.2e}", flush=True)


if __name__ == "__main__":
    solve(int(sys.argv[1]), int(sys.argv[2]))
