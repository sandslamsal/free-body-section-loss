"""The cross-model family from the REGULARISED alternative model.

The fields in fields_crossmodel.npz were generated before the compressive
descending branch of fixed_crack_oracle.py was crack-band regularised. On
that version the theta = 0 capacity at delta = 3.5 mm moved 25 per cent
between 40x20 and 80x40, and on 80x40 the capacity ROSE with damage, so the
27.8 pp identification offset was measured against a generator that had not
earned the authority the comparison gives it. This module regenerates the
same deterioration family from the regularised model on three meshes, so
that (a) monotonicity of capacity in theta can be verified on every mesh
and (b) mesh convergence of the theta = 0 capacity is measured rather than
asserted, before the identification is re-run.

The 50 mm measurement grid coincides with generator nodes on 40x20 (every
node) and 80x40 (every second node); on 60x30 it does not, so that mesh
contributes the convergence point only and no u_ on the measurement grid is
stored for it. Nothing else about the generation is changed from
generate_crossmodel_fields.py.

Output: fields_crossmodel_reg.npz, saved incrementally after every field.

Run:  python generate_crossmodel_reg.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixed_crack_oracle as FX                                            # noqa: E402

RHO_NOM = 0.012
THETA = [0.0, 0.10, 0.20, 0.30, 0.40]
DELTA = 3.5
MESHES = [(40, 20), (60, 30), (80, 40)]
REF_NX, REF_NY = 40, 20            # the measurement grid, 50 mm spacing
OUT = HERE / "fields_crossmodel_reg.npz"


def sample_indices(nx: int, ny: int) -> np.ndarray | None:
    """Generator node numbers on the 41 by 21 measurement grid, or None
    when the grids do not coincide (60x30). A silent half-cell offset would
    look exactly like model error, so nothing is ever interpolated."""
    if nx % REF_NX or ny % REF_NY:
        return None
    rx, ry = nx // REF_NX, ny // REF_NY
    return np.array([(j * ry) * (nx + 1) + i * rx
                     for j in range(REF_NY + 1) for i in range(REF_NX + 1)])


def main() -> None:
    ref = np.load(HERE / "fields_theta.npz")
    xy_ref = ref["xy"]
    out = {"xy": xy_ref, "theta_true": np.array(THETA),
           "deltas": np.array([DELTA]), "rho_nom": np.array([RHO_NOM]),
           "gc_nmm": np.array([FX.GC])}
    # theta = 0 on every mesh first, so mesh convergence is measurable
    # before the damaged states are in.
    jobs = [(nx, ny, 0.0) for nx, ny in MESHES]
    jobs += [(nx, ny, th) for nx, ny in MESHES for th in THETA[1:]]
    t0 = time.time()
    for nx, ny, th in jobs:
        tag = f"{nx}x{ny}"
        idx = sample_indices(nx, ny)
        u, g, st, hist = FX.solve(RHO_NOM * (1.0 - th), DELTA,
                                  nx=nx, ny=ny, verbose=False)
        d, lam, rn, nit, cv, fr = hist[-1]
        chk = FX.section_check(u, g, st)
        key = f"{tag}_{th:.2f}_{DELTA:.1f}"
        for kk, vv in chk.items():
            out[f"chk_{kk}_{key}"] = np.array([vv])
        out[f"curve_{key}"] = np.array([[h[0], h[1]] for h in hist])
        out[f"unconv_{key}"] = np.array([sum(1 for h in hist if not h[4]),
                                         len(hist)])
        if idx is not None:
            assert np.allclose(g.xy[idx], xy_ref), "measurement grid mismatch"
            out[f"u_{key}"] = u.reshape(-1, 2)[idx]
        out[f"lam_{key}"] = np.array([lam])
        out[f"resid_{key}"] = np.array([rn])
        out[f"conv_{key}"] = np.array([float(cv)])
        out[f"crack_{key}"] = np.array([fr])
        out[f"ufull_{key}"] = u.reshape(-1, 2)
        out[f"xyfull_{tag}"] = g.xy
        np.savez(OUT, **out)
        print(f"  {tag}  theta={th:.2f}  lam={lam:.4f}  resid={rn:.2e} N"
              f" ({rn / (lam * 800.0e3):.2%})  cracked={fr:.1%}"
              f"  x_R={chk['x_reaction_mm']:.0f} mm"
              f"  M closure={100 * chk['M_cut_kNm'] / chk['M_statics_kNm']:.1f} %"
              f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    print(f"\n-> {OUT.name}  ({len(jobs)} fields, "
          f"{(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
