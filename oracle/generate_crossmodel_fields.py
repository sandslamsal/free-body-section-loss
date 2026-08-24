"""Fields from a model the identification does not know about.

The reference family in fields_theta.npz was produced by the solver, the
constitutive and the element that the identification itself assumes, so the
accuracy it supports is a statement about self-consistency before it is one
about a structure. This module produces the same family of deterioration
states from the fixed smeared-crack quadrilateral model, which disagrees
with the Compatible Stress Field Method about crack rotation, about whether
concrete carries tension, about the shape of the compressive law, about
what drives compression softening, about shear across a crack and about the
element. The identification is then run on these fields unchanged.

The generator mesh is deliberately finer than the measurement grid and is
chosen so the grid is a subset of its nodes: at 80 by 40 the nodes fall
every 25 mm and the 50 mm measurement grid is every second one, so the
field handed to the identification is the alternative model's own nodal
displacement with no interpolation of ours in between. The same family is
also generated at 40 by 20, where the generator nodes and the measurement
grid coincide exactly, so that the part of any discrepancy owed to
resolution can be separated from the part owed to the model.

Output: fields_crossmodel.npz. Generated once and never regenerated; the
recovery scripts read it.
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
MESHES = [(80, 40), (40, 20)]
REF_NX, REF_NY = 40, 20            # the measurement grid, 50 mm spacing
OUT = HERE / "fields_crossmodel.npz"


def sample_indices(nx: int, ny: int) -> np.ndarray:
    """Generator node numbers lying on the 41 by 21 measurement grid.

    Both meshes number nodes row by row from the origin, and both are
    uniform, so the measurement node (i, j) is generator node
    (j*ry)*(nx+1) + i*rx with integer strides. The assertion below refuses
    to proceed unless the two grids really do coincide, since a silent
    half-cell offset here would look exactly like model error.
    """
    rx, ry = nx // REF_NX, ny // REF_NY
    assert rx * REF_NX == nx and ry * REF_NY == ny, (nx, ny)
    return np.array([(j * ry) * (nx + 1) + i * rx
                     for j in range(REF_NY + 1) for i in range(REF_NX + 1)])


def main() -> None:
    ref = np.load(HERE / "fields_theta.npz")
    xy_ref = ref["xy"]
    out = {"xy": xy_ref, "theta_true": np.array(THETA),
           "deltas": np.array([DELTA]), "rho_nom": np.array([RHO_NOM])}
    t0 = time.time()
    for nx, ny in MESHES:
        idx = sample_indices(nx, ny)
        tag = f"{nx}x{ny}"
        for th in THETA:
            u, g, st, hist = FX.solve(RHO_NOM * (1.0 - th), DELTA,
                                      nx=nx, ny=ny, verbose=False)
            assert np.allclose(g.xy[idx], xy_ref), "measurement grid mismatch"
            d, lam, rn, nit, cv, fr = hist[-1]
            chk = FX.section_check(u, g, st)
            key = f"{tag}_{th:.2f}_{DELTA:.1f}"
            for kk, vv in chk.items():
                out[f"chk_{kk}_{key}"] = np.array([vv])
            out[f"curve_{key}"] = np.array([[h[0], h[1]] for h in hist])
            out[f"unconv_{key}"] = np.array([sum(1 for h in hist if not h[4]),
                                             len(hist)])
            out[f"u_{key}"] = u.reshape(-1, 2)[idx]
            out[f"lam_{key}"] = np.array([lam])
            out[f"resid_{key}"] = np.array([rn])
            out[f"conv_{key}"] = np.array([float(cv)])
            out[f"crack_{key}"] = np.array([fr])
            out[f"ufull_{key}"] = u.reshape(-1, 2)
            out[f"xyfull_{tag}"] = g.xy
            print(f"  {tag}  theta={th:.2f}  lam={lam:.4f}  resid={rn:.2e} N"
                  f" ({rn / (lam * 800.0e3):.2%})  cracked={fr:.1%}"
                  f"  x_R={chk['x_reaction_mm']:.0f} mm"
                  f"  M closure={100 * chk['M_cut_kNm'] / chk['M_statics_kNm']:.1f} %"
                  f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    np.savez(OUT, **out)
    print(f"\n-> {OUT.name}  ({len(MESHES) * len(THETA)} fields, "
          f"{(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
