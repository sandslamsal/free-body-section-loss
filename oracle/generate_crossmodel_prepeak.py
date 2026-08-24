"""The same alternative-model family, taken before its own limit point.

The fields in fields_crossmodel.npz are read at a 3.5 mm deflection, the
station the study reports everywhere else. The alternative model turns out
to reach its limit point near 2.9 mm, so those states sit within a few per
cent of its peak, and on the finer mesh well past it. Two things then
differ at once, the form of the model and where on its own equilibrium path
the measurement is taken, and a single comparison cannot separate them.

This module removes the second difference. The same family is generated at
a 2.0 mm deflection, where both models are still on the rising branch, so
the comparison is like for like. Whatever survives here is the cost of the
model form alone.

Output: fields_crossmodel_prepeak.npz, same schema as the 3.5 mm family.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixed_crack_oracle as FX                                            # noqa: E402
from generate_crossmodel_fields import (RHO_NOM, THETA, sample_indices)    # noqa: E402

DELTA = 2.0
NX, NY = 40, 20                # nodes coincide with the measurement grid
OUT = HERE / "fields_crossmodel_prepeak.npz"


def main() -> None:
    xy_ref = np.load(HERE / "fields_theta.npz")["xy"]
    idx = sample_indices(NX, NY)
    tag = f"{NX}x{NY}"
    out = {"xy": xy_ref, "theta_true": np.array(THETA),
           "deltas": np.array([DELTA]), "rho_nom": np.array([RHO_NOM])}
    t0 = time.time()
    for th in THETA:
        u, g, st, hist = FX.solve(RHO_NOM * (1.0 - th), DELTA, nx=NX, ny=NY)
        assert np.allclose(g.xy[idx], xy_ref), "measurement grid mismatch"
        d, lam, rn, nit, cv, fr = hist[-1]
        chk = FX.section_check(u, g, st)
        key = f"{tag}_{th:.2f}_{DELTA:.1f}"
        out[f"u_{key}"] = u.reshape(-1, 2)[idx]
        out[f"lam_{key}"] = np.array([lam])
        out[f"resid_{key}"] = np.array([rn])
        out[f"conv_{key}"] = np.array([float(cv)])
        out[f"crack_{key}"] = np.array([fr])
        out[f"curve_{key}"] = np.array([[h[0], h[1]] for h in hist])
        out[f"unconv_{key}"] = np.array([sum(1 for h in hist if not h[4]),
                                         len(hist)])
        for kk, vv in chk.items():
            out[f"chk_{kk}_{key}"] = np.array([vv])
        print(f"  {tag}  theta={th:.2f}  lam={lam:.4f}  resid={rn:.2e} N"
              f" ({rn / (lam * 800.0e3):.2%})  cracked={fr:.1%}"
              f"  x_R={chk['x_reaction_mm']:.0f} mm"
              f"  M closure={100 * chk['M_cut_kNm'] / chk['M_statics_kNm']:.1f} %"
              f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    np.savez(OUT, **out)
    print(f"\n-> {OUT.name}  ({len(THETA)} fields, "
          f"{(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
