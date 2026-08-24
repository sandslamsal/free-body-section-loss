"""Does the omitted tension stiffening explain the residual bias?

Section 7.1 records a deliberate deviation from the reference formulation:
the reinforcement carries the bare-bar stress, and the tension chord model
of Marti et al., in which concrete between cracks carries tension through
bond and raises the force a given average strain implies, is not used. That
omission was argued to bias the recovered section loss, and the bias
observed is 2 to 7 percentage points low. The two have never been connected
by measurement, and if the omission accounts for the bias then Section 7.4
attributes it to the wrong cause.

A full tension chord treatment would make the identifying quantity only
approximately affine and is beyond what this check needs. What is needed is
the size of the effect, so tension stiffening is represented by an average
tensile stress carried by the concrete of the band, sigma_ct = beta f_ct,
and beta is swept over the range the stabilised-cracking literature spans.
The result is a bracket on how much of the bias the omission can carry, not
a tension chord implementation.

Run:  python tension_chord.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figdata as FD                                                       # noqa: E402
from csfm_constitutive import membrane                                     # noqa: E402
from identify import rho_x_of_theta                                        # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains, bracket_root                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
DELTA = "3.5"


def couple_ts(prob, cx, cy, ex, ey, gxy, area, theta, beta, f_ct):
    """Band couple with an average tensile stress beta*f_ct in the band."""
    sel = np.abs(cx - FD.X_CUT) < FD.BAND_W
    X = torch.tensor(cx[sel]).unsqueeze(-1); Y = torch.tensor(cy[sel]).unsqueeze(-1)
    st = membrane(torch.tensor(ex[sel]).unsqueeze(-1),
                  torch.tensor(ey[sel]).unsqueeze(-1),
                  torch.tensor(gxy[sel]).unsqueeze(-1),
                  rho_x_of_theta(prob, X, Y, torch.tensor(float(theta))),
                  prob.rho_y(X, Y), prob.mat, soften=True)
    sx = st["sigma_x"].squeeze().numpy()
    ys = cy[sel]
    inb = ys < FD.BAND
    # tension stiffening acts only where the band is in tension
    sx = sx + beta * f_ct * (inb & (sx > 0.0))
    dA = area / (2.0 * FD.BAND_W) * prob.t
    T = float((sx[inb] * dA).sum()) / 1e3
    wT = np.clip(sx[inb], 0.0, None)
    wC = np.clip(-sx[~inb], 0.0, None)
    yT = float((wT * ys[inb]).sum() / max(wT.sum(), 1e-9))
    yC = float((wC * ys[~inb]).sum() / max(wC.sum(), 1e-9))
    return T * (yC - yT) / 1e3


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    f_ct = 0.30 * prob.mat.fc ** (2.0 / 3.0)
    g = np.linspace(0.0, 0.70, 141)
    betas = [0.0, 0.05, 0.10, 0.20]
    print(f"f_ct = {f_ct:.2f} MPa;  beta f_ct is the average tensile stress\n")
    print(f"{'true':>6}" + "".join(f"{f'beta={b:.2f}':>12}" for b in betas))
    err = {b: [] for b in betas}
    for th in [float(t) for t in d["theta_true"]]:
        k = f"u_{th:.2f}_{DELTA}"
        if k not in d.files:
            continue
        lam = float(d[f"lam_{th:.2f}_{DELTA}"][0])
        cx, cy, ex, ey, gxy = element_strains(d["xy"], d[k], FD.NX, FD.NY)
        M_req = lam * prob.P / 2.0 * (FD.X_CUT - 370.0) / 1e6
        row = []
        for b in betas:
            f = np.array([couple_ts(prob, cx, cy, ex, ey, gxy, area, q, b,
                                    f_ct) - M_req for q in g])
            r = bracket_root(f, g)
            row.append(r)
            if np.isfinite(r) and th > 0:
                err[b].append(abs(r - th) * 100)
        print(f"{th:>6.2f}" + "".join(
            f"{('none' if np.isnan(v) else f'{v:.3f}'):>12}" for v in row),
            flush=True)
    print("\nmean |error| in pp:")
    for b in betas:
        print(f"  beta = {b:.2f}: {np.mean(err[b]):.1f}")
    print("\nIf a plausible beta closes the bias, the omission of tension")
    print("stiffening explains it and the discretization does not.")


if __name__ == "__main__":
    main()
