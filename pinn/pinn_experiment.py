"""The identification run as an inverse network, under both objectives.

Sections 2 and 4 argue from the objectives themselves, on fields held fixed
at the generating value. That is the identification setting when the field
is measured, but it is not what an inverse physics-informed network does: a
network can deform its field to accommodate a trial parameter, anchored by
the measurements, and the parameter is recovered through that interaction.
The argument is therefore incomplete until the two objectives have been put
to the network they are claims about.

This script does that. One architecture, one initialisation, one set of
sparse noisy gauges, one optimizer and one iteration budget. The only thing
that changes is which functional carries the parameter:

  inherited     L_data + w * L_equilibrium(theta)   the residual a forward
                                                    formulation minimizes
  proposed      L_data + w * L_reconcile(theta)     the moment of the
                                                    tractions on a cut

with theta a leaf of the graph in both, so both recover it by the same
reverse-mode sweep and neither is given an advantage in machinery.

Training is in two phases, and the reason is a confound found the hard way.
Released from the start alongside an unfitted field, theta does not measure
deterioration: it absorbs field error, because moving one scalar is a cheaper
way to satisfy a physics term than correcting a field still twenty-five times
its noise floor. In a first attempt the reconciliation was driven to 8e-4 and
returned a value wrong by 27 percentage points. Phase one therefore fits the
strain and displacement fields on measurements and compatibility alone,
neither of which involves theta, until the data loss reaches the floor the
measurement noise sets. Phase two releases theta. Since phase one is
independent of both the parameter and the objective, one fitted field per
deterioration state serves every objective and weight, so the two objectives
are compared from an identical starting field.

The weight is swept for the inherited objective, because the honest form of
the claim is about residual-DOMINATED objectives and the reader is entitled
to know where the pathology sets in rather than being shown one weight.

Run:  python pinn_experiment.py [n_iter]
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import pinn_inverse as P                                                   # noqa: E402
from csfm_constitutive import membrane                                     # noqa: E402
from identify import (DeteriorationParameter, equilibrium_loss_theta,
                      rho_x_of_theta)                # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains                                  # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
OUT = HERE.parent / "figures" / "pinn_experiment.json"
NX, NY = 40, 20
BAND, X_CUT, A_ARM = 150.0, 700.0, 370.0
N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
N_FIT = 12000
N_GAUGE, NOISE = 240, 0.02
WIDTH, DEPTH, N_COLL = 64, 4, 512

torch.set_default_dtype(torch.float32)


def fine_from_solver(prob, xy, u):
    """The dense reference the gauges are sampled from."""
    cx, cy, ex, ey, gxy = element_strains(xy, u, NX, NY)
    return {"cx": cx, "cy": cy,
            "strain": np.stack([ex, ey, gxy], axis=1)}


def cut_moment(e_net, prob, theta, n=1500, gen=None):
    """Moment of sigma_x over a strip standing for the cut, in kN m.

    Differentiable in theta and in the network, so both objectives are
    optimized by the same machinery.
    """
    g = gen if gen is not None else torch.Generator().manual_seed(3)
    hw = prob.L / NX
    x = (X_CUT - hw + 2 * hw * torch.rand(n, 1, generator=g)).requires_grad_(True)
    y = (torch.rand(n, 1, generator=g) * prob.H).requires_grad_(True)
    ex, ey, gxy = P.strain_net(e_net, prob, x, y)
    st = membrane(ex, ey, gxy, rho_x_of_theta(prob, x, y, theta),
                  prob.rho_y(x, y), prob.mat, soften=True)
    # The moment axis is the mid-height of the cut, not the mid-height of
    # the band. Any axis serves when the net axial force on the cut
    # vanishes, but here sigma_x is formed from a fixed measured strain at
    # a TRIAL theta, so the cut is not axially balanced unless the trial is
    # correct, and the choice of axis then matters. An axis inside the band
    # gives the theta-dependent part an arm that changes sign within the
    # band: its contribution nearly cancels and the observable goes blind,
    # varying by 0.2 kN m over the whole admissible range where the correct
    # axis gives about 180.
    y0 = prob.H / 2.0
    # mean over the strip times the area it stands for
    return (st["sigma_x"] * (y - y0)).mean() * prob.H * prob.t / 1e6


def reconcile_loss(e_net, prob, theta, lam):
    """Squared relative mismatch between the cut and what statics requires.

    The moment of the tractions on the cut balances the applied moment, so
    it carries the opposite sense and must be negated before the two are
    differenced. Differencing them directly leaves a constant of roughly
    twice the applied moment in the residual, which swamps the effect of
    theta entirely: the loss then sits at 3.60 for every trial value,
    varying by one part in ten thousand across the whole admissible range,
    and the parameter is driven to a bound by a slope that means nothing.
    """
    M_req = lam * prob.P / 2.0 * (X_CUT - A_ARM) / 1e6
    return ((-cut_moment(e_net, prob, theta) - M_req) / M_req) ** 2


def fit_field(prob, meas, seed=0, n_iter=N_FIT, lr=3e-3):
    """Phase one: the field alone, on data and compatibility."""
    torch.manual_seed(seed)
    e_net = P.StrainPINN(width=WIDTH, depth=DEPTH, n_out=3)
    u_net = P.StrainPINN(width=WIDTH, depth=DEPTH, n_out=2)
    opt = torch.optim.Adam(list(e_net.parameters()) + list(u_net.parameters()),
                           lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter,
                                                       eta_min=lr / 30)
    gen = torch.Generator().manual_seed(seed + 1)
    for _ in range(n_iter):
        opt.zero_grad()
        x, y = prob.interior(N_COLL, gen)
        x.requires_grad_(True); y.requires_grad_(True)
        ld = P.data_loss(e_net, prob, meas)
        lc = P.compat_loss(e_net, u_net, prob, x, y)
        (ld + lc).backward()
        opt.step(); sched.step()
    return e_net, u_net, float(ld) / meas["data_floor"]


def run(prob, meas, lam, objective, w, e0, u0, seed=0, n_iter=N_ITER):
    """Phase two: release theta on a copy of the fitted field."""
    e_net = copy.deepcopy(e0)
    u_net = copy.deepcopy(u0)
    par = DeteriorationParameter(theta_init=0.35)
    opt = torch.optim.Adam(
        [{"params": e_net.parameters(), "lr": 3e-4},
         {"params": u_net.parameters(), "lr": 3e-4},
         {"params": par.parameters(), "lr": 5e-3}])
    gen = torch.Generator().manual_seed(seed + 2)
    hist = []
    for it in range(n_iter):
        opt.zero_grad()
        x, y = prob.interior(N_COLL, gen)
        x.requires_grad_(True); y.requires_grad_(True)
        th = par()
        ld = P.data_loss(e_net, prob, meas)
        lc = P.compat_loss(e_net, u_net, prob, x, y)
        if objective == "inherited":
            lp = equilibrium_loss_theta(e_net, prob, x, y, th, True)
        else:
            lp = reconcile_loss(e_net, prob, th, lam)
        (ld + w * lp + lc).backward()
        opt.step()
        if it % 500 == 0 or it == n_iter - 1:
            hist.append({"it": it, "data": float(ld), "phys": float(lp),
                         "theta": par.value})
    return par.value, float(ld) / meas["data_floor"], hist


def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    xy = d["xy"]
    results = []
    t0 = time.time()
    for th_true in (0.10, 0.20, 0.30):
        k = f"u_{th_true:.2f}_3.5"
        if k not in d.files:
            continue
        lam = float(d[f"lam_{th_true:.2f}_3.5"][0])
        fine = fine_from_solver(prob, xy, d[k])
        meas = P.make_measurements(prob, fine, N_GAUGE, NOISE, seed=7)
        print(f"\ntrue theta = {th_true:.2f},  {meas['n']} gauges, "
              f"{100*NOISE:.0f} % noise, lambda = {lam:.3f}", flush=True)
        cache = HERE.parent / "figures" / f"field_{th_true:.2f}.pt"
        if cache.exists():
            blob = torch.load(cache, weights_only=False)
            e0, u0, ratio = blob["e"], blob["u"], blob["ratio"]
            print(f"  field loaded, {ratio:.1f}x the noise floor", flush=True)
        else:
            e0, u0, ratio = fit_field(prob, meas)
            torch.save({"e": e0, "u": u0, "ratio": ratio}, cache)
            print(f"  field fitted to {ratio:.1f}x the noise floor", flush=True)
        sweep = [0.1, 1.0, 10.0]
        for obj, weights in (("inherited", sweep), ("proposed", sweep)):
            for w in weights:
                th_hat, fr, hist = run(prob, meas, lam, obj, w, e0, u0)
                err = 100 * (th_hat - th_true)
                print(f"  {obj:<10} w = {w:6.1f}  ->  theta_hat = "
                      f"{th_hat:.3f}   error {err:+6.1f} pp   "
                      f"(field {fr:.1f}x floor)", flush=True)
                results.append({"true": th_true, "objective": obj,
                                "weight": w, "theta_hat": th_hat,
                                "error_pp": err, "fit_ratio": fr,
                                "field_fit": ratio, "history": hist})
    OUT.parent.mkdir(exist_ok=True)
    json.dump({"n_iter": N_ITER, "n_gauge": N_GAUGE, "noise": NOISE,
               "wall_s": time.time() - t0, "runs": results},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}   ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
