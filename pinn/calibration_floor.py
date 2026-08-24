"""What a commissioning reading buys, beyond the zero point.

Section 7.10 identifies the concrete tension the map neglects from a
reading taken where theta is known, and carries it into the deteriorated
states. By construction that forces theta_hat = 0 on the calibration
reading itself. But the false-positive floor of Table 2 and the detection
threshold of Section 7.11 are both computed uncalibrated, from a sound-tie
distribution whose mean offset the calibration would remove. So the
obvious question has not been answered: what are the floor and the
threshold *after* calibration?

They cannot simply go to zero, because the calibration reading carries
noise of its own. Self-referencing converts a systematic offset into the
difference of two noisy readings: the mean goes, and the spread grows by
roughly root two. Whether that is a good trade is what this module
measures, and it is measured the way an inspection would experience it,
with an independent noise draw on the commissioning reading and on the
later one.

Two routes are tested, because they are not equally available:

  in time    the commissioning reading is the same member before
             deterioration. A laboratory specimen has one by construction;
             a structure first instrumented after decades of exposure does
             not.
  in space   the calibration reading is a zone of the same member known to
             be sound, read at the same time. Section 7.13 is a reason to
             doubt this one: deterioration elsewhere moves the load path
             and degrades the recovery of a zone that is itself intact.
             Tested here on a purpose-generated (0.30, 0.00) field, in
             which the right-hand zone really is sound.

Run:  /usr/local/bin/python3.12 calibration_floor.py
      -> figures/calibration_floor.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import figdata as FD                                                       # noqa: E402
from problem import DeepBeam                                               # noqa: E402
from recover_utils import element_strains, bracket_root                    # noqa: E402

FIELDS = HERE.parent / "oracle" / "fields_theta.npz"
ASYM_SOUND = HERE.parent / "oracle" / "field_asym_sound.npz"
OUT = HERE.parent / "figures" / "calibration_floor.json"

DELTA = "3.5"
THETA = [0.0, 0.10, 0.20, 0.30, 0.40]
NOISE, N_REAL, PFA, ARM = 0.05, 200, 0.05, 370.0
GRID = np.linspace(-0.70, 0.70, 281)


def curves(prob, area, cx, cy, comps, lam):
    """Band tension, lever arm and couple at every trial value, plus the
    moment statics demands. Everything the calibrated root needs."""
    T, z, C = [], [], []
    for g in GRID:
        t, zz, c = FD.band_couple(prob, cx, cy, comps[0], comps[1], comps[2],
                                  area, g)
        T.append(t); z.append(zz); C.append(c)
    m_req = (lam * prob.P / 2.0) * (FD.X_CUT - ARM) / 1e6
    return np.array(T), np.array(z), np.array(C), m_req


def tc_from(prob, area, cx, cy, comps, lam, theta_known=0.0):
    """The tension the map is missing, read where theta is known."""
    T, z, C, m_req = curves(prob, area, cx, cy, comps, lam)
    i = int(np.argmin(np.abs(GRID - theta_known)))
    return (m_req - C[i]) * 1e3 / z[i]


def recover_with(prob, area, cx, cy, comps, lam, tc):
    T, z, C, m_req = curves(prob, area, cx, cy, comps, lam)
    return bracket_root((T + tc) * z / 1e3 - m_req, GRID)


def load(d, key, lam_key):
    cx, cy, ex, ey, gxy = element_strains(d["xy"], d[key], FD.NX, FD.NY)
    return cx, cy, (ex, ey, gxy), float(d[lam_key][0])


def noisy(rng, comps, sd):
    return [a + rng.normal(0.0, sd, a.shape) for a in comps]


def stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if not v.size:
        return None
    return {"mean": float(v.mean()), "sd": float(v.std()),
            "q95": float(np.quantile(v, 0.95)), "n": int(v.size)}


# ----------------------------------------------------------------------
def main() -> None:
    d = np.load(FIELDS)
    prob = DeepBeam()
    area = (prob.L / FD.NX) * (prob.H / FD.NY) / 2.0
    out = {"what": "the false-positive floor and detection threshold after a "
                   "calibration reading, against the uncalibrated ones",
           "noise": NOISE, "n_real": N_REAL, "pfa": PFA}

    print("What a commissioning reading buys beyond the zero point")
    print("=" * 66)

    cx, cy, comps0, lam0 = load(d, f"u_0.00_{DELTA}", f"lam_0.00_{DELTA}")
    scale = float(np.abs(comps0[0][cy < FD.BAND]).mean())
    sd = NOISE * scale
    print(f"  noiseless T_c on the sound state: "
          f"{tc_from(prob, area, cx, cy, comps0, lam0):+.2f} kN")

    # ---- 1. the floor: two independent readings of the same sound tie ---
    rng = np.random.default_rng(17)
    raw, cal = [], []
    for _ in range(N_REAL):
        a = noisy(rng, comps0, sd)                 # commissioning reading
        b = noisy(rng, comps0, sd)                 # the later reading
        tc = tc_from(prob, area, cx, cy, a, lam0)
        cal.append(recover_with(prob, area, cx, cy, b, lam0, tc))
        raw.append(bracket_root(curves(prob, area, cx, cy, b, lam0)[2]
                                - curves(prob, area, cx, cy, b, lam0)[3],
                                GRID))
    s_raw, s_cal = stats(raw), stats(cal)
    print("\n1. a sound tie read twice, in points of section")
    for nm, s in (("uncalibrated", s_raw), ("calibrated", s_cal)):
        print(f"   {nm:>13}: mean {100*s['mean']:+6.2f}  sd {100*s['sd']:5.2f}"
              f"  95th pct {100*s['q95']:+6.2f}   ({s['n']}/{N_REAL} rooted)")
    out["sound_tie"] = {"uncalibrated": s_raw, "calibrated": s_cal}
    thr_raw, thr_cal = s_raw["q95"], s_cal["q95"]

    # ---- 2. detection at the same false-alarm rate ----------------------
    print(f"\n2. detection at {PFA:.0%} false alarms, thresholds "
          f"{100*thr_raw:.1f} and {100*thr_cal:.1f} points")
    pod = {"uncalibrated": {}, "calibrated": {}}
    rec = {"uncalibrated": {}, "calibrated": {}}
    for th in THETA[1:]:
        cxd, cyd, compsd, lamd = load(d, f"u_{th:.2f}_{DELTA}",
                                      f"lam_{th:.2f}_{DELTA}")
        hr = hc = 0
        vr, vc = [], []
        for _ in range(N_REAL):
            a = noisy(rng, comps0, sd)
            b = noisy(rng, compsd, sd)
            tc = tc_from(prob, area, cx, cy, a, lam0)
            T, z, C, m_req = curves(prob, area, cxd, cyd, b, lamd)
            r_raw = bracket_root(C - m_req, GRID)
            r_cal = bracket_root((T + tc) * z / 1e3 - m_req, GRID)
            vr.append(r_raw); vc.append(r_cal)
            hr += bool(np.isfinite(r_raw) and r_raw > thr_raw)
            hc += bool(np.isfinite(r_cal) and r_cal > thr_cal)
        pod["uncalibrated"][f"{th:.2f}"] = hr / N_REAL
        pod["calibrated"][f"{th:.2f}"] = hc / N_REAL
        rec["uncalibrated"][f"{th:.2f}"] = stats(vr)
        rec["calibrated"][f"{th:.2f}"] = stats(vc)
        print(f"   theta {th:.2f}:  raw {100*stats(vr)['mean']:6.2f} "
              f"(err {100*(stats(vr)['mean']-th):+5.2f})  P={hr/N_REAL:.2f}"
              f"   |  cal {100*stats(vc)['mean']:6.2f} "
              f"(err {100*(stats(vc)['mean']-th):+5.2f})  P={hc/N_REAL:.2f}")
    out["threshold"] = {"uncalibrated": thr_raw, "calibrated": thr_cal}
    out["pod"] = pod
    out["recovery"] = rec

    # ---- 3. the same calibration carried in space, not in time ---------
    print("\n3. the sound zone as a spatial substitute for a baseline")
    if not ASYM_SOUND.exists():
        print("   field_asym_sound.npz absent; spatial route not tested")
        out["spatial"] = None
    else:
        a = np.load(ASYM_SOUND)
        cxa, cya, ex, ey, gxy = element_strains(a["xy"], a["u"], FD.NX, FD.NY)
        lam_a = float(a["lam"][0])
        compsa = (ex, ey, gxy)
        th1, th2 = (float(v) for v in a["theta_true"])
        print(f"   field ({th1:.2f}, {th2:.2f}), lambda {lam_a:.4f}, "
              f"R = ({float(a['R_left_kN'][0]):.1f}, "
              f"{float(a['R_right_kN'][0]):.1f}) kN")
        # T_c read on the sound right-hand zone, carried to the left cut
        x_r, x_l = 1300.0, FD.X_CUT
        arm_r = 2000.0 - ARM
        def at(xcut, arm, tc=0.0):
            T, z, C = [], [], []
            for g in GRID:
                keep = np.abs(cxa - xcut) < FD.BAND_W
                t, zz, c = FD.band_couple(prob, cxa, cya, *compsa, area, g)
                T.append(t); z.append(zz); C.append(c)
            return np.array(T), np.array(z), np.array(C)
        tc_space = tc_from(prob, area, cxa, cya, compsa, lam_a, 0.0)
        print(f"   T_c read on this member as a whole: {tc_space:+.2f} kN, "
              f"against {tc_from(prob, area, cx, cy, comps0, lam0):+.2f} on "
              f"the sound member")
        out["spatial"] = {
            "theta_true": [th1, th2], "lam": lam_a,
            "tc_on_asymmetric_kN": float(tc_space),
            "tc_on_sound_member_kN":
                float(tc_from(prob, area, cx, cy, comps0, lam0)),
            "note": "read on the whole member; a zone-restricted free body "
                    "needs the two-cut form of Section 7.13"}

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
