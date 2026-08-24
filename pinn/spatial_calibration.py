"""Can a sound ZONE calibrate T_c, when a sound MEMBER is unavailable?

Section 7.10 identifies the concrete tension the constitutive map neglects
from a reading taken where theta is known, and carries it into the
deteriorated states. On a laboratory specimen that reading is the member's
own sound state, taken before deterioration. A structure first instrumented
after decades of exposure has no such reading, and the substitute offered
there is a zone of the same member known to be sound, read at the same time.

Section 7.13 is a reason to doubt the substitute: deterioration in one span
moves the load path and degrades the recovery of a zone that is itself
intact. The field the study already had, (0.30, 0.10), cannot settle it
because neither zone is sound. This runs the test on a purpose-generated
(0.30, 0.00) field, in which the right-hand zone really is undamaged, and
reports both fields so the comparison is visible.

The cut, arm and demand convention is two_parameter's, which is the
validated one: cut 2 takes its free body toward the RIGHT support, so its
demand is R_right * (a_right - x_cut) and not the left-hand form. The
(0.30, 0.10) row reproduces the published (0.278, 0.019) exactly, which is
what says the setup is right.

Run:  /usr/local/bin/python3.12 spatial_calibration.py
"""
import sys, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H)); sys.path.insert(0, str(H.parent / 'oracle'))
import two_parameter as TP
from problem import DeepBeam
from recover_utils import element_strains, bracket_root

prob = DeepBeam()
GRID = np.linspace(-0.70, 0.70, 281)

def couple_at(sysm, x_cut, th_pair):
    return TP.band_couple_two(prob, sysm.cx, sysm.cy, sysm.ex, sysm.ey,
                              sysm.gxy, sysm.area, x_cut, *th_pair)

def scan(sysm, x_cut, which):
    """T, z, C over the grid, varying only the parameter that cut sees."""
    T=[];z=[];C=[]
    for g in GRID:
        pair = (g, 0.0) if which == 0 else (0.0, g)
        t, zz, c = couple_at(sysm, x_cut, pair)
        T.append(t); z.append(zz); C.append(c)
    return map(np.array, (T, z, C))

for tag, f in (("sound zone   (0.30, 0.00)", 'field_asym_sound.npz'),
               ("both damaged (0.30, 0.10)", 'field_asym.npz')):
    p = H.parent / 'oracle' / f
    if not p.exists():
        print(f"{tag}: absent"); continue
    a = np.load(p)
    sysm = TP.CoupleSystem(prob, a['xy'], a['u'])
    th = [float(v) for v in a['theta_true']]
    RL, RR = float(a['R_left_kN'][0]), float(a['R_right_kN'][0])
    dem = TP.moment_demands(RL, RR)
    print(f"\n{tag}   lambda {float(a['lam'][0]):.4f}   "
          f"R = ({RL:.1f}, {RR:.1f}) kN   demands {dem[0]:.1f}, {dem[1]:.1f} kN m")

    T1,z1,C1 = scan(sysm, TP.CUTS[0], 0)
    T2,z2,C2 = scan(sysm, TP.CUTS[1], 1)
    r1 = bracket_root(C1 - dem[0], GRID)
    r2 = bracket_root(C2 - dem[1], GRID)
    print(f"  uncalibrated:  cut1 {r1:+.4f} (err {100*(r1-th[0]):+5.1f} pp)"
          f"   cut2 {r2:+.4f} (err {100*(r2-th[1]):+5.1f} pp)")

    # T_c read at cut 2 ASSUMING that zone is sound, carried to cut 1
    i0 = int(np.argmin(np.abs(GRID - 0.0)))
    tc = (dem[1] - C2[i0]) * 1e3 / z2[i0]
    r1c = bracket_root((T1 + tc) * z1 / 1e3 - dem[0], GRID)
    print(f"  T_c from cut 2 assuming it is sound: {tc:+7.2f} kN")
    print(f"  cut 1 calibrated in space: {r1c:+.4f} "
          f"(err {100*(r1c-th[0]):+5.1f} pp)")
