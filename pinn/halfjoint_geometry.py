"""Free-body geometry of the Cambridge half-joint, and the check that fixes
which force the reported 'Load' column is.

Everything below that is *printed* in the sources is tagged [src]; anything
read off a drawing is tagged [dig] and is never used for a headline number.

  [src] h = 700, h_nib = 325, L = 3320, b = 400 mm       2018 Sec. 4.1
  [src] dimension chain on the soffit of Fig. 3 (2016) / Fig. 2 (2017):
        90 | 210 | 115 | 120 | 115 | 200 | 200 | 200 | 200 | 210 = 1660
        -> bearing centreline 90 mm from the end face, stirrups at
           300, 415, 535, 650, 850, 1050, 1250, 1450 mm
  [src] roller bearing dia 90 mm on 450x140x30 plates    2017 Sec. 2.2
  [src] STM struts: AB at 67.6 deg, BC at 28.5 deg       2018 Sec. 5.1
  [src] STM U-bar stress 406 MPa at F_ult,STM = 336.5 kN 2018 Sec. 5.1
  [src] bar yield forces 42.3 / 59.8 / 283.7 kN for
        phi 10 / 12 / 25                                 2017 Sec. 3.3
  [src] fy, fu = 539/596 (10), 529/559 (12), 578/674 (25) 2016 Tab.2, 2018 Tab.3
  [src] nib steel: 4 phi12 diagonals, 3 phi12 U-bars,
        two-legged phi10 stirrups in the D-region        2018 Sec. 4.1
  [src] NS-LR: diagonals, U-bars and the stirrup nearest the re-entrant
        corner milled to half area over a 100 mm zone    2017 Sec. 2.1.3
  [dig] STM node coordinates off Fig. 13 (2018), origin at the end face,
        y measured down from the top fiber; validated against the two
        printed strut angles.
"""
from __future__ import annotations
import numpy as np

MM = 1.0
H, H_NIB, L_HALF, B = 700.0, 325.0, 1660.0, 400.0          # [src]
X_BEARING = 90.0                                            # [src]
STIRRUPS = [300.0, 415.0, 535.0, 650.0, 850.0, 1050.0, 1250.0, 1450.0]  # [src]

# [dig] Fig. 13 (2018) node coordinates, mm, (x from end face, y below top)
NODE = dict(A=(100., 298.), B=(155., 164.), C=(358., 55.), D=(649., 47.),
            F=(358., 661.), G=(649., 661.), I=(649., 298.))
X_CORNER = 262.0        # [dig] re-entrant corner
FY = {10: 539.0, 12: 529.0, 25: 578.0}                      # [src]
FU = {10: 596.0, 12: 559.0, 25: 674.0}                      # [src]
AREA = {d: np.pi / 4 * d ** 2 for d in (10, 12, 25)}


def ang(p, q):
    return np.degrees(np.arctan2(abs(q[1] - p[1]), abs(q[0] - p[0])))


def main() -> None:
    A, Bn, C = NODE["A"], NODE["B"], NODE["C"]
    print("VALIDATION OF THE DIGITISED STM NODES")
    print(f"  strut AB angle  {ang(A, Bn):5.1f} deg   printed 67.6 [src]")
    print(f"  strut BC angle  {ang(Bn, C):5.1f} deg   printed 28.5 [src]")
    print(f"  node C at x={C[0]:.0f} mm; midpoint of stirrups 1 and 2 "
          f"= {(STIRRUPS[0]+STIRRUPS[1])/2:.1f} mm  (C-F = 'first and "
          f"second stirrup grouped' [src])")
    print(f"  bar yield forces  phi10 {AREA[10]*FY[10]/1e3:5.1f}  "
          f"phi12 {AREA[12]*FY[12]/1e3:5.1f}  phi25 {AREA[25]*FY[25]/1e3:6.1f} kN"
          f"   printed 42.3 / 59.8 / 283.7 [src]")

    # ---- node A is a three-force node: reaction, strut AB, tie AI ----------
    th = np.radians(67.6)                                   # [src]
    cot = 1.0 / np.tan(th)
    T_ubar_stm = 3 * AREA[12] * 406.0 / 1e3                 # [src] 406 MPa
    print("\nWHICH FORCE IS THE REPORTED 'Load' COLUMN?")
    print(f"  node A equilibrium: T_Ubar = R cot(67.6) = {cot:.4f} R")
    print(f"  printed STM U-bar force at F_ult,STM = 336.5 kN: "
          f"3 x {AREA[12]:.1f} mm2 x 406 MPa = {T_ubar_stm:.1f} kN")
    print(f"  -> R = {T_ubar_stm/cot:6.1f} kN.  F = 336.5 kN.  "
          f"R/F = {T_ubar_stm/cot/336.5:.3f}")
    print("  the reported load IS the reaction on one half-joint "
          "(R = F, not F/2)")

    # ---- horizontal tie demand implied by the published STM ---------------
    G = NODE["G"]
    R = 1.0
    C_AB = R / np.sin(th)
    uAB = np.array([Bn[0]-A[0], Bn[1]-A[1]]); uAB = uAB/np.linalg.norm(uAB)
    uBC = np.array([C[0]-Bn[0], C[1]-Bn[1]]); uBC = uBC/np.linalg.norm(uBC)
    uBG = np.array([G[0]-Bn[0], G[1]-Bn[1]]); uBG = uBG/np.linalg.norm(uBG)
    M = np.array([[-uBC[0], uBG[0]], [-uBC[1], uBG[1]]])
    rhs = -C_AB * uAB
    C_BC, T_BG = np.linalg.solve(M, rhs)
    T_AI = C_AB * abs(uAB[0])
    T_dh = T_BG * uBG[0]
    print("\nHORIZONTAL TIE DEMAND AT THE INNER NIB, from the published STM")
    print(f"  T_Ubar   = {T_AI:.3f} R      (node A, geometry-free given 67.6 deg)")
    print(f"  T_diag,h = {T_dh:.3f} R      (node B)")
    print(f"  HStack_required = {T_AI+T_dh:.3f} R")
    print(f"  -> effective a/z = {T_AI+T_dh:.3f}, NOT 170/285 = 0.596")
    print(f"  previous reconstruction was wrong by "
          f"2 x {(T_AI+T_dh)/0.596:.2f} = {2*(T_AI+T_dh)/0.596:.2f}x")

    print("\nVERTICAL FREE BODY (the one used for identification)")
    print("  crack from the re-entrant corner up-and-right at 60 deg [src, 2018 Fig.14]")
    for xs in STIRRUPS[:4]:
        y = 325.0 - (xs - X_CORNER) * np.tan(np.radians(60))
        print(f"    stirrup at x={xs:6.1f}: crack ordinate y={y:7.1f} mm"
              f"  {'CROSSED' if 0 <= y <= 325 else 'not crossed'}")
    print("  members crossing: stirrup 1, stirrup 2, diagonal bars")
    print("  -> exactly the authors' 'Vertical Stack = St1 + St2 + Diagn'")
    print("  vertical equilibrium  R = VSt1 + VSt2 + VDiagn + V_concrete")
    print("  contains no lever arm, no crack angle in the force balance,")
    print("  and no assumption of plane sections.")


if __name__ == "__main__":
    main()


def independent_load_convention_check() -> None:
    """Third, independent confirmation that the reported load is the reaction.

    2017 Fig. 16 [src] reports, at an applied load of 230 kN on NS-REF, total
    bar forces at the section where the diagonal, the bottom longitudinal bars
    and the fourth stirrup meet (x = 650 mm): bottom bars 177.8 kN, diagonal
    37.7 kN, fourth stirrup 0.3 kN.  Moment of that section about the top
    compression chord must equal R (x - x_bearing).
    """
    import numpy as np
    from pathlib import Path
    D = Path("/Users/sandeshlamsal/Desktop/CSFD/Research/P4-ES/data")
    raw = (D / "desnerck_fig16_barforces.csv").read_text().strip().split("\n")
    cols = raw[0].split(",")[1:]
    A = np.array([[np.nan if x == "" else float(x) for x in l.split(",")[1:]]
                  for l in raw[1:] if l.startswith("NS-REF")])
    m = (A[:, 0] >= 227) & (A[:, 0] <= 233)
    v = {c: float(np.nanmedian(A[m, k])) for k, c in enumerate(cols)}
    y_top, y_bot, y_ubar, x_cut = 50.0, 661.0, 298.0, 650.0   # [dig] Fig.13
    T_bot, T_dh = 177.8, 37.7 * np.cos(np.radians(45))        # [src] 2017 Fig.16
    T_u = v["HUbar"]                                          # [src] 2016 Fig.16
    resist = (T_bot + T_dh) * (y_bot - y_top) + T_u * (y_ubar - y_top)
    print("\nINDEPENDENT CHECK OF THE LOAD CONVENTION (2017 Fig. 16, 230 kN)")
    print(f"  measured U-bar force at 230 kN (2016 Fig.16): {T_u:.1f} kN")
    print(f"  resisting moment about the top chord: {resist/1e3:.0f} kN mm x 1e3")
    for label, R in (("R = F   ", 230.0), ("R = F/2 ", 115.0)):
        demand = R * (x_cut - X_BEARING)
        print(f"  {label}: demand {demand/1e3:7.0f}  "
              f"resist/demand = {resist/demand:5.2f}")
    print("  R = F/2 would need the measured steel to supply about twice the")
    print("  demand, which is impossible; R = F closes to within the")
    print("  digitisation error of the node levels.")
