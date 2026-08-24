"""Extract the Cambridge half-joint open data (Desnerck, Lees & Morley) into csv.

Sources, all CC BY 4.0:
  2016_Desnerck_reinforcement_layout.xlsx   sheet 'Figure 16' -- measured bar
      forces at the inner nib vs applied load, specimens NS-REF/ND/NU/RS
  2016 ... sheet 'Table 5'                  -- failure loads and modes
  2017_Desnerck_local_bar_reductions.xlsx   sheet 'Figure 10' -- failure loads
      of nine specimens including NS-LR (nib bars milled to 50 % area)
  2017 ... sheets 'Figure 14','Figure 15'   -- bottom-bar stresses / force
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import openpyxl

SRC = Path("/private/tmp/claude-501/-Users-sandeshlamsal-Desktop-CSFD/"
           "143fd6b6-0188-4cb8-a8d6-edca2ae2cc3b/scratchpad/cambridge_halfjoint")
OUT = Path("/Users/sandeshlamsal/Desktop/CSFD/Research/P4-ES/data")

COLS = ["load_kN", "Ubar", "Diagn", "VSt1", "VSt2", "HDiagn", "VDiagn",
        "HUbar", "HStack", "St1", "St1St2", "VStack"]


def fig16() -> dict[str, np.ndarray]:
    wb = openpyxl.load_workbook(SRC / "2016_Desnerck_reinforcement_layout.xlsx",
                                read_only=True, data_only=True)
    rows = list(wb["Figure 16"].iter_rows(values_only=True))
    wb.close()
    hdr = rows[3]
    off = {v.strip(): j for j, v in enumerate(hdr)
           if isinstance(v, str) and v.strip().startswith("NS-")}
    data = {}
    for name, j in off.items():
        recs = []
        for r in rows[7:]:
            v = r[j:j + 12]
            if v[0] is None:
                continue
            recs.append([float(x) if isinstance(x, (int, float)) else np.nan
                         for x in v])
        data[name] = np.array(recs)
    return data


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = fig16()
    hdr = ",".join(["specimen"] + COLS)
    lines = [hdr]
    for nm, a in d.items():
        for row in a:
            lines.append(nm + "," + ",".join("" if np.isnan(x) else f"{x:.4f}"
                                             for x in row))
    (OUT / "desnerck_fig16_barforces.csv").write_text("\n".join(lines) + "\n")
    print("wrote desnerck_fig16_barforces.csv")
    for nm, a in d.items():
        i = int(np.nanargmax(a[:, 0]))
        print(f"\n{nm}: n={len(a)}  load {np.nanmin(a[:,0]):.2f} .. "
              f"{np.nanmax(a[:,0]):.2f} kN")
        print("   at peak load: " + "  ".join(
            f"{c}={a[i,k]:.1f}" for k, c in enumerate(COLS)
            if not np.isnan(a[i, k])))
        # column-wise maxima (peak load row may not hold the peak of each)
        print("   col maxima:   " + "  ".join(
            f"{c}={np.nanmax(a[:,k]):.1f}" for k, c in enumerate(COLS)))


if __name__ == "__main__":
    main()
