"""Exact extraction of the measured force-displacement backbones of Test Units
VK1, VK2 and VK3 from Bimschas (2010), ETH Zurich diss. no. 18849,
DOI 10.3929/ethz-a-006050338.

Figures 5.20 (VK1 left, VK3 right; PDF p. 350) and 5.22 (VK2; PDF p. 352) are
VECTOR artwork, not raster.  The plotted series (III) -- "measured
force-displacement backbone for south excursion of the 1st cycles" -- is the
only bold polyline (stroke width 0.766 pt) and carries filled round markers
(fill+stroke paths, 3.67 pt diameter).  Both are recovered as exact PDF user
-space coordinates, so no pixel digitisation and no eye-reading is involved.

Axis calibration comes from the tick marks, which are unambiguous:
11 x ticks spanning 0..100 mm and 7 y ticks spanning 0..1200 kN in both
figures.  The residual of the linear tick fit is reported.

Writes  data/bimschas_backbones.csv  and prints the calibration residuals.
"""
from __future__ import annotations

import csv
from pathlib import Path

import fitz
import numpy as np

THESIS = Path("/Users/sandeshlamsal/Desktop/CSFD/Research/P3-Pending/"
              "experimental/bimschas2010_thesis.pdf")
OUT = Path("/Users/sandeshlamsal/Desktop/CSFD/Research/P4-ES/data")

BOLD_W = 0.766          # stroke width of series (III)
X_STEP_MM, Y_STEP_KN = 10.0, 200.0


def _axes(page):
    """Return the axis frames as (x_left, y_base, x_right, y_top) tuples."""
    hor, ver = [], []
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] != "l":
                continue
            (x0, y0), (x1, y1) = it[1], it[2]
            if abs(y1 - y0) < 0.05 and abs(x1 - x0) > 40:
                hor.append((round(y0, 2), min(x0, x1), max(x0, x1)))
            if abs(x1 - x0) < 0.05 and abs(y1 - y0) > 40:
                ver.append((round(x0, 2), min(y0, y1), max(y0, y1)))
    frames = []
    for xv, ytop, ybot in sorted(set(ver)):
        for yh, xl, xr in sorted(set(hor)):
            if abs(xl - xv) < 1.0 and abs(yh - ybot) < 1.0:
                frames.append((xv, yh, xr, ytop))
    return frames


def _ticks(page, frame):
    """Tick coordinates on the x and y axes of one panel."""
    x0, ybase, x1, ytop = frame
    xt, yt = [], []
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] != "l":
                continue
            (ax, ay), (bx, by) = it[1], it[2]
            if abs(bx - ax) < 0.05 and 0.5 < abs(by - ay) < 6 \
                    and max(ay, by) > ybase - 0.5 and min(ay, by) < ybase + 0.5 \
                    and x0 - 0.5 <= ax <= x1 + 0.5:
                xt.append(ax)
            if abs(by - ay) < 0.05 and 0.5 < abs(bx - ax) < 6 \
                    and max(ax, bx) > x0 - 0.5 and min(ax, bx) < x0 + 0.5 \
                    and ytop - 0.5 <= ay <= ybase + 0.5:
                yt.append(ay)
    return np.array(sorted(set(np.round(xt, 2)))), \
        np.array(sorted(set(np.round(yt, 2))))


def _uniform_run(vals, n_expected, tol=0.05):
    """Best (lowest residual) run of >= n_expected uniformly spaced values.

    A stray dark segment can masquerade as a tick, so every candidate run is
    built and the one whose linear fit has the smallest residual is kept.
    """
    runs = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            step = vals[j] - vals[i]
            if step <= 1.0:
                continue
            seq, nxt = [vals[i]], vals[i] + step
            while True:
                k = int(np.argmin(np.abs(vals - nxt)))
                if abs(vals[k] - nxt) > tol * step:
                    break
                seq.append(vals[k])
                nxt = vals[k] + step
            if len(seq) >= n_expected:
                a = np.array(seq)
                idx = np.arange(len(a))
                r = float(np.max(np.abs(np.polyval(np.polyfit(idx, a, 1), idx) - a)))
                runs.append((r, len(a), a))
    if not runs:
        return np.array(vals[:1])
    runs.sort(key=lambda t: (t[0], -t[1]))
    return runs[0][2]


def _calibrate(xt, yt):
    xr = _uniform_run(xt, 11)
    yr = _uniform_run(yt, 7)
    ix, iy = np.arange(len(xr)), np.arange(len(yr))
    px = np.polyfit(ix, xr, 1)          # pt per 10 mm
    py = np.polyfit(iy, yr, 1)          # pt per 200 kN (descending in kN)
    res_x = float(np.max(np.abs(np.polyval(px, ix) - xr)))
    res_y = float(np.max(np.abs(np.polyval(py, iy) - yr)))
    return dict(n_x=len(xr), n_y=len(yr),
                x0=px[1], sx=px[0] / X_STEP_MM,           # pt per mm
                y0=yr[-1], sy=abs(py[0]) / Y_STEP_KN,     # pt per kN (y down)
                res_x_mm=res_x / (px[0] / X_STEP_MM),
                res_y_kN=res_y / (abs(py[0]) / Y_STEP_KN))


def _series(page, frame):
    """Bold polyline vertices + filled marker centers inside one panel."""
    x0, ybase, x1, ytop = frame
    inside = lambda x, y: x0 - 2 < x < x1 + 2 and ytop - 2 < y < ybase + 2
    verts, marks = [], []
    for dr in page.get_drawings():
        w = dr.get("width") or 0.0
        r = dr["rect"]
        cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
        if not inside(cx, cy):
            continue
        if abs(w - BOLD_W) < 1e-3 and len(dr["items"]) > 2:
            pts = [dr["items"][0][1]] + [it[2] for it in dr["items"]
                                         if it[0] == "l"]
            verts = [(float(p[0]), float(p[1])) for p in pts]
        if dr["type"] == "fs" and 3.5 < r.width < 4.1 \
                and abs(r.width - r.height) < 0.2:
            marks.append((cx, cy))
    marks.sort()
    return verts, marks


def extract(pdf_page, panel_names):
    doc = fitz.open(THESIS)
    page = doc[pdf_page - 1]
    frames = _axes(page)
    frames.sort()
    out = {}
    for name, fr in zip(panel_names, frames):
        xt, yt = _ticks(page, fr)
        cal = _calibrate(xt, yt)
        verts, marks = _series(page, fr)
        # markers are the data points; the polyline is only their connector.
        # keep polyline vertices that carry no marker (e.g. the origin)
        pts = list(marks)
        for v in verts:
            if all((v[0] - m[0]) ** 2 + (v[1] - m[1]) ** 2 > 0.25 for m in pts):
                pts.append(v)
        pts.sort()
        d = np.array([(p[0] - cal["x0"]) / cal["sx"] for p in pts])
        V = np.array([(cal["y0"] - p[1]) / cal["sy"] for p in pts])
        out[name] = (d, V, cal, len(marks), len(verts))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    res.update(extract(350, ["VK1", "VK3"]))
    res.update(extract(352, ["VK2"]))

    rows = []
    for name in ("VK1", "VK2", "VK3"):
        d, V, cal, nm, nv = res[name]
        print(f"{name}: {len(d)} points ({nm} markers, {nv} polyline vertices); "
              f"{cal['n_x']} x-ticks, {cal['n_y']} y-ticks; "
              f"tick-fit residual {cal['res_x_mm']:.4f} mm / "
              f"{cal['res_y_kN']:.3f} kN")
        i = int(np.argmax(V))
        print(f"   peak {V[i]:.1f} kN at {d[i]:.2f} mm; "
              f"last {V[-1]:.1f} kN at {d[-1]:.2f} mm")
        print("   " + "  ".join(f"({dd:.2f},{vv:.1f})" for dd, vv in zip(d, V)))
        for dd, vv in zip(d, V):
            rows.append((name, round(float(dd), 3), round(float(vv), 1)))

    with (OUT / "bimschas_backbones.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "delta_top_mm", "V_kN"])
        w.writerow(["# Bimschas (2010) ETH diss. 18849, DOI 10.3929/ethz-a-006050338"])
        w.writerow(["# measured backbone, south excursion of the 1st cycles"])
        w.writerow(["# VK1,VK3 from Fig. 5.20 (PDF p.350); VK2 from Fig. 5.22 (PDF p.352)"])
        w.writerow(["# extracted from the PDF vector paths, not pixel-digitised"])
        w.writerows(rows)
    print(f"\nwrote {OUT / 'bimschas_backbones.csv'}")


if __name__ == "__main__":
    main()
