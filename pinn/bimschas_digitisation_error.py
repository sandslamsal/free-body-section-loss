"""Digitisation error of a pixel-read of Bimschas (2010) Fig. 5.20, measured
against the exact PDF vector coordinates recovered by bimschas_extract.py.

Reference (pixel) curves: the marker-detection digitisation held locally in
Research/P3-Pending/experimental/{vk1,vk3}_backbone.csv.  Nothing from that
study is used in the analysis; this script only quantifies how large a
pixel-digitisation error would have been, since the figure is vector artwork
and can be read exactly.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

VEC = Path("/Users/sandeshlamsal/Desktop/CSFD/Research/P4-ES/data/"
           "bimschas_backbones.csv")
PIX = Path("/Users/sandeshlamsal/Desktop/CSFD/Research/P3-Pending/experimental")


def load_vec():
    out = {}
    for r in csv.reader(VEC.open()):
        if not r or r[0].startswith("#") or r[0] == "unit":
            continue
        out.setdefault(r[0], []).append((float(r[1]), float(r[2])))
    return {k: np.array(v) for k, v in out.items()}


def main():
    vec = load_vec()
    eV, ed = [], []
    for unit, fname in (("VK1", "vk1_backbone.csv"), ("VK3", "vk3_backbone.csv")):
        pix = np.loadtxt(PIX / fname, delimiter=",", skiprows=1)
        v = vec[unit]
        for dp, Vp in pix:
            j = int(np.argmin(np.abs(v[:, 0] - dp)))
            ed.append(dp - v[j, 0])
            eV.append(Vp - v[j, 1])
        print(f"{unit}: {len(pix)} pixel points matched to vector points")
    eV, ed = np.array(eV), np.array(ed)
    print(f"force       bias {eV.mean():+.2f} kN, RMS {np.sqrt((eV**2).mean()):.2f} kN,"
          f" max |{np.abs(eV).max():.2f}| kN")
    print(f"displacement bias {ed.mean():+.3f} mm, RMS {np.sqrt((ed**2).mean()):.3f} mm,"
          f" max |{np.abs(ed).max():.3f}| mm")
    rp = np.loadtxt(PIX / "vk1_backbone.csv", delimiter=",", skiprows=1)[:, 1].max() \
        / np.loadtxt(PIX / "vk3_backbone.csv", delimiter=",", skiprows=1)[:, 1].max()
    rv = vec["VK1"][:, 1].max() / vec["VK3"][:, 1].max()
    print(f"peak ratio VK1/VK3: pixel {rp:.4f}, vector {rv:.4f} "
          f"({100*(rp/rv-1):+.3f} %)")
    print(f"elasticity to rho:  pixel {np.log(rp)/np.log(2/3):.4f}, "
          f"vector {np.log(rv)/np.log(2/3):.4f}")


if __name__ == "__main__":
    main()
