#!/usr/bin/env python3
"""Summarize converged nozzle runs and make compact M2 comparison plots."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("NOZZLE_OUTPUT_ROOT", ROOT / "output"))
if not OUT.is_absolute():
    OUT = ROOT / OUT
REFERENCE = ROOT / "output" / "reference"
grids = ["64x16", "128x32", "256x64"]

def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def interp(ref, x, key):
    for a, b in zip(ref, ref[1:]):
        xa, xb = float(a["x"]), float(b["x"])
        if xa <= x <= xb:
            t = (x-xa)/(xb-xa)
            return (1-t)*float(a[key]) + t*float(b[key])
    return float(ref[-1][key])

summary = []
plot_data = {}
for col, case in enumerate(("subsonic", "choked")):
    ref = rows(REFERENCE / f"{case}.csv")
    for grid in grids:
        run = OUT / f"{case}_{grid}"
        hist = rows(run / "residual.csv")
        field = rows(run / "field.csv")
        last = hist[-1]
        ny = max(int(r["j"]) for r in field) + 1
        center = [r for r in field if int(r["j"]) == ny//2]
        err = max(abs(float(r["M"])-interp(ref, float(r["x"]), "M")) for r in center)
        imbalance = float(last["mass_imbalance"])
        summary.append((case, grid, last, err, imbalance))
        raw = [(int(r["iter"])+1, float(r["normalized"])) for r in hist]
        stride = max(1, len(raw)//500)
        plot_data[(case, grid)] = raw[::stride] + ([raw[-1]] if raw[-1] != raw[::stride][-1] else [])

# Dependency-free SVG plot: log10 residual versus iteration, one panel per case.
W, H, panel_w, left, top = 1100, 470, 500, 55, 35
colors = {"64x16":"#1f77b4", "128x32":"#d62728", "256x64":"#2ca02c"}
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">', '<rect width="100%" height="100%" fill="white"/>']
for p, case in enumerate(("subsonic", "choked")):
    x0 = left + p*panel_w; y0 = top; pw, ph = 430, 350
    svg += [f'<text x="{x0+pw/2}" y="20" text-anchor="middle" font-family="sans-serif" font-size="16">{case}</text>', f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="black"/><line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="black"/>']
    for grid in grids:
        d=plot_data[(case,grid)]; xmax=max(x for x,y in d); pts=[]
        for x,y in d:
            xx=x0+pw*x/xmax; yy=y0+ph*max(0,min(1,(-__import__('math').log10(max(y,1e-12)))/6))
            pts.append(f'{xx:.1f},{yy:.1f}')
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[grid]}" stroke-width="1.3"/>')
        svg.append(f'<text x="{x0+pw-95}" y="{y0+18+18*grids.index(grid)}" font-family="sans-serif" font-size="12" fill="{colors[grid]}">{grid}</text>')
    for k in range(7):
        yy=y0+ph*k/6; svg.append(f'<text x="{x0-8}" y="{yy+4}" text-anchor="end" font-family="sans-serif" font-size="10">1e-{k}</text>')
svg.append('</svg>')
(OUT / "residual_grid_comparison.svg").write_text("\n".join(svg))

with (OUT / "grid_metrics.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "grid", "status", "iterations", "normalized_residual", "mass_imbalance", "centerline_max_abs_M_error"])
    for case, grid, last, err, imbalance in summary:
        status = (OUT / f"{case}_{grid}" / "run_status.txt").read_text().splitlines()[0].split("=", 1)[1]
        w.writerow([case, grid, status, int(last["iter"])+1, last["normalized"], imbalance, err])
print("wrote", OUT / "grid_metrics.csv")
print("wrote", OUT / "residual_grid_comparison.svg")
