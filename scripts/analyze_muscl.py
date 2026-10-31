#!/usr/bin/env python3
"""Strict acceptance audit plus standalone scientific figures (matplotlib)."""
import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/muscl_repair"


def kv(path):
    return dict((k.strip(), v.strip()) for k, v in
                (s.split("=", 1) for s in path.read_text().splitlines() if "=" in s))


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def zeroed_directional_slopes(field, nx, ny):
    """Final-state minmod extrema audit; ignore roundoff-sized gradients."""
    q = [[float(r[k]) for k in ("rho", "u", "v", "p")] for r in field]
    x, y = ([float(r[k]) for r in field] for k in ("x", "y"))
    zeroed, total = 0, 0
    def minmod(a, b):
        return 0. if a*b <= 0 else min((a, b), key=abs)
    for j in range(ny):
        for i in range(nx):
            n = j*nx+i
            b, t = max(0, j-1)*nx+i, min(ny-1, j+1)*nx+i
            l, r = j*nx+max(0, i-1), j*nx+min(nx-1, i+1)
            for k in range(4):
                def sy(a, z):
                    return (q[z][k]-q[a][k])/(y[z]-y[a])
                if 0 < j < ny-1:
                    a, d = sy(b, n), sy(n, t)
                    gy = minmod(a, d);total += 1
                    zeroed += a*d <= 0 and max(abs(a), abs(d)) > 1e-12
                else:
                    gy = sy(n, t) if j == 0 else sy(b, n)
                if 0 < i < nx-1:
                    a = (q[n][k]-q[l][k]-gy*(y[n]-y[l]))/(x[n]-x[l])
                    d = (q[r][k]-q[n][k]-gy*(y[r]-y[n]))/(x[r]-x[n])
                    total += 1
                    zeroed += a*d <= 0 and max(abs(a), abs(d)) > 1e-12
    return zeroed, total


def reference(case, x):
    g = 1.4
    def area_mach(m):
        return (2/(g+1)*(1+(g-1)*m*m/2))**((g+1)/(2*(g-1)))/m
    astar = 1 if case == "choked" else 1.5/area_mach(math.sqrt(2/(g-1)*((1/.92)**((g-1)/g)-1)))
    ratio = (1+.5*((x-5)/5)**2)/astar
    sup = case == "choked" and x > 5
    lo, hi = (1., 20.) if sup else (1e-12, 1.)
    if abs(ratio-1) < 1e-14:
        m = 1.
    else:
        for _ in range(100):
            mid = (lo+hi)/2
            move_low = area_mach(mid) < ratio if sup else area_mach(mid) > ratio
            if move_low:
                lo = mid
            else:
                hi = mid
        m = (lo+hi)/2
    t = 1/(1+.2*m*m)
    return m, t**3.5, astar*math.sqrt(g)*(2/(g+1))**((g+1)/(2*(g-1)))


def metrics(group, case, nx):
    path = OUT/("muscl_final" if group == "muscl" else group)/f"{case}_{nx}x{nx//4}"
    meta = kv(path/"run_status.txt")
    cfg = kv(path/"config.cfg")
    for key, expected in dict(length=10, throat_half_height=.5, area_ratio=1.5,
                              gamma=1.4, R=1, p0=1, T0=1,
                              outlet_pressure=.16 if case == "choked" else .92).items():
        assert abs(float(cfg[key])-expected) < 1e-12, ("reference requires the benchmark geometry/gas/BC", path, key)
    history, section, field, flux = (rows(path/name) for name in
                                    ("residual.csv", "sections.csv", "field.csv", "face_mass.csv"))
    last = history[-1]
    # Audit saved samples against the original three thresholds, without relying
    # on the solver's exit label alone.
    assert meta["status"] == "converged", path
    for key, threshold in (("normalized", "residual_drop"), ("mass_imbalance", "mass_tolerance"),
                           ("window_change", "stability_tolerance")):
        assert math.isfinite(float(last[key])) and float(last[key]) <= float(cfg[threshold]), (path, key)
    assert meta["normalized_residual"] == last["normalized"]
    assert meta["mass_imbalance"] == last["mass_imbalance"]
    assert int(meta["iterations"]) == len(history)
    face_mass = [float(r["mass_flux"]) for r in flux]
    assert abs(float(last["inlet_mass"])+face_mass[0]) < 2e-13
    assert abs(float(last["outlet_mass"])-face_mass[-1]) < 2e-13
    assert all(math.isfinite(float(r[k])) for r in field for k in ("rho", "u", "v", "p", "M"))
    exit_normal_mach = [float(r["u"])/math.sqrt(1.4*float(r["p"])/float(r["rho"])) for r in field if int(r["i"]) == nx-1]
    if case == "choked":
        assert min(exit_normal_mach) > 1, ("outlet is not everywhere supersonic", path)
    else:
        assert max(float(r["M"]) for r in field) < 1, ("subsonic case changed branch", path)
    errors_m = [abs(float(r["mean_local_M"])-reference(case, float(r["x"]))[0]) for r in section]
    errors_p = [abs(float(r["mean_p"])-reference(case, float(r["x"]))[1]) for r in section]
    center = [r for r in field if int(r["j"]) == nx//8]
    center_error = max(abs(float(r["M"])-reference(case, float(r["x"]))[0]) for r in center)
    symmetry = 0.
    for r, reflected in zip(field, [field[(nx//4-1-j)*nx+i] for j in range(nx//4) for i in range(nx)]):
        for key in ("rho", "u", "v", "p"):
            symmetry = max(symmetry, abs(float(r[key]) - (-1 if key == "v" else 1)*float(reflected[key])))
    recon = rows(path/"reconstruction.csv")
    active = sum(float(r["max_primitive_face_increment"]) > 1e-13 for r in recon)
    if group == "muscl":
        assert active > .95*nx*(nx//4), ("inactive reconstruction", path)
    row = dict(group=group, case=case, grid=f"{nx}x{nx//4}", status=meta["status"],
               iterations=meta["iterations"], residual=last["normalized"],
               mass_imbalance=last["mass_imbalance"], window_change=last["window_change"],
               section_mass_spread=(max(face_mass)-min(face_mass))/abs(face_mass[0]),
               outlet_mass=face_mass[-1], reference_mass=reference(case, 0)[2],
               reference_mass_relative_error=abs(face_mass[-1]/reference(case, 0)[2]-1),
               mean_M_L1_error=sum(errors_m)/nx, mean_M_Linf_error=max(errors_m),
               mean_p_L1_error=sum(errors_p)/nx, center_M_Linf_error=center_error,
               throat_mean_M=(float(section[nx//2-1]["mean_local_M"])+float(section[nx//2]["mean_local_M"]))/2,
               exit_mean_M=section[-1]["mean_local_M"], exit_mean_p=section[-1]["mean_p"],
               min_exit_normal_M=min(exit_normal_mach),
               min_face_rho=meta["min_face_rho_ever"], min_face_p=meta["min_face_p_ever"],
               positivity_cell_evaluations=meta["positivity_cell_evaluations"], step_retries=meta["step_retries"],
               active_cells=active, total_cells=nx*nx//4, symmetry_max_error=symmetry)
    zeroed, directions = zeroed_directional_slopes(field, nx, nx//4) if group == "muscl" else (0, 0)
    row.update(final_minmod_zeroed_variable_directions=zeroed, interior_variable_directions=directions,
               min_primitive_face_increment=min(float(r["max_primitive_face_increment"]) for r in recon),
               max_primitive_face_increment=max(float(r["max_primitive_face_increment"]) for r in recon))
    return row, history, section, field


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    table, data = [], {}
    for group in ("first", "muscl"):
        for case in ("subsonic", "choked"):
            for nx in (64, 128, 256):
                row, *dataset = metrics(group, case, nx)
                table.append(row)
                data[group, case, nx] = dataset
    for row in table:
        nx = int(row["grid"].split("x")[0])
        first_field = data["first", row["case"], nx][2]
        muscl_field = data["muscl", row["case"], nx][2]
        row["first_vs_muscl_max_M_difference"] = max(abs(float(a["M"])-float(b["M"])) for a, b in zip(first_field, muscl_field))
    with (OUT/"validation_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
    lines = ["|工况|网格|迭代|归一化残差|质量不平衡|200 步变化|全截面流量极差/入口流量|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        if r["group"] == "muscl":
            lines.append(f'|{r["case"]}|{r["grid"]}|{r["iterations"]}|{float(r["residual"]):.4e}|{float(r["mass_imbalance"]):.4e}|{float(r["window_change"]):.4e}|{r["section_mass_spread"]:.4e}|')
    (OUT/"validation_table.md").write_text("\n".join(lines)+"\n")
    print("All 12 current-code runs pass the unchanged criteria; wrote validation_metrics.csv")
    if args.no_plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25})
    colors = {64: "#0072B2", 128: "#D55E00", 256: "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    fig.suptitle("New MUSCL: SSPRK3, CFL=2.0; unchanged convergence thresholds", fontsize=11)
    for ax, case in zip(axes, ("subsonic", "choked")):
        for nx in colors:
            history = data["muscl", case, nx][0]
            sampled = history[::max(1, len(history)//2000)] + [history[-1]]
            ax.semilogy([int(r["iter"]) for r in sampled], [float(r["normalized"]) for r in sampled], color=colors[nx], label=f"MUSCL {nx}x{nx//4}")
        legacy = rows(OUT/"experiments"/f'original_{case}_64_cfl{1.2 if case=="subsonic" else .6}'/"residual.csv")[::50]
        ax.semilogy([int(r["iter"]) for r in legacy], [float(r["normalized"]) for r in legacy], color=".5", alpha=.7, label=f'Original 64x16: Euler, CFL={1.2 if case=="subsonic" else .6}')
        ax.axhline(1e-6, ls="--", color="black", lw=1)
        ax.set(title=case, xlabel="Pseudo-time iterations", ylabel="Normalized residual", ylim=(7e-7, 2))
        ax.legend(fontsize=8)
    fig.savefig(OUT/"residual_comparison.png", dpi=180);fig.savefig(OUT/"residual_comparison.svg");plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for col, case in enumerate(("subsonic", "choked")):
        x = np.linspace(0, 10, 1001);reference_values = [reference(case, xx) for xx in x]
        for row, (key, label, refindex) in enumerate((("mean_local_M", "Section-mean local Mach", 0), ("mean_p", "Section-mean pressure", 1))):
            ax = axes[row, col];ax.plot(x, [r[refindex] for r in reference_values], "k--", label="Quasi-1D model")
            for nx in colors:
                section = data["muscl", case, nx][1]
                ax.plot([float(r["x"]) for r in section], [float(r[key]) for r in section], color=colors[nx], label=f"MUSCL {nx}x{nx//4}")
            section = data["first", case, 256][1]
            ax.plot([float(r["x"]) for r in section], [float(r[key]) for r in section], color=".6", ls=":", label="First order 256x64")
            ax.set(title=case if row == 0 else "", xlabel="x", ylabel=label);ax.legend(fontsize=8)
    fig.savefig(OUT/"flow_comparison.png", dpi=180);fig.savefig(OUT/"flow_comparison.svg");plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), layout="constrained")
    for ax, case in zip(axes, ("subsonic", "choked")):
        field = data["muscl", case, 256][2]
        x = np.array([float(r["x"]) for r in field]).reshape(64, 256)
        y = np.array([float(r["y"]) for r in field]).reshape(64, 256)
        m = np.array([float(r["M"]) for r in field]).reshape(64, 256)
        plot = ax.contourf(x, y, m, levels=35, cmap="viridis")
        ax.set(title=f"{case}: converged MUSCL 256x64", xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(plot, ax=ax, label="Mach")
    fig.savefig(OUT/"mach_fields.png", dpi=180);plt.close(fig)
    wave = rows(OUT/"accuracy/entropy_wave.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    for name in ("first", "muscl"):
        subset = [r for r in wave if r["reconstruction"] == name]
        n = np.array([int(r["nx"]) for r in subset]);e = np.array([float(r["L1_rho"]) for r in subset])
        ax.loglog(n, e, "o-", label=name)
    ax.loglog(n, e[-1]*(n[-1]/n)**2, "k--", label="slope 2")
    ax.set(xlabel="Axial cells", ylabel="Density L1 error after one period", title="Independent smooth entropy-wave accuracy")
    ax.legend();fig.savefig(OUT/"entropy_wave_accuracy.png", dpi=180);plt.close(fig)


if __name__ == "__main__":
    main()
