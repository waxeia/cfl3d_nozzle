#!/usr/bin/env python3
"""Reproduce the frozen pre-repair solver and isolate its Roe energy error."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/muscl_repair"


def config(case, nx, reconstruction, cfl, max_iter, destination):
    values = {}
    for line in (ROOT / f"cases/{case}.cfg").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    values.update(nx=nx, ny=nx // 4, reconstruction=reconstruction, cfl=cfl,
                  max_iter=max_iter, report_every=1000, output=str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "input.cfg"
    path.write_text("".join(f"{k} = {v}\n" for k, v in values.items()))
    return path


def run(exe, cfg, destination):
    with (destination / "stdout.log").open("w") as log:
        result = subprocess.run([str(exe), str(cfg)], stdout=log, stderr=subprocess.STDOUT)
    (destination / "process_exit_code.txt").write_text(f"{result.returncode}\n")
    print(destination.name, result.returncode, flush=True)
    if result.returncode not in (0, 2):
        raise RuntimeError(f"unexpected solver failure: {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "ablation"))
    args = parser.parse_args()
    legacy = OUT / "legacy"
    exe = legacy / "nozzle_solver"
    subprocess.run(["g++","-O3","-std=c++17",str(legacy/"nozzle_solver.cpp"),"-o",str(exe)],check=True)
    if args.mode == "baseline":
        for case in ("subsonic", "choked"):
            for nx in (64, 128, 256):
                dest = OUT / "baseline" / f"{case}_{nx}x{nx//4}"
                cfg = config(case, nx, "first", 1.2 if case == "subsonic" else .6,
                             60000 if case == "subsonic" else 30000, dest)
                run(exe, cfg, dest)
    else:
        source = (legacy / "nozzle_solver.cpp").read_text()
        old = "Vec4{1,u,v,H-.5*q2},Vec4{0,-ny,nx,-ut}"
        assert source.count(old) == 1
        variant = legacy / "roe_corrected.cpp"
        variant.write_text(source.replace(old, "Vec4{1,u,v,.5*q2},Vec4{0,-ny,nx,ut}"))
        fixed = legacy / "roe_corrected"
        subprocess.run(["g++", "-O3", "-std=c++17", str(variant), "-o", str(fixed)], check=True)
        for label,old_term,new_term in (("contact_only","Vec4{1,u,v,H-.5*q2}","Vec4{1,u,v,.5*q2}"),
                                        ("shear_only","Vec4{0,-ny,nx,-ut}","Vec4{0,-ny,nx,ut}")):
            path=legacy/f"{label}.cpp"
            path.write_text(source.replace(old_term,new_term))
            binary=legacy/label
            subprocess.run(["g++","-O3","-std=c++17",str(path),"-o",str(binary)],check=True)
            dest=OUT/"experiments"/f"{label}_subsonic_64_cfl0.5"
            run(binary,config("subsonic",64,"muscl",.5,60000,dest),dest)
        for case in ("subsonic", "choked"):
            for cfl in ((1.2, .5, .1) if case == "subsonic" else (.6,)):
                for label, binary in (("original", exe), ("roe_only", fixed)):
                    dest = OUT / "experiments" / f"{label}_{case}_64_cfl{cfl}"
                    cfg = config(case, 64, "muscl", cfl, 60000, dest)
                    run(binary, cfg, dest)


if __name__ == "__main__":
    main()
