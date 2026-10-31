#!/usr/bin/env python3
"""Run the CPU-only acceptance matrix; never accept max_iter as convergence."""
import argparse
from pathlib import Path
from reproduce_muscl_legacy import config, run, OUT, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=("small", "muscl", "first", "euler"))
    p.add_argument("--exe", type=Path, default=ROOT / "build-muscl/nozzle_solver")
    p.add_argument("--cfl", type=float)
    p.add_argument("--output-group")
    p.add_argument("--max-iter", type=int, default=400000)
    args = p.parse_args()
    grids = (32,) if args.group == "small" else (64,) if args.group == "euler" else (64, 128, 256)
    for case in ("subsonic", "choked"):
        for nx in grids:
            dest = OUT / (args.output_group or args.group) / f"{case}_{nx}x{nx//4}"
            reconstruction = "first" if args.group == "first" else "muscl"
            cfl = (1.2 if case == "subsonic" else .6) if args.group == "first" else .8
            if args.cfl is not None:
                cfl = args.cfl
            cfg = config(case, nx, reconstruction, cfl, args.max_iter, dest)
            integrator = "euler" if args.group in ("first", "euler") else "ssprk3"
            with cfg.open("a") as f:
                f.write(f"time_integrator = {integrator}\n")
            run(args.exe, cfg, dest)
            status = dict(line.split("=", 1) for line in (dest / "run_status.txt").read_text().splitlines())
            if status["status"] != "converged":
                raise RuntimeError(f"acceptance failed: {dest}")


if __name__ == "__main__":
    main()
