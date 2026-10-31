#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
solver_exe=${SOLVER_EXE:-$root/build-all2/nozzle_solver}
output_root=${OUTPUT_ROOT:-output}
if [[ ! -x "$solver_exe" && -x "$root/build/nozzle_solver" ]]; then
  solver_exe="$root/build/nozzle_solver"
fi
if [[ ! -x "$solver_exe" ]]; then
  printf 'solver executable not found: %s\n' "$solver_exe" >&2
  exit 1
fi
for case_name in subsonic choked; do
  if [[ "$case_name" == subsonic ]]; then max_iter=60000; else max_iter=30000; fi
  for grid in 64x16 128x32 256x64; do
    nx=${grid%x*}; ny=${grid#*x}
    cfg=$(mktemp "${TMPDIR:-/tmp}/cfl3d_nozzle_${case_name}_${grid}.XXXXXX.cfg")
    sed -e "s/^nx = .*/nx = $nx/" -e "s/^ny = .*/ny = $ny/" \
        -e "s/^max_iter = .*/max_iter = $max_iter/" \
        -e "s|^output = .*|output = ${output_root}/${case_name}_${grid}|" \
        "$root/cases/${case_name}.cfg" > "$cfg"
    set +e
    "$solver_exe" "$cfg"
    rc=$?
    set -e
    test "$rc" -eq 0 || test "$rc" -eq 2
    printf 'grid=%s case=%s config=%s rc=%s\n' "$grid" "$case_name" "$cfg" "$rc"
  done
done
