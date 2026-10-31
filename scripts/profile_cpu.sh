#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cfg=$(mktemp "${TMPDIR:-/tmp}/cfl3d_profile.XXXXXX.cfg")
sed -e 's/^nx = .*/nx = 128/' -e 's/^ny = .*/ny = 32/' \
    -e 's/^max_iter = .*/max_iter = 3000/' \
    -e 's|^output = .*|output = output/profile_subsonic_128x32|' "$root/cases/subsonic.cfg" > "$cfg"
for rep in 1 2 3 4 5; do
  /usr/bin/time -f "repeat=$rep wall=%e" "$root/build/nozzle_solver" "$cfg" >"$root/output/profile_${rep}.log" 2>&1 || test $? -eq 2
done
printf 'profile logs: output/profile_1.log ... output/profile_5.log\n'
