#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
dest="$root/.deps/kokkos"
packages="$root/.deps/packages"
mkdir -p "$dest" "$packages"
cd "$packages"
apt download libkokkos-dev libkokkos3.7 libmemkind-dev libmemkind0 libhwloc-dev libhwloc15 libnuma-dev libnuma1
for archive in libkokkos-dev_*.deb libkokkos3.7_*.deb libmemkind-dev_*.deb libmemkind0_*.deb libhwloc-dev_*.deb libhwloc15_*.deb libnuma-dev_*.deb libnuma1_*.deb; do
  dpkg-deb -x "$archive" "$dest"
done
ln -sfn usr/include "$dest/include"
ln -sfn usr/lib "$dest/lib"
printf 'Kokkos extracted to %s\n' "$dest/usr"
printf 'Configure with: cmake -S . -B build-kokkos -DNOZZLE_KOKKOS_ROOT=%s\n' "$dest/usr"
