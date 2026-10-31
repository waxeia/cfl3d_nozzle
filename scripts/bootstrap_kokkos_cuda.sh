#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
version=${KOKKOS_VERSION:-3.7.01}
cuda_sm=${CUDA_SM:-86}
kokkos_arch=${KOKKOS_ARCH:-AMPERE86}
deps="$root/.deps"
archive="$deps/kokkos_${version}.orig.tar.gz"
source_dir="$deps/kokkos-${version}-cuda-source"
build_dir="$deps/kokkos-${version}-cuda-build"
install_dir="$deps/kokkos-cuda"
url="https://archive.ubuntu.com/ubuntu/pool/universe/k/kokkos/kokkos_${version}.orig.tar.gz"

command -v nvcc >/dev/null 2>&1 || { printf 'nvcc is required\n' >&2; exit 1; }
mkdir -p "$deps"
if [[ ! -f "$archive" ]]; then
  curl -fL "$url" -o "$archive"
fi
if [[ "$version" == 3.7.01 ]]; then
  printf '%s  %s\n' '0481b24893d1bcc808ec68af1d56ef09b82a1138a1226d6be27c3b3c3da65ceb' "$archive" | sha256sum -c -
fi
rm -rf "$source_dir" "$build_dir" "$install_dir"
mkdir -p "$source_dir"
tar -xzf "$archive" --strip-components=1 -C "$source_dir"

# Kokkos 3.7's wrapper defaults to an architecture removed by CUDA 12.x.
sed -i -E "s/^default_arch=\"sm_[0-9]+\"/default_arch=\"sm_${cuda_sm}\"/" "$source_dir/bin/nvcc_wrapper"

cmake -S "$source_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_CXX_COMPILER="$source_dir/bin/nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ARCH_${kokkos_arch}=ON \
  -DKokkos_ENABLE_TESTS=OFF
cmake --build "$build_dir" -j"${BUILD_JOBS:-2}"
cmake --install "$build_dir"

printf 'Kokkos CUDA installed in %s\n' "$install_dir"
printf 'Configure this project with:\n'
printf '  cmake -S . -B build-kokkos-cuda -DCMAKE_BUILD_TYPE=Release -DNOZZLE_ENABLE_CUDA=OFF -DKokkos_DIR=%s/lib/cmake/Kokkos -DCMAKE_CXX_COMPILER=%s/bin/nvcc_wrapper\n' "$install_dir" "$install_dir"
