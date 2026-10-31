#!/usr/bin/env bash
set -u
echo "host=$(hostname)"
echo "compiler=$(c++ --version | head -1)"
echo "cmake=$(cmake --version | head -1)"
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -4; else echo "CUDA=unavailable"; fi
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi -L || true; else echo "GPU=unavailable"; fi
kokkos=$(find /usr/include /usr/local/include "$HOME/.local/include" -type f -name Kokkos_Core.hpp 2>/dev/null | head -1)
if [[ -n "$kokkos" ]]; then echo "Kokkos=$kokkos"; else echo "Kokkos=not-found"; fi
grep -E 'OpenMP_CXX_FLAGS|OpenMP_CXX_FOUND' build/CMakeCache.txt 2>/dev/null || true
