#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cpu_exe=${CPU_EXE:-$root/build-all2/nozzle_solver}
cuda_exe=${CUDA_EXE:-$root/build-all2/nozzle_cuda}
kokkos_omp_exe=${KOKKOS_OMP_EXE:-$root/build-all2/nozzle_kokkos}
kokkos_cuda_exe=${KOKKOS_CUDA_EXE:-$root/build-kokkos-cuda3/nozzle_kokkos}
repeats=${REPEATS:-5}
threads=${OMP_NUM_THREADS:-8}
summary="$root/output/backend_profile_128x32.csv"
mkdir -p "$root/output/backend_profile_runs"
printf 'backend,repeat,status,iterations,seconds,transfer_seconds,wall_seconds,normalized_residual,mass_imbalance,window_change\n' > "$summary"

value() {
  awk -F= -v key="$2" '$1==key {print $2}' "$1"
}

run_one() {
  backend=$1
  rep=$2
  exe=$3
  run_dir="$root/output/backend_profile_runs/${backend}_${rep}"
  cfg=$(mktemp "${TMPDIR:-/tmp}/cfl3d_backend_${backend}_${rep}.XXXXXX.cfg")
  backend_key=serial
  if [[ "$backend" == openmp ]]; then backend_key=openmp; fi
  sed -e "s/^backend = .*/backend = $backend_key/" -e "s|^output = .*|output = $run_dir|" "$root/cases/choked_perf_128x32.cfg" > "$cfg"
  start_ns=$(date +%s%N)
  if [[ "$backend" == kokkos-openmp ]]; then
    OMP_NUM_THREADS="$threads" OMP_PROC_BIND=spread OMP_PLACES=threads "$exe" "$cfg" "$run_dir" > "$run_dir.log"
  elif [[ "$backend" == cuda || "$backend" == kokkos-cuda ]]; then
    "$exe" "$cfg" "$run_dir" > "$run_dir.log"
  else
    OMP_NUM_THREADS="$threads" OMP_PROC_BIND=spread OMP_PLACES=threads "$exe" "$cfg" > "$run_dir.log"
  fi
  end_ns=$(date +%s%N)
  wall=$(awk -v begin="$start_ns" -v end="$end_ns" 'BEGIN {printf "%.6f", (end-begin)/1e9}')
  status_file="$run_dir/run_status.txt"
  transfer=$(value "$status_file" transfer_seconds)
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$backend" "$rep" "$(value "$status_file" status)" "$(value "$status_file" iterations)" "$(value "$status_file" seconds)" "${transfer:-0}" "$wall" "$(value "$status_file" normalized_residual)" "$(value "$status_file" mass_imbalance)" "$(value "$status_file" window_change)" >> "$summary"
}

for backend in serial openmp cuda kokkos-openmp kokkos-cuda; do
  case "$backend" in
    serial|openmp) exe=$cpu_exe ;;
    cuda) exe=$cuda_exe ;;
    kokkos-openmp) exe=$kokkos_omp_exe ;;
    kokkos-cuda) exe=$kokkos_cuda_exe ;;
  esac
  if [[ ! -x "$exe" ]]; then
    printf 'skip %s: executable not found: %s\n' "$backend" "$exe" >&2
    continue
  fi
  for rep in $(seq 1 "$repeats"); do
    printf 'running backend=%s repeat=%s/%s\n' "$backend" "$rep" "$repeats"
    run_one "$backend" "$rep" "$exe"
  done
done
printf 'wrote %s\n' "$summary"
