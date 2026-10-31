# M4–M5 并行与交付报告

> 2026-10-26 更新：本文数据属于 MUSCL 修复前的历史记录。Roe 能量特征向量错误及 MUSCL 当前验收见 [MUSCL_REPAIR_REPORT.md](MUSCL_REPAIR_REPORT.md)。旧并行计时未在新公式下重新认证。
> 2026-10-14 已在单元面积和质心修复后重新构建并完成下方全部 128×32 重复计时。五个后端均使用当前源码；基础 CTest 为 3/3 通过。原始 25 次记录保存在 `output/backend_profile_128x32.csv`。

## CPU 对照和 profiling

主程序保留标准库 CPU 路径，并提供真正的 OpenMP 面通量/单元归约路径；CMake 使用 `-fopenmp`。本报告采用 `cases/choked_perf_128x32.cfg`（128×32 单元、一阶 Roe、CFL=0.6、相同 1e-6/1e-3/200 步判据），每个后端独立完整运行 5 次。`seconds` 是求解迭代段，`wall_seconds` 是脚本从启动到进程退出的端到端墙钟时间，后者包含初始化、设备分配/拷贝和最终场写出。

|后端|执行位置|平均 iterations|平均求解秒|平均墙钟秒|墙钟相对串行|样本标准差(墙钟)|
|---|---|---:|---:|---:|---:|---:|
|Serial CPU|CPU|6883|2.9110|2.9314|1.00×|0.0158|
|OpenMP CPU (8 threads)|CPU|6883|1.5097|1.5313|1.91×|0.0462|
|原生 CUDA|RTX 3060 Laptop|6883|1.2255|1.4742|1.99×|0.1128|
|Kokkos OpenMP (8 threads)|CPU|6883|1.1704|1.2009|2.44×|0.0237|
|Kokkos CUDA|RTX 3060 Laptop|6883|1.3012|1.5492|1.89×|0.1194|

每个后端的五次 `run_status.txt` 均为 `converged`，最终归一化残差约 `9.99904e-7`、质量不平衡约 `5.01014e-6`、窗口变化约 `1.59019e-6`，迭代数完全相同。Serial 与 OpenMP 的最终场逐字节相同；CUDA、Kokkos OpenMP 和 Kokkos CUDA 因浮点求值路径产生不同文件哈希，但相对 Serial 的任一场变量最大绝对差异不超过 `3.6e-15`，数值上一致。原生 CUDA 的平均 `transfer_seconds` 为 0.1503 s，Kokkos CUDA 记录为 0.000167 s；该字段只覆盖各后端内部显式计时的传输区间，性能结论采用包含初始化、运行和写出的 `wall_seconds`。

## CUDA/Kokkos 能力边界

本机硬件为 NVIDIA RTX 3060 Laptop GPU（6 GiB，驱动 580.178.04）、CUDA 12.6、GNU C++ 13.3、20 个逻辑 CPU。原生 CUDA 后端把面通量写入独立设备数组，再执行单元残差、局部步长、正性试探和诊断归约；Kokkos 后端使用同一数据布局和显式公式，在 OpenMP 与 CUDA 默认执行空间分别验证。两者目前只支持显式、一阶、Roe 路径；没有实现 GPU/Kokkos AF 块线求解，不能把设备显式结果称为隐式加速。

128×32 网格的真实方向线规模是：`xi=x` 方向 32 条线、每条 128 个单元；`eta=y` 方向 128 条线、每条 32 个单元。M3 CPU AF 仅用于对照，未移植到 CUDA/Kokkos。

## 复现

```bash
cmake -S . -B build -DNOZZLE_ENABLE_OPENMP=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/nozzle_solver cases/subsonic.cfg
./build/nozzle_solver cases/choked.cfg
./scripts/run_grid_study.sh
python3 scripts/analyze_results.py
./build/nozzle_solver cases/choked_explicit_m3.cfg
./build/nozzle_solver cases/choked_implicit_m3.cfg
REPEATS=5 OMP_NUM_THREADS=8 ./scripts/profile_backends.sh
./scripts/profile_cpu.sh
./scripts/capability_report.sh
```

无 root 的 Kokkos OpenMP 环境可运行 `scripts/bootstrap_kokkos_ubuntu.sh`；Kokkos CUDA 可运行 `scripts/bootstrap_kokkos_cuda.sh`，脚本固定校验 Ubuntu Kokkos 3.7.01 源码、设置 `sm_86`/`AMPERE86` 并安装到项目 `.deps/`。`profile_backends.sh` 默认使用 `build-all2/` 和 `build-kokkos-cuda3/`，也可用 `CPU_EXE`、`CUDA_EXE`、`KOKKOS_OMP_EXE`、`KOKKOS_CUDA_EXE` 覆盖。所有正式输出、状态和残差记录均保留在 `output/`。
