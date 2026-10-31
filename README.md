# CFL3D nozzle reproduction study

这是一个独立的 C++17/CMake 单块二维无粘喷管有限体积基线。M0 的源码证据和变量/方向映射见 [CFL3D_MAPPING.md](CFL3D_MAPPING.md)；原始 `CFL3D-master` 目录不被修改。

代码存储四变量二维守恒状态 `(rho,rho*u,rho*v,rho*E)`，使用 Roe 类 FDS/Harten 熵修正通量；`rusanov` 仅是调试基线。壁面为滑移反射，入口为总压/总温加特征关系，出口在亚声速时给静压、超声速时外推。`init=quasi1d` 只用于初始化，不替代二维收敛验收。

单元面积按有符号四边形 shoelace 公式计算，单元位置采用面积质心；反向/退化单元、面连接错误和面向量不闭合会直接验收失败。面积审计的原因、影响与修复后回归数据见 [docs/GEOMETRY_FIX_REPORT.md](docs/GEOMETRY_FIX_REPORT.md)。

构建和基础验收：

```bash
cmake -S . -B build -DNOZZLE_ENABLE_OPENMP=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/nozzle_solver --self-test
./build/nozzle_solver --linear-test
```

M2 两个正式工况和三组网格：

```bash
./build/nozzle_solver cases/subsonic.cfg
./build/nozzle_solver cases/choked.cfg
# 可选的第二个参数把结果写入新目录，不覆盖配置中的 output
./build/nozzle_solver cases/choked.cfg output/my_choked_run
./scripts/run_grid_study.sh
python3 scripts/analyze_results.py
```

网格回归也可用 `OUTPUT_ROOT=output/area_fix ./scripts/run_grid_study.sh` 写到独立目录，并用 `NOZZLE_OUTPUT_ROOT=output/area_fix python3 scripts/analyze_results.py` 汇总。

结果写入 `output/<case>_<grid>/`，只有同时满足归一化残差 `<=1e-6`、质量流率不平衡 `<=1e-3` 和最后 200 步状态窗口 `<=1e-4` 才写入 `status=converged`。`output/grid_metrics.csv` 和 `output/residual_grid_comparison.svg` 是网格与残差汇总。

M3 采用 5×5 块接口的 CPU Thomas 线求解器。二维物理更新仍只激活前四个分量，第五个展向动量分量保持为零；这是对 CFL3D 五变量 AF 路径的明确嵌入，而不是把四变量系统误称为完整五变量 CFL3D。显式/隐式同离散对比见 `docs/M2_M3_REPORT.md`。

M4–M5 的真实后端和重复计时：

```bash
REPEATS=5 OMP_NUM_THREADS=8 ./scripts/profile_backends.sh
./scripts/capability_report.sh
```

本环境的原生 CUDA、Kokkos OpenMP/CUDA 后端都已对同一 128×32 壅塞工况完成收敛验证；它们只实现显式一阶 Roe 路径，GPU/Kokkos AF 块线求解仍未实现。具体计时口径、硬件、线规模和复现命令见 `docs/M4_M5_REPORT.md`。


CPU 显式 MUSCL 修复与验证（2026-10-26）：

```bash
cmake -S . -B build-muscl -DCMAKE_BUILD_TYPE=Release \
  -DNOZZLE_ENABLE_OPENMP=OFF -DNOZZLE_ENABLE_CUDA=OFF -DNOZZLE_ENABLE_KOKKOS=OFF
cmake --build build-muscl -j2
ctest --test-dir build-muscl --output-on-failure
python3 scripts/validate_muscl.py small --cfl 2 --output-group small_cfl2
python3 scripts/validate_muscl.py muscl --cfl 2 --output-group muscl_final
python3 scripts/validate_muscl.py first
./build-muscl/muscl_numerics_test --accuracy-study output/muscl_repair/accuracy
python3 scripts/analyze_muscl.py --no-plots
# 有 matplotlib 的 Python 环境可同时生成残差、流场和独立精度图
python3 scripts/analyze_muscl.py
```

单个工况也可以直接运行 `cases/subsonic_muscl.cfg`、`cases/choked_muscl.cfg`。
MUSCL 默认采用显式 SSPRK3；一阶默认仍为 Euler。CFL 是本程序四面波速和的定义，
不能直接与其他软件中不同定义的 CFL 比较。每次运行保存展开后的 `config.cfg`、
`build_info.txt`、完整 `residual.csv`、`run_status.txt`、`field.csv`、截面平均
`sections.csv`、实际守恒面流量 `face_mass.csv` 和局部重构/保护记录 `reconstruction.csv`。

这次修复同时纠正了 Roe 接触波、剪切波的能量特征向量，因此历史一阶/并行结果不能再被当成
新公式的验收结果。CUDA/Kokkos 公共公式同步修正了这两个项，但本轮没有重新进行设备性能验证。
详细根因、证据、精度结论的适用范围及复现记录见 [MUSCL 修复报告](docs/MUSCL_REPAIR_REPORT.md)。

## GitHub 源码包

仓库保留源码、CMake 配置、算例、测试、复现脚本和审计报告。另保留
`output/muscl_repair/legacy/` 中两份冻结源码，供旧问题复现及对照测试使用，
以及验证汇总表和报告引用的四张图。完整迭代历史、流场、二进制、构建目录、
`.deps/`、原版 CFL3D 目录及压缩包不纳入 Git；本地原文件仍然保留。
`.gitignore` 使用顶层白名单，新增顶层目录时需明确决定是否纳入版本管理。

从 GitHub 克隆后可直接按上述 CPU 构建命令运行四项 CTest；完整结果需要重新运行
验证矩阵和精度实验。复现修复前基线及消融实验分别运行：

```bash
python3 scripts/reproduce_muscl_legacy.py baseline
python3 scripts/reproduce_muscl_legacy.py ablation
```

报告中的全量审计包及其 `manifest.sha256` 属于原本地验证记录，不随精简源码包上传。
`scripts/analyze_muscl.py` 需要先生成完整的一阶、MUSCL 和精度实验结果；图表依赖
Python 的 `matplotlib`，CPU 求解器本身不依赖它。

官方参考源码来自 [NASA/CFL3D](https://github.com/nasa/CFL3D)，仅作学习与核对，
不是本项目编译依赖。需要核对官方实现时，可另行获取到被 Git 忽略的
`CFL3D-master/`；当前上游版本可能与原本地归档不同。
