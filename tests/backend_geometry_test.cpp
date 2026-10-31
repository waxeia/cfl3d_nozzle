#include "explicit_backend_common.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

int main() {
  BackendConfig config;
  config.nx = 16;
  config.ny = 6;
  config.length = 10.0;
  config.throat_half_height = 0.5;
  config.area_ratio = 1.5;

  BackendMesh mesh;
  mesh.build(config);
  if (!mesh.geometryOk()) {
    std::cerr << "backend mesh topology/closure check failed\n";
    return 1;
  }

  const double dx = config.length / config.nx;
  const auto half_height = [&](double x) {
    const double z = (x - 0.5 * config.length) / (0.5 * config.length);
    return config.throat_half_height *
           (1.0 + (config.area_ratio - 1.0) * z * z);
  };
  double measured_total = 0.0;
  double expected_total = 0.0;
  for (int j = 0; j < config.ny; ++j) {
    for (int i = 0; i < config.nx; ++i) {
      const double expected =
          dx * (half_height(i * dx) + half_height((i + 1) * dx)) /
          config.ny;
      const double measured = mesh.cells[mesh.id(i, j)].area;
      const double scale = std::max({1.0, std::abs(expected), std::abs(measured)});
      if (std::abs(measured - expected) > 1.0e-12 * scale) {
        std::cerr << "backend cell area mismatch at (" << i << ',' << j
                  << "): " << measured << " vs " << expected << '\n';
        return 1;
      }
      measured_total += measured;
      expected_total += expected;
    }
  }
  if (std::abs(measured_total - expected_total) >
      1.0e-12 * std::max(1.0, std::abs(expected_total))) {
    std::cerr << "backend total area mismatch\n";
    return 1;
  }

  config.init = "uniform";
  config.init_mach = 0.25;
  const auto state = backendInitialize(config, mesh);
  const DPrim expected_state{config.p0 / (config.gas_R * config.T0),
                             config.init_mach * std::sqrt(config.gamma * config.gas_R * config.T0),
                             0.0, config.p0};
  for (std::size_t n = 0; n < mesh.cells.size(); ++n) {
    const DPrim actual = dPrimitive(&state[4 * n], config.gamma);
    if (std::abs(actual.rho - expected_state.rho) > 1.0e-12 ||
        std::abs(actual.u - expected_state.u) > 1.0e-12 ||
        std::abs(actual.v - expected_state.v) > 1.0e-12 ||
        std::abs(actual.p - expected_state.p) > 1.0e-12) {
      std::cerr << "backend uniform initialization mismatch\n";
      return 1;
    }
  }

  std::cout << "backend geometry and uniform initialization PASS\n";
  return 0;
}
