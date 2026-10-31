#pragma once

#include <array>
#include <cmath>

namespace nozzle_geometry {

using Point2 = std::array<double, 2>;

struct QuadGeometry {
  double signed_area = 0.0;
  double centroid_x = 0.0;
  double centroid_y = 0.0;
};

// The vertices must follow the polygon boundary.  A counter-clockwise polygon
// has positive area; retaining the sign lets callers detect an inverted cell.
inline QuadGeometry quadrilateral(const Point2 &p0, const Point2 &p1,
                                  const Point2 &p2, const Point2 &p3) {
  const std::array<Point2, 4> p{p0, p1, p2, p3};
  double twice_area = 0.0;
  double centroid_x_sum = 0.0;
  double centroid_y_sum = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const Point2 &a = p[i];
    const Point2 &b = p[(i + 1) % p.size()];
    const double cross = a[0] * b[1] - b[0] * a[1];
    twice_area += cross;
    centroid_x_sum += (a[0] + b[0]) * cross;
    centroid_y_sum += (a[1] + b[1]) * cross;
  }

  QuadGeometry result;
  result.signed_area = 0.5 * twice_area;
  if (twice_area != 0.0) {
    result.centroid_x = centroid_x_sum / (3.0 * twice_area);
    result.centroid_y = centroid_y_sum / (3.0 * twice_area);
  }
  return result;
}

}  // namespace nozzle_geometry
