// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved

#ifndef FLECSI_TOPO_NTREE_GEOMETRY_HH
#define FLECSI_TOPO_NTREE_GEOMETRY_HH

#include "flecsi/util/geometry/point.hh"

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <float.h>
#include <iostream>
#include <math.h>
#include <vector>

namespace flecsi {
namespace topo {

// class ntree_geometry
template<typename T, Dimension D>
struct ntree_geometry {

  using point_t = util::point<T, D>;
  using element_t = T;
  // Tolerance for the computations
  static constexpr element_t tol =
    std::numeric_limits<element_t>::epsilon() * 10.;

  // Return true if point origin lies within the spheroid centered at
  // center with radius.
  static bool
  within(const point_t & origin, const point_t & center, element_t r1) {
    return util::distance(origin, center) - r1 <= tol;
  }

  // Return true if the two spheres identified by origin/r1 and
  // center/r2 are intersecting (a sphere englobe the other's origin)
  static bool within_sphere(const point_t & origin,
    const point_t & center,
    element_t r1,
    element_t r2) {
    return util::distance(origin, center) - std::max(r1, r2) <= tol;
  }

  // Return true if point origin lies within the box specified by
  // min/max point.
  static bool
  within_box(const point_t & min, const point_t & max, const point_t & origin) {
    bool intersect = true;
    for(int i = 0; i < D; ++i) {
      intersect &= (origin[i] <= max[i]) && (origin[i] >= min[i]);
    }
    return intersect;
  }

  // Intersection between two boxes defined by their min and max bound
  static bool intersects_box_box(const point_t & min_b1,
    const point_t & max_b1,
    const point_t & min_b2,
    const point_t & max_b2) {
    bool intersect = true;
    for(int i = 0; i < D; ++i) {
      intersect &= max_b1[i] >= min_b2[i] && max_b2[i] >= min_b1[i];
    }
    return intersect;
  }

  // Intersection of two spheres based on center and radius
  static bool intersects_sphere_sphere(const point_t & c1,
    const element_t r1,
    const point_t & c2,
    const element_t r2) {
    return util::distance(c1, c2) - (r1 + r2) <= tol;
  }

  // Intersection of sphere and box; Compute the closest point from the
  // rectangle to the sphere and this distance less than sphere radius
  static bool intersects_sphere_box(const point_t & min,
    const point_t & max,
    const point_t & c,
    const element_t r) {
    point_t x;
    for(int i = 0; i < D; ++i) {
      x[i] = std::clamp(c[i], min[i], max[i]);
    }
    element_t dist = util::distance(x, c);
    return dist - r <= tol;
  }

  // Multipole method acceptance based on MAC.
  // The angle === l/r < MAC (l source box width, r distance sink -> source)
  // Barnes & Hut 1986
  bool box_MAC(const point_t & position_source,
    const point_t & position_sink,
    const point_t & box_source_min,
    const point_t & box_source_max,
    double macangle) {
    double dmax = util::distance(box_source_min, box_source_max);
    double disttoc = util::distance(position_sink, position_source);
    return dmax / disttoc - macangle <= tol;
  }
}; // class ntree_geometry

} // namespace topo
} // namespace flecsi

#endif
