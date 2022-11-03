// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_NARRAY_COLORING_UTILS_HH
#define FLECSI_TOPO_NARRAY_COLORING_UTILS_HH

#include "flecsi/flog.hh"
#include "flecsi/topo/narray/types.hh"
#include "flecsi/topo/types.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/common.hh"

#include <functional>
#include <optional>
#include <vector>

namespace flecsi {
namespace topo {
namespace narray_utils {

inline std::vector<std::size_t>
factor(std::size_t np) {
  std::vector<std::size_t> facs;
  const auto test = [&](std::size_t f) {
    while(!(np % f)) {
      facs.push_back(f);
      np /= f;
    }
  };

  // Trivial wheel factorization:
  test(2);
  test(3);
  for(std::size_t i = 5, step = 2; i * i <= np; i += step, step = 2 * 3 - step)
    test(i);
  if(np > 1) // i.e., the largest prime factor is >3 and unique
    facs.push_back(np);
  std::reverse(facs.begin(), facs.end());

  return facs;
} // factor

/*!
  Create an axial color distribution for the given number of processes.

  @param np total number of colors
  @param indices number of entities per axis

  \return number of colors per axis after decomposition
 */

inline narray_impl::colors
distribute(Color np, narray_impl::gcoord indices) {
  narray_impl::colors parts(indices.size(), 1);

  auto facs = factor(np);

  // greedy decomposition
  for(auto fac : facs) {
    auto maxind = std::distance(
      indices.begin(), std::max_element(indices.begin(), indices.end()));
    parts[maxind] *= fac;
    indices[maxind] /= fac;
  }

  return parts;
} // decomp

/*!
  Create a vector of axis definitions with default settings (hdepth=0,
  bdepth=0, periodic=false, etc) for the given extents and number of colors.

  The method first finds the distribution of colors per axis.
  Then, the end offsets for each color per axis is computed and used to
  initialize each axis's definition object.

  @param num_colors total number of colors
  @param indices number of entities per axis

  \return vector of axis definitions
 */
inline std::vector<narray_impl::axis_definition>
make_axes(Color num_colors, const narray_impl::gcoord & indices) {
  std::vector<narray_impl::axis_definition> axes;
  auto colors = distribute(num_colors, indices);
  flog(warn) << flog::container{colors} << std::endl;
  for(std::size_t d = 0; d < indices.size(); d++) {
    flecsi::util::equal_map em{indices[d], colors[d]};
    axes.push_back({em});
  }
  return axes;
} // make_axes

} // namespace narray_utils
} // namespace topo
} // namespace flecsi

#endif
