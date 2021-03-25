/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*! @file */

#if !defined(__FLECSI_PRIVATE__)
#error Do not include this file directly!
#endif

#include "flecsi/data/copy.hh"
#include "flecsi/execution.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/util/color_map.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/serialize.hh"

#include <algorithm>
#include <cstddef>
#include <map>
#include <set>
#include <vector>

namespace flecsi {
namespace topo {
namespace narray_impl {

enum masks : uint32_t { interior = 0b00, low = 0b01, high = 0b10 };

enum axes : Dimension { x_axis, y_axis, z_axis };

using coord = std::vector<std::size_t>;
using hypercube = std::array<coord, 2>;
using interval = std::pair<std::size_t, std::size_t>;
using colors = std::vector<Color>;

/*
  Input type for color method.
 */

struct coloring_definition {
  colors axis_colors;
  coord axis_extents;
  coord axis_hdepths;
  coord axis_bdepths;
  std::vector<bool> axis_periodic;
  bool diagonals = false;
  bool create_plan = true;
};

struct index_coloring {
  /*
    Store the axis orientations of this color.
   */

  std::uint32_t faces;

  /*
    The global extents.
   */

  coord global;

  /*
    The local extents of this color. This is the full size including
    boundary depth, and ghosts. The "extents" coordinate implicitly
    defines a hypercube from {0, 0, ..., 0} to extents{...}.
   */

  coord extents;

  /*
    The global coordinate offset of the local hypercube.
    Local to global id translation can be computed with this.
   */

  coord offset;

  /*
    The logical entities, i.e., the entities for this color without
    boundary padding or ghosts.
   */

  hypercube logical;

  /*
    The extended entities, i.e., the logical entities including boundary
    padding. The boundary depth can be computed like:

      boundary_depth_low[axis] = logical[0][axis] - extended[0][axis];
      boundary_depth_high[axis] = extended[1][axis] - logical[1][axis];

    The ghost depth can be computed like:
      shared_depth_low[axis] = logical[0][axis];
      shared_depth_high[axis] = extents[axis] - logical[1][axis];

    (note: We use logical to compute the ghost depth because an edge
      cannot have both boundary padding, and ghosts.)
   */

  hypercube extended;

  /*
    Offsets on the remote color.
   */

  std::map<Color,
    std::vector<std::pair</* local ghost offset, remote shared offset */
      std::size_t,
      std::size_t>>>
    points;

  /*
    Local ghost intervals.
   */

  std::vector<std::pair<std::size_t, std::size_t>> intervals;
};

} // namespace narray_impl

/*----------------------------------------------------------------------------*
  Base.
 *----------------------------------------------------------------------------*/

struct narray_base {
  using index_coloring = narray_impl::index_coloring;
  using coord = narray_impl::coord;
  using hypercube = narray_impl::hypercube;
  using colors = narray_impl::colors;
  using coloring_definition = narray_impl::coloring_definition;

  struct coloring {
    MPI_Comm comm;
    Color colors;
    std::vector<index_coloring> idx_colorings;
  }; // struct coloring

  static std::size_t idx_size(index_coloring const & ic, std::size_t) {
    std::size_t allocation{1};
    for(auto e : ic.extents) {
      allocation *= e;
    }
    return allocation;
  }

  static void idx_itvls(index_coloring const & ic,
    std::vector<std::size_t> & num_intervals,
    MPI_Comm const & comm) {
    num_intervals = util::mpi::all_gather(
      [&ic](int, int) { return ic.intervals.size(); }, comm);
  }

  static void set_dests(field<data::intervals::Value>::accessor<wo> a,
    std::vector<std::pair<std::size_t, std::size_t>> const & intervals,
    MPI_Comm const &) {
    flog_assert(a.span().size() == intervals.size(), "interval size mismatch");
    std::size_t i{0};
    for(auto it : intervals) {
      a[i++] = data::intervals::make({it.first, it.second}, process());
    } // for
  }

  template<PrivilegeCount N>
  static void set_ptrs(
    field<data::points::Value>::accessor1<privilege_repeat<wo, N>> a,
    std::map<Color, std::vector<std::pair<std::size_t, std::size_t>>> const &
      shared_ptrs,
    MPI_Comm const &) {
    for(auto const & si : shared_ptrs) {
      for(auto p : si.second) {
        // si.first: owner
        // p.first: local ghost offset
        // p.second: remote shared offset
        a[p.first] = data::points::make(si.first, p.second);
      } // for
    } // for
  }
}; // struct narray_base

} // namespace topo

/*----------------------------------------------------------------------------*
  Serialization Rules
 *----------------------------------------------------------------------------*/

template<>
struct util::serial<topo::narray_impl::index_coloring> {
  using type = topo::narray_impl::index_coloring;
  template<class P>
  static void put(P & p, const type & s) {
    serial_put(p,
      std::tie(s.faces,
        s.global,
        s.extents,
        s.offset,
        s.logical,
        s.extended,
        s.points,
        s.intervals));
  }
  static type get(const std::byte *& p) {
    const serial_cast r{p};
    return type{r, r, r, r, r, r, r, r};
  }
};

} // namespace flecsi
