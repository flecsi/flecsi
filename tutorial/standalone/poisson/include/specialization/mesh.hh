/*
   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/topo/narray/coloring_utils.hh>
#include <flecsi/topo/narray/interface.hh>

#include "types.hh"

namespace poisson {

struct mesh : flecsi::topo::specialization<flecsi::topo::narray, mesh> {

  /*--------------------------------------------------------------------------*
    Policy Information.
   *--------------------------------------------------------------------------*/

  enum index_space { vertices };
  using index_spaces = has<vertices>;
  enum range { interior, logical, all, global };
  enum axis { x_axis, y_axis };
  using axes = has<x_axis, y_axis>;
  enum boundary { low, high };

  using coord = base::coord;
  using colors = base::colors;
  using hypercube = base::hypercube;
  using coloring_definition = base::coloring_definition;

  struct meta_data {
    double xdelta;
    double ydelta;
  };

  static constexpr std::size_t dimension = 2;

  template<auto>
  static constexpr std::size_t privilege_count = 2;

  /*--------------------------------------------------------------------------*
    Interface.
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    template<axis A, range SE = interior>
    std::size_t size() {
      if constexpr(SE == interior) {
        const bool low = B::template is_low<index_space::vertices, A>();
        const bool high = B::template is_high<index_space::vertices, A>();

        if(low && high) { /* degenerate */
          return size<A, logical>() - 2;
        }
        else if(low || high) {
          return size<A, logical>() - 1;
        }
        else { /* interior */
          return size<A, logical>();
        }
      }
      else if constexpr(SE == logical) {
        return B::template size<index_space::vertices, A, B::range::logical>();
      }
      else if(SE == all) {
        return B::template size<index_space::vertices, A, B::range::all>();
      }
      else if(SE == global) {
        return B::template size<index_space::vertices, A, B::range::global>();
      }
    }

    template<axis A, range SE = interior>
    auto vertices() {
      if constexpr(SE == interior) {
        const bool low = B::template is_low<index_space::vertices, A>();
        const bool high = B::template is_high<index_space::vertices, A>();
        const std::size_t start =
          B::template logical<index_space::vertices, 0, A>();
        const std::size_t end =
          B::template logical<index_space::vertices, 1, A>();

        return flecsi::topo::make_ids<index_space::vertices>(
          flecsi::util::iota_view<flecsi::util::id>(start + low, end - high));
      }
      else if constexpr(SE == logical) {
        return B::
          template extents<index_space::vertices, A, B::range::logical>();
      }
      else if(SE == all) {
        return B::template extents<index_space::vertices, A, B::range::all>();
      }
    }

    double xdelta() {
      return (*(this->policy_meta_)).xdelta;
    }

    double ydelta() {
      return (*(this->policy_meta_)).ydelta;
    }

    double dxdy() {
      return xdelta() * ydelta();
    }

    template<axis A>
    double value(std::size_t i) {
      return (A == x_axis ? xdelta() : ydelta()) *
             (B::template offset<index_space::vertices, A, B::range::global>() +
               i);
    }

    template<axis A, boundary BD>
    bool is_boundary(std::size_t i) {

      auto const loff =
        B::template offset<index_space::vertices, A, B::range::logical>();
      auto const lsize =
        B::template size<index_space::vertices, A, B::range::logical>();
      const bool l = B::template is_low<index_space::vertices, A>();
      const bool h = B::template is_high<index_space::vertices, A>();

      if(l && h) { /* degenerate */
        if constexpr(BD == boundary::low) {
          return i == loff;
        }
        else {
          return i == (lsize + loff - 1);
        }
      }
      else if(l) {
        if constexpr(BD == boundary::low) {
          return i == loff;
        }
        else {
          return false;
        }
      }
      else if(h) {
        if constexpr(BD == boundary::low) {
          return false;
        }
        else {
          return i == (lsize + loff - 1);
        }
      }
      else { /* interior */
        return false;
      }
    } // is_boundary
  }; // struct interface

  static auto distribute(std::size_t np, std::vector<std::size_t> indices) {
    return flecsi::topo::narray_utils::distribute(np, indices);
  } // distribute

  /*--------------------------------------------------------------------------*
    Color Method.
   *--------------------------------------------------------------------------*/

  static coloring color(colors axis_colors, coord axis_extents) {
    coord hdepths{1, 1};
    coord bdepths{0, 0};
    std::vector<bool> periodic{false, false};
    coloring_definition cd{
      axis_colors, axis_extents, hdepths, bdepths, periodic};

    auto [nc, ne, pcs, partitions] =
      flecsi::topo::narray_utils::color(cd, MPI_COMM_WORLD);

    flog_assert(nc == flecsi::processes(),
      "current implementation is restricted to 1-to-1 mapping");

    coloring c;
    c.comm = MPI_COMM_WORLD;
    c.colors = nc;
    c.idx_colorings.emplace_back(std::move(pcs));
    c.partitions.emplace_back(std::move(partitions));
    return c;
  } // color

  /*--------------------------------------------------------------------------*
    Initialization.
   *--------------------------------------------------------------------------*/

  using grect = std::array<std::array<double, 2>, 2>;

  static void set_geometry(mesh::accessor<flecsi::rw> sm, grect const & g) {
    meta_data & md = sm.policy_meta_;
    double xdelta =
      std::abs(g[0][1] - g[0][0]) / (sm.size<x_axis, global>() - 1);
    double ydelta =
      std::abs(g[1][1] - g[1][0]) / (sm.size<y_axis, global>() - 1);

    md.xdelta = xdelta;
    md.ydelta = ydelta;
  }

  static void initialize(flecsi::data::topology_slot<mesh> & s,
    coloring const &,
    grect const & geometry) {
    flecsi::execute<set_geometry, flecsi::mpi>(s, geometry);
  } // initialize

}; // struct mesh

} // namespace poisson
