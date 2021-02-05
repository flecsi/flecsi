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

#define __FLECSI_PRIVATE__
#include "flecsi/data.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/narray/coloring_utils.hh"
#include "flecsi/topo/narray/interface.hh"

using namespace flecsi;

struct mesh : topo::specialization<topo::narray, mesh> {

  enum index_space { entities };
  using index_spaces = has<entities>;
  enum range { logical, extended, all };
  enum axis { x_axis, y_axis };
  using axes = has<x_axis, y_axis>;

  using coord = topo::narray_impl::coord;
  using coloring_definition = topo::narray_impl::coloring_definition;

  struct meta_data {
    double delta;
  };

  static constexpr std::size_t dimension = 2;

  template<auto>
  static constexpr std::size_t privilege_count = 2;

  /*--------------------------------------------------------------------------*
    Interface
   *--------------------------------------------------------------------------*/

  template<class B>
  struct interface : B {

    template<axis A, range SE = logical>
    std::size_t size() {
      switch(SE) {
        case logical:
          return B::
            template size<index_space::entities, A, B::range::logical>();
          break;
        case extended:
          return B::
            template size<index_space::entities, A, B::range::extended>();
          break;
        case all:
          return B::template size<index_space::entities, A, B::range::all>();
          break;
      }
    }

    template<axis A, range SE = logical>
    auto extents() {
      switch(SE) {
        case logical:
          return B::
            template extents<index_space::entities, A, B::range::logical>();
          break;
        case extended:
          return B::
            template extents<index_space::entities, A, B::range::extended>();
          break;
        case all:
          return B::template extents<index_space::entities, A, B::range::all>();
          break;
      }
    }
  };

  static coloring color(std::vector<coloring_definition> index_definitions) {
    auto [colors, index_colorings] =
      topo::narray_impl::color(index_definitions);

    flog_assert(colors == processes(),
      "current implementation is restricted to 1-to-1 mapping");

    coloring c;
    c.colors = colors;
    for(auto idx : index_colorings) {
      for(auto ic : idx) {
        c.idx_colorings.emplace_back(ic.second);
      }
    }
    return c;
  } // color
};
