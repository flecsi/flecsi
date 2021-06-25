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

#include "flecsi/data.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/narray/coloring_utils.hh"
#include "flecsi/topo/narray/interface.hh"

using namespace flecsi;

struct mesh_helper : topo::specialization<topo::narray, mesh_helper> 
{}; 

template<std::size_t D>
struct axes_helper{ };

template<>
struct axes_helper<1> {
  enum axis {x_axis}; 
  using axes = typename mesh_helper::template has<x_axis>;
};

template<>
struct axes_helper<2> {
  enum axis {x_axis, y_axis}; 
  using axes = typename mesh_helper::template has<x_axis, y_axis>;
};

template<>
struct axes_helper<3> {
  enum axis {x_axis, y_axis, z_axis}; 
  using axes = typename mesh_helper::template has<x_axis, y_axis, z_axis>;
};

template<std::size_t D>
struct mesh : topo::specialization<topo::narray, mesh<D>> {
  static_assert((D >= 1 && D <= 6), "Invalid dimension for testing !");

  enum index_space { entities };
  enum range {
    logical,
    extended,
    all,
    boundary_low,
    boundary_high,
    ghost_low,
    ghost_high,
    global
  };
  //enum axis { x_axis, y_axis, z_axis };
  
  using axis = typename axes_helper<D>::axis; 
  using axes = typename axes_helper<D>::axes; 

  struct meta_data {
    double delta;
  };

  static constexpr std::size_t dimension = D;
 
  template<auto>
  static constexpr std::size_t privilege_count = 2; 

  using index_spaces = typename mesh::template has<entities>;
  using coord = typename mesh::base::coord;
  using coloring_definition = typename mesh::base::coloring_definition;
  using coloring = typename mesh::base::coloring;

  static coloring color(std::vector<coloring_definition> index_definitions) {
    auto [colors, index_colorings] =
      topo::narray_utils::color(index_definitions, MPI_COMM_WORLD);

    flog_assert(colors == processes(),
      "current implementation is restricted to 1-to-1 mapping");

    coloring c;
    c.comm = MPI_COMM_WORLD;
    c.colors = colors;
    for(auto idx : index_colorings) {
      for(auto ic : idx) {
        c.idx_colorings.emplace_back(ic.second);
      }
    }
    return c;
  } // color

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
        case boundary_low:
          return B::
            template size<index_space::entities, A, B::range::boundary_low>();
          break;
        case boundary_high:
          return B::
            template size<index_space::entities, A, B::range::boundary_high>();
          break;
        case ghost_low:
          return B::
            template size<index_space::entities, A, B::range::ghost_low>();
          break;
        case ghost_high:
          return B::
            template size<index_space::entities, A, B::range::ghost_high>();
          break;
        case global:
          return B::template size<index_space::entities, A, B::range::global>();
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
        case boundary_low:
          return B::template extents<index_space::entities,
            A,
            B::range::boundary_low>();
          break;
        case boundary_high:
          return B::template extents<index_space::entities,
            A,
            B::range::boundary_high>();
          break;
        case ghost_low:
          return B::
            template extents<index_space::entities, A, B::range::ghost_low>();
          break;
        case ghost_high:
          return B::
            template extents<index_space::entities, A, B::range::ghost_high>();
          break;
      }
    }

    template<axis A, range SE = logical>
    auto offset() {
      switch(SE) {
        case logical:
          return B::
            template offset<index_space::entities, A, B::range::logical>();
          break;
        case extended:
          return B::
            template offset<index_space::entities, A, B::range::extended>();
          break;
        case all:
          return B::template offset<index_space::entities, A, B::range::all>();
          break;
        case boundary_low:
          return B::
            template offset<index_space::entities, A, B::range::boundary_low>();
          break;
        case boundary_high:
          return B::template offset<index_space::entities,
            A,
            B::range::boundary_high>();
          break;
        case ghost_low:
          return B::
            template offset<index_space::entities, A, B::range::ghost_low>();
          break;
        case ghost_high:
          return B::
            template offset<index_space::entities, A, B::range::ghost_high>();
          break;
        case global:
          return B::
            template offset<index_space::entities, A, B::range::global>();
          break;
      }
    }
  };
}; // mesh

