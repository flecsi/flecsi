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
#else
#include <flecsi/utils/const_string.h>
#endif

namespace flecsi {
namespace topology {

constexpr size_t color_index_space = 4097;

/*!
  The color_topology_u type allows users to register data on a
  per-color (similar to an MPI rank) basis. This topology will
  have one instance of each registered field type per color.

  @ingroup topology
 */

struct color_topology_t {
  using type_identifier_t = color_topology_t;
  static constexpr size_t type_identifier_hash =
    flecsi_internal_hash(color_topology_t);

  struct coloring_t {
    coloring_t(size_t size) : size_(size) {}

    size_t size() const { return size_; }

  private:
    size_t size_;
  };

}; // struct color_topology_u

} // namespace topology
} // namespace flecsi
