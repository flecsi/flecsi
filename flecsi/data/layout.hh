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

#include <cstddef>

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

/*!
  A FleCSI \c layout provides a specific interface for different
  logical data layouts, e.g., dense vs. sparse. The actual data layout
  is implementation-dependent.
 */

enum layout : size_t {
  raw, ///< Uninitialized memory with no objects constructed or destroyed.
  single, ///< Access to the single element of an array.
  dense, ///< Ordinary array of objects.
  ragged, ///< Array of resizable arrays of objects.
  sparse, ///< Array of mappings from integers to objects.
  particle ///< Unordered elements are added/removed up to a maximum number.
};

/// \}
} // namespace data
} // namespace flecsi
