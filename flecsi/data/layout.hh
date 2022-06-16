// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LAYOUT_HH
#define FLECSI_DATA_LAYOUT_HH

#include <cstddef>

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

/*!
  A type of logical data structure.
  The interface for each is provided by an \c accessor.
 */

enum layout : size_t {
  raw, ///< Uninitialized memory with no objects constructed or destroyed.
  single, ///< Access to the single element of an array.
  dense, ///< Ordinary array of objects.
  ragged, ///< Array of resizable arrays of objects.
  sparse ///< Array of mappings from integers to objects.
};

/// \}
} // namespace data
} // namespace flecsi

#endif
