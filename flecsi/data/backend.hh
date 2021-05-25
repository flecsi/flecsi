// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_BACKEND_HH
#define FLECSI_DATA_BACKEND_HH

#include <flecsi-config.h>

#include <cstddef>
#include <utility>

#include "flecsi/data/field.hh"
#include "flecsi/topo/core.hh" // single_space

namespace flecsi::data {
/// \addtogroup data
/// \{

// Use {} if unknown:
enum completeness { unknown = 0, complete = 1, incomplete = 2 };
using size2 = std::pair<std::size_t, std::size_t>; // rows, columns
using subrow = std::pair<std::size_t, std::size_t>; // [begin, end)
// The "infinite" size used for resizable regions.
constexpr inline std::size_t logical_size = std::size_t(1) << 32;

// The size types are independent of backend:
struct prefixes_base {
  using row = std::size_t;
  using Field = field<row, single>;
};
/// \}
} // namespace flecsi::data

/*----------------------------------------------------------------------------*
  This section works with the build system to select the correct backend
  implemenation for the data model.
 *----------------------------------------------------------------------------*/

#if FLECSI_BACKEND == FLECSI_BACKEND_legion

#include "flecsi/data/leg/policy.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi

#include "flecsi/data/mpi/policy.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx

#include "flecsi/data/hpx/policy.hh"

#endif // FLECSI_BACKEND

#endif
