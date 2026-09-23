// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_BACKEND_HH
#define FLECSI_DATA_BACKEND_HH

#include <cstddef>
#include <utility>
#include <vector>

#include "flecsi/config.hh"
#include "flecsi/data/field.hh"
#include "flecsi/topo/core.hh" // single_space

namespace flecsi::data {
/// \addtogroup data
/// \{

// Use {} if unknown:
enum completeness { complete = 1, incomplete = 2 };
using size2 = std::pair<std::size_t, util::id>; // rows, columns
using subrow = std::pair<util::id, util::id>; // [begin, end)

// The size types are independent of backend:
struct prefixes_base {
  // A helper allowing to detect when resizing can be skipped
  struct size_request {
    size_request() = default;
    size_request(std::size_t sz, bool rsz_req = false)
      : sz(sz), hard(rsz_req) {}
    operator std::size_t() const {
      return sz;
    }
    bool required() const {
      return hard;
    }

  private:
    std::size_t sz;
    bool hard;
  };
  using Field = field<size_request, single>;
};
struct borrow_base {
  using Claim = std::size_t;
  using Claims = std::vector<Claim>;
  static constexpr Claim nil = -1;
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
