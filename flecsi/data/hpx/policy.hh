// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_HPX_POLICY_HH
#define FLECSI_DATA_HPX_POLICY_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>

#include "flecsi/config.hh"
#include "flecsi/data/local/storage.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/run/hpx/context.hh"
#include "flecsi/util/types.hh"

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace flecsi {
namespace data {

/// \defgroup hpx-data HPX Data
/// HPX-specific data management.
/// \ingroup data
/// \{

/// Type of the dependency represented by a future
enum class dependency : std::uint8_t {
  /// This field has no future associated with it (yet)
  none = 0,
  /// The future represents a read operation
  read = 1,
  /// The future represents a write operation
  write = 2
};
/// \}

struct backend_storage : local::detail::storage {

  ~backend_storage() {
    // whenever region is destroyed we have to wait for all pending tasks to
    // complete
    if(future.valid() && !future.is_ready()) {
      future.wait();
      if(future.has_exception()) {
        // there is no way to report the error to the user's code at this point,
        // thus termination is the only option
        flog_fatal(
          "future is in exceptional state during destruction of region:\n" +
          ::hpx::diagnostic_information(future.get_exception_ptr()));
      }
    }
  }

  // Synchronize with all pending operations on this storage.
  void synchronize() {
    if(future.valid() && !future.is_ready()) {
      future.get();
    }
  }

  ::hpx::shared_future<void> future;
  dependency dep = dependency::none;
};
} // namespace data
} // namespace flecsi

#include "flecsi/data/local/policy.hh"

#endif // FLECSI_DATA_HPX_POLICY_HH
