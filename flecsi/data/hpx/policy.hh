// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_HPX_POLICY_HH
#define FLECSI_DATA_HPX_POLICY_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>

#include "flecsi/config.hh"
#include "flecsi/data/local/storage.hh"
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

struct fate {
  using future = ::hpx::shared_future<void>;

  fate() = default;
  fate(future f) noexcept : f(std::move(f)) {}
  fate(const fate &) = default; // note that destroying either will wait
  fate(fate &&) = default;
  ~fate() {
    if(*this) {
      f.wait();
      if(f.has_exception())
        flog_fatal("destroying exceptional future:\n" +
                   ::hpx::diagnostic_information(f.get_exception_ptr()));
    }
  }
  fate & operator=(const fate & r) & noexcept {
    if(&r != this)
      *this = fate(r);
    return *this;
  }
  fate & operator=(fate && r) & noexcept {
    fate w(std::move(r));
    std::swap(f, w.f);
    return *this;
  }

  explicit operator bool() const {
    return f.valid() && !f.is_ready();
  }
  const future & get() const {
    return f;
  }
  future release() {
    return std::move(f);
  }

private:
  future f;
};

struct backend_storage : local::detail::storage<> {
  // Synchronize with all pending operations on this storage.
  void synchronize() {
    future = {};
  }

  // In practice, this never waits, because tasks keep the region_impl alive
  // and every ghost copy is followed by a task using the same regions.
  fate future;
  dependency dep = dependency::none;
};
} // namespace data
} // namespace flecsi

#include "flecsi/data/local/policy.hh"

#endif // FLECSI_DATA_HPX_POLICY_HH
