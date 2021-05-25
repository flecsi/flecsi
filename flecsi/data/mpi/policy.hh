// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_MPI_POLICY_HH
#define FLECSI_DATA_MPI_POLICY_HH

#include "flecsi/data/local/storage.hh"

namespace flecsi {
namespace data {
// Direct data storage.
struct backend_storage : local::detail::storage
{
  constexpr void synchronize() const noexcept {
    // this is a noop for the MPI backend
  }
};
} // namespace data
} // namespace flecsi

#include "flecsi/data/local/policy.hh"

#endif
