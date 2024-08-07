// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "flecsi/run/local/context.hh"
#include "flecsi/util/mpi.hh"

#include <mpi.h>

namespace flecsi::run {

dependencies_guard::dependencies_guard(dependencies_config d)
  : mpi(d.mpi.size(), pointers(d.mpi).data()), kokkos(d.kokkos) {}

local::context::context(const config_base & c)
  : run::context(c, util::mpi::size(), util::mpi::rank()) {}

} // namespace flecsi::run
