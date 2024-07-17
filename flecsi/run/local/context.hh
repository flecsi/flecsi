// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_LOCAL_CONTEXT_HH
#define FLECSI_RUN_LOCAL_CONTEXT_HH

#include "flecsi/run/context.hh"
#include "flecsi/util/mpi.hh"

namespace flecsi::run {

// Common initialization and finalize operations for MPI and HPX

struct dependencies_guard {
  dependencies_guard(dependencies_config = {});

private:
  util::mpi::init mpi;
#ifdef FLECSI_ENABLE_KOKKOS
  Kokkos::ScopeGuard kokkos;
#endif
};

namespace local {

struct context : run::context {
  context(const config_base &);
};

} // namespace local
} // namespace flecsi::run

#endif
