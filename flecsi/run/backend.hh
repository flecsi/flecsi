// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_BACKEND_HH
#define FLECSI_RUN_BACKEND_HH

#include <flecsi-config.h>

//----------------------------------------------------------------------------//
// This section works with the build system to select the correct backend
// implementation for the task model.
//----------------------------------------------------------------------------//

#if FLECSI_BACKEND == FLECSI_BACKEND_legion

#include "flecsi/run/leg/context.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_mpi

#include "flecsi/run/mpi/context.hh"

#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx

#include "hpx/context.hh"

#endif // FLECSI_BACKEND

namespace flecsi::run {
// Now that the backend's context_t is complete:
inline std::optional<context_t> context::ctx;
context_t &
context::instance() {
  return *ctx;
} // instance
} // namespace flecsi::run

#endif
