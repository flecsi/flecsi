// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "flecsi/run/local/context.hh"
#include "flecsi/util/mpi.hh"

#include <mpi.h>

namespace flecsi::run::local {

//----------------------------------------------------------------------------//
// Implementation of context_t::initialize (for MPI and HPX).
//----------------------------------------------------------------------------//

int
context::initialize(int argc, char ** argv, bool dependent) {
  using util::mpi::test;

  if(dependent) {
    int provided = 0;
    util::mpi::test(
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));

    if(provided < MPI_THREAD_MULTIPLE) {
      std::cerr << "Your implementation of MPI does not support "
                   "MPI_THREAD_MULTIPLE which is required!"
                << std::endl;
      std::abort();
    }
  } // if

  std::tie(context::process_, context::processes_) = util::mpi::info();

  auto status = context::initialize_generic(argc, argv, dependent);

  if(status != success && dependent) {
    util::mpi::test(MPI_Finalize());
  } // if

#if defined(FLECSI_ENABLE_KOKKOS)
  if(dependent) {
    Kokkos::initialize(argc, argv);
  }
#endif

  return status;
} // initialize

//----------------------------------------------------------------------------//
// Implementation of context_t::finalize (for MPI and HPX).
//----------------------------------------------------------------------------//

void
context::finalize() {

  context::finalize_generic();

#ifndef GASNET_CONDUIT_MPI
  if(context::initialize_dependent_) {
    util::mpi::test(MPI_Finalize());
  } // if
#endif

#if defined(FLECSI_ENABLE_KOKKOS)
  if(context::initialize_dependent_) {
    Kokkos::finalize();
  }
#endif
} // finalize

} // namespace flecsi::run::local
