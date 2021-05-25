/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */

#include <hpx/hpx_init.hpp>

#include "flecsi/data.hh"
#include "flecsi/run/hpx/context.hh"
#include "flecsi/util/mpi.hh"

namespace flecsi::run {

//----------------------------------------------------------------------------//
// Implementation of context_t::initialize.
//----------------------------------------------------------------------------//

int
context_t::initialize(int argc, char ** argv, bool dependent) {

  if(dependent) {
    util::mpi::test(MPI_Init(&argc, &argv));
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
// Implementation of context_t::finalize.
//----------------------------------------------------------------------------//

void
context_t::finalize() {

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

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(std::function<int()> const & action) {
  context::start();

  context::threads_per_process_ = hpx::get_num_worker_threads();
  context::threads_ = context::processes_;

  hpx::init_params params;
  params.cfg = {// allocate at least two cores
    "hpx.force_min_os_threads!=2",
    // make sure hpx_main is always executed
    "hpx.run_hpx_main!=1",
    // allow for unknown command line options
    "hpx.commandline.allow_unknown!=1",
    // disable HPX' short options
    "hpx.commandline.aliasing!=0"};

  return hpx::init(
    [=](int, char*[]) -> int {
      // guard destroyed after action call
      int result = 0;
      {
        auto g = detail::data_guard();
        result = action();
      }
      // tell the runtime it's ok to exit
      hpx::finalize();
      return result;
    },
    argv_.size(),
    argv_.data(),
    params);
}

} // namespace flecsi::run
