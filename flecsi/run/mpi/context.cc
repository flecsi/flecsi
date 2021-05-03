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

#include "flecsi/run/mpi/context.hh"
#include "flecsi/data.hh"

using namespace boost::program_options;

namespace flecsi::run {

//----------------------------------------------------------------------------//
// Implementation of context_t::initialize.
//----------------------------------------------------------------------------//

int
context_t::initialize(int argc, char ** argv, bool dependent) {
  using util::mpi::test;

  if(dependent) {
    test(MPI_Init(&argc, &argv));
  } // if

  std::tie(context::process_, context::processes_) = util::mpi::info();

  auto status = context::initialize_generic(argc, argv, dependent);

  if(status != success && dependent) {
    test(MPI_Finalize());
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
context_t::start(const std::function<int()> & action) {
  context::start();

  context::threads_per_process_ = 1;
  context::threads_ = context::processes_;

  std::vector<char *> largv;
  largv.push_back(argv_[0]);

  for(auto opt = unrecognized_options_.begin();
      opt != unrecognized_options_.end();
      ++opt) {
    largv.push_back(opt->data());
  } // for

  return detail::data_guard(), action(); // guard destroyed after action call
}

} // namespace flecsi::run
