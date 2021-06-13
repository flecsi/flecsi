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
#if defined(FLECSI_ENABLE_MPI)
#include "flecsi/util/mpi.hh"
#endif

#include <cstring>
#include <string>
#include <vector>

namespace flecsi::run {

#if defined(FLECSI_ENABLE_MPI)
namespace detail {

bool
detect_mpi_environment() {
#if defined(__bgq__)
  // If running on BG/Q, we can safely assume to always run in an
  // MPI environment
  return true;
#else
  static char const* const envvars[] = {"MV2_COMM_WORLD_RANK",
    "PMI_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "ALPS_APP_PE",
    "PMIX_RANK"};

  for(auto const & envvar : envvars) {
    char * env = std::getenv(envvar);
    if(env)
      return true;
  }
  return false;
#endif
}
} // namespace detail
#endif

//----------------------------------------------------------------------------//
// Implementation of context_t::initialize.
//----------------------------------------------------------------------------//

int
context_t::initialize(int argc, char ** argv, bool dependent) {

#if defined(FLECSI_ENABLE_MPI)
  bool has_mpi = detail::detect_mpi_environment();
  if(dependent && has_mpi) {
    util::mpi::test(MPI_Init(&argc, &argv));
  }
#endif

  auto status = context::initialize_generic(argc, argv, dependent);

#if defined(FLECSI_ENABLE_MPI)
  if(status != success && dependent && has_mpi) {
    util::mpi::test(MPI_Finalize());
  } // if
#endif

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

#if defined(FLECSI_ENABLE_MPI)
  if(context::initialize_dependent_ && detail::detect_mpi_environment()) {
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

namespace detail {

template<typename Comm>
struct free_on_exit {
  free_on_exit(Comm & comm, Comm value) : comm_(comm) {
    comm_ = value;
  }
  ~free_on_exit() {
    comm_.free();
  }
  Comm & comm_;
};
} // namespace detail

int
context_t::start(std::function<int()> const & action) {

  ::hpx::init_params params;
  params.cfg = {// allocate at least two cores
    "hpx.force_min_os_threads!=2",
    // make sure hpx_main is always executed
    "hpx.run_hpx_main!=1",
    // allow for unknown command line options
    "hpx.commandline.allow_unknown!=1",
    // disable HPX' short options
    "hpx.commandline.aliasing!=0"};

  return ::hpx::init(
    [=](int, char *[]) -> int {
      context::start();

      context::process_ = ::hpx::get_locality_id();
      context::processes_ = ::hpx::get_num_localities(::hpx::launch::sync);
      context::threads_per_process_ = ::hpx::get_num_worker_threads();
      context::threads_ = context::processes_;

      // initialize communicators
      detail::free_on_exit on_exit1(world_comm_,
        ::hpx::collectives::create_communicator("/flecsi/world_comm",
          ::hpx::collectives::num_sites_arg(context::processes_),
          ::hpx::collectives::this_site_arg(context::process_)));

      detail::free_on_exit on_exit2(world_channel_comm_,
        ::hpx::collectives::create_channel_communicator(::hpx::launch::sync,
          "/flecsi/world_channel_comm",
          ::hpx::collectives::num_sites_arg(context::processes_),
          ::hpx::collectives::this_site_arg(context::process_)));

      // guard destroyed after action call
      int result = (flecsi::detail::data_guard(), action());

      // tell the runtime it's ok to exit
      ::hpx::finalize();
      return result;
    },
    argv_.size(),
    argv_.data(),
    params);
}

} // namespace flecsi::run
