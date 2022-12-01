#include "flecsi/run/mpi/context.hh"
#include "flecsi/data.hh"

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace boost::program_options;

namespace flecsi::run {

context_t::dependent::dependent(int & argc, char **& argv) : mpi(argc, argv) {
#ifdef FLECSI_ENABLE_KOKKOS
  Kokkos::initialize(argc, argv);
#endif
}
context_t::dependent::~dependent() {
#ifdef FLECSI_ENABLE_KOKKOS
  Kokkos::finalize();
#endif
}

//----------------------------------------------------------------------------//
// Implementation of context_t::initialize.
//----------------------------------------------------------------------------//

int
context_t::initialize(int argc, char ** argv, bool dependent) {
  using util::mpi::test;

  if(dependent) {
    dep.emplace(argc, argv);
  } // if

  std::tie(context::process_, context::processes_) = util::mpi::info();

  auto status = context::initialize_generic(argc, argv);

  if(status != success) {
    dep.reset();
  } // if

  return status;
} // initialize

//----------------------------------------------------------------------------//
// Implementation of context_t::finalize.
//----------------------------------------------------------------------------//

void
context_t::finalize() {

  context::finalize_generic();
  dep.reset();
} // finalize

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(const std::function<int()> & action) {
  context::start();

  context::threads_per_process_ = 1;
  context::threads_ = context::processes_;

  return detail::data_guard(), action(); // guard destroyed after action call
}

} // namespace flecsi::run
