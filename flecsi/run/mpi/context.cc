#include "flecsi/run/mpi/context.hh"
#include "flecsi/data.hh"

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace boost::program_options;

namespace flecsi::run {

dep_base::dependent::dependent(int & argc, char **& argv) : mpi(argc, argv) {
#ifdef FLECSI_ENABLE_KOKKOS
  Kokkos::initialize(argc, argv);
#endif
}
dep_base::dependent::~dependent() {
#ifdef FLECSI_ENABLE_KOKKOS
  Kokkos::finalize();
#endif
}

//----------------------------------------------------------------------------//
// Implementation of context_t::initialize.
//----------------------------------------------------------------------------//

context_t::context_t(int argc, char ** argv, bool d)
  : dep_base{d ? opt(std::in_place, argc, argv) : std::nullopt},
    context(argc, argv, util::mpi::size(), util::mpi::rank()) {
  // This is necessary only because clients can skip flecsi::finalize:
  if(exit_status() != success) {
    dep.reset();
  } // if
} // initialize

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(const std::function<int()> & action) {
  context::start();

  context::threads_per_process_ = 1;
  context::threads_ = context::processes_;

  return detail::data_guard(), task_local_base::guard(), action();
}

} // namespace flecsi::run
