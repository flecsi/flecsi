#include "flecsi/run/mpi/context.hh"
#include "flecsi/data.hh"

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace boost::program_options;

namespace flecsi::run {

dependencies_guard::dependencies_guard(arguments::dependent & d)
  : dependencies_guard(d, d.mpi.size(), arguments::pointers(d.mpi).data()) {}
dependencies_guard::dependencies_guard(arguments::dependent & d,
  int mc,
  char ** mv)
  : mpi(mc, mv) {
#ifdef FLECSI_ENABLE_KOKKOS
  [](int kc, char ** kv) { Kokkos::initialize(kc, kv); }(
    d.kokkos.size(), arguments::pointers(d.kokkos).data());
#else
  (void)d;
#endif
}
dependencies_guard::~dependencies_guard() {
#ifdef FLECSI_ENABLE_KOKKOS
  Kokkos::finalize();
#endif
}

context_t::context_t(const arguments::config & c, arguments::action & a)
  : context(c, a, util::mpi::size(), util::mpi::rank()) {}

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
