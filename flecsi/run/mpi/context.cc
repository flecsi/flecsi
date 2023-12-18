#include "flecsi/run/mpi/context.hh"
#include "flecsi/data.hh"

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace flecsi::run {

dependencies_guard::dependencies_guard(dependencies_config d)
  : mpi(d.mpi.size(), pointers(d.mpi).data())
#ifdef FLECSI_ENABLE_KOKKOS
    ,
    kokkos(d.kokkos)
#endif
{
  (void)d;
}

context_t::context_t(const config & c)
  : context(c, util::mpi::size(), util::mpi::rank()) {}

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(const std::function<int()> & action, bool) {
  context::start();

  context::threads_per_process_ = 1;
  context::threads_ = context::processes_;

  return detail::data_guard(), task_local_base::guard(), action();
}

} // namespace flecsi::run
