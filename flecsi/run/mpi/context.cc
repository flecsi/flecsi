#include "flecsi/run/mpi/context.hh"
#include "flecsi/data.hh"

namespace flecsi::run {

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
