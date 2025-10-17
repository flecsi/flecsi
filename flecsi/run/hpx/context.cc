// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include <hpx/hpx_init.hpp>
#include <hpx/modules/execution_base.hpp> // yield_while
#include <hpx/modules/runtime_local.hpp> // get_thread_manager
#include <hpx/mutex.hpp> // unlock_guard

#include "flecsi/data.hh"
#include "flecsi/run/hpx/context.hh"
#include "flecsi/util/mpi.hh"

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace flecsi::run {

context_t::context_t(const config & c) : context(c), cfg(c.hpx) {}

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(std::function<int()> const & action, bool) {

  ::hpx::init_params params;
  // HPX doesn't know its own options for some reason, so we have to use a !:
  params.cfg = {
    // Instruct the HPX runtime to occupy at least two cores for scheduling
    // FleCSI tasks.  This setting has to be taken into account when
    // running more than one HPX locality (rank) on the same node.  Any
    // single node should not run more than `N` localities, where `N ==
    // num_cores / 2`.
    "hpx.force_min_os_threads!=2",
    // Disable installing HPX signal handlers as FleCSI is a library
    // and therefore should not consume signals itself.
    "hpx.handle_signals!=0",
    // Instruct HPX to schedule the initial task (the lambda passed to
    // `hpx::init`) on all localities.
    "hpx.run_hpx_main!=1"};
  params.cfg.insert(params.cfg.end(),
    std::move_iterator(cfg.begin()),
    std::move_iterator(cfg.end()));
  cfg.clear();

  char * argv{};
  return ::hpx::init(
    [&](int, char *[]) -> int {
      // manage task_local variables for this task
      run::task_local_base::guard tlg;

      context::start();

      flog_assert(::hpx::get_locality_id() == process_,
        "HPX locality " << ::hpx::get_locality_id() << " != MPI rank "
                        << process_);
      flog_assert(::hpx::get_num_localities(::hpx::launch::sync) == processes_,
        "HPX locality count " << ::hpx::get_num_localities(::hpx::launch::sync)
                              << " != MPI size " << processes_);
      context::threads_per_process_ = ::hpx::get_num_worker_threads();
      context::threads_ = context::processes_;
      channel = ::hpx::collectives::create_channel_communicator(
        ::hpx::launch::sync, "/flecsi/p2p_comm");
      world0 = world_comm();

      struct guard {
        context_t & c;
        ~guard() {
          c.channel = {};
          c.world0 = {};
          ::hpx::finalize();
        }
      } g{*this};
      return flecsi::detail::data_guard(), action();
    },
    0,
    &argv,
    params);
}

communicator
context_t::world_comm() {
  using namespace ::hpx::collectives;
  return create_communicator("/flecsi/world_comm/",
    num_sites_arg(processes_),
    this_site_arg(process_),
    generation_arg(world++));
}

void
context_t::termination_detection() {
  std::unique_lock g(out_mutex);
  out_cv.wait(g, [this] { return !out; });
}
} // namespace flecsi::run

namespace flecsi::detail {
namespace {
void
storage(task_local_data * p) noexcept {
  ::hpx::threads::set_thread_data(
    ::hpx::threads::get_outer_self_id(), reinterpret_cast<std::size_t>(p));
}
} // namespace

void
create_storage() {
  storage(new task_local_data());
}

task_local_data *
storage() noexcept {
  return reinterpret_cast<task_local_data *>(
    ::hpx::threads::get_thread_data(::hpx::threads::get_outer_self_id()));
}

void
reset_storage() noexcept {
  delete storage();
  storage(nullptr);
}
} // namespace flecsi::detail
