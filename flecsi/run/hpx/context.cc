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
  params.cfg = {// allocate at least two cores
    "hpx.force_min_os_threads!=2",
    "hpx.handle_signals!=0",
    // call the below on every process
    "hpx.run_hpx_main!=1"};
  params.cfg.insert(params.cfg.end(),
    std::move_iterator(cfg.begin()),
    std::move_iterator(cfg.end()));
  cfg.clear();

  char * argv{};
  return ::hpx::init(
    [=](int, char *[]) -> int {
      // manage task_local variables for this task
      run::task_local_base::guard tlg;

      context::start();

      context::process_ = ::hpx::get_locality_id();
      context::processes_ = ::hpx::get_num_localities(::hpx::launch::sync);
      context::threads_per_process_ = ::hpx::get_num_worker_threads();
      context::threads_ = context::processes_;

      // guard destroyed after action call
      const int ret = (flecsi::detail::data_guard(), action());

      // free communicators (must happen before hpx::finalize as the
      // cleanup operations require for the runtime system to be up and
      // running)
      p2p_comms_.clear();
      world_comms_.clear();

      // tell the runtime it's ok to exit
      ::hpx::finalize();
      return ret;
    },
    0,
    &argv,
    params);
}

template<typename Map, typename CreateComm>
auto
context_t::get_communicator_data(Map & map,
  std::string name,
  CreateComm && create_comm) {

  std::unique_lock l(mtx);
  auto it = map.find(name);
  if(it == map.end()) {
    // create new communicator if not found in the map
    typename Map::mapped_type::first_type comm;
    {
      ::hpx::unlock_guard<decltype(l)> ul(l);
      comm = create_comm(name.c_str(),
        ::hpx::collectives::num_sites_arg(context::processes_),
        ::hpx::collectives::this_site_arg(context::process_));
    }

    // try to insert the newly created communicator (might already have been
    // inserted concurrently)
    it = map.try_emplace(std::move(name), std::move(comm), 0).first;
  }
  ++it->second.second; // increment generation
  return it->second;
}

context_t::communicator_data
context_t::world_comm(std::string name) {

  flog_assert(!name.empty(), "communicators must have valid name");
  auto comm = get_communicator_data(
    world_comms_, "/flecsi/world_comm/" + std::move(name), [](auto &&... args) {
      return ::hpx::collectives::create_communicator(
        std::forward<decltype(args)>(args)...);
    });
  comm.first.set_info(hpx::collectives::num_sites_arg(processes_),
    hpx::collectives::this_site_arg(process_));
  return comm;
}

context_t::channel_communicator_data
context_t::p2p_comm(std::string name) {

  flog_assert(!name.empty(), "communicators must have valid name");
  return get_communicator_data(
    p2p_comms_, "/flecsi/p2p_comm/" + std::move(name), [](auto &&... args) {
      return ::hpx::collectives::create_channel_communicator(
        ::hpx::launch::sync, std::forward<decltype(args)>(args)...);
    });
}

void
context_t::termination_detection() {
  auto & tm = hpx::threads::get_thread_manager();
  hpx::util::yield_while(
    [&tm]() -> bool {
      // we need to wait until all HPX threads (except the current one plus all
      // background threads) have exited
      return tm.get_thread_count() >
             static_cast<std::int64_t>(tm.get_background_thread_count() + 1);
    },
    "termination_detection");
}
} // namespace flecsi::run

namespace flecsi::detail {

void
create_storage() {
  auto * stg = storage();
  if(stg == nullptr) {
    // first creation of thread local storage
    ::hpx::threads::set_thread_data(::hpx::threads::get_outer_self_id(),
      reinterpret_cast<std::size_t>(new task_local_data()));
  }
  else {
    ++stg->count;
  }
}

task_local_data *
storage() noexcept {
  return reinterpret_cast<task_local_data *>(
    ::hpx::threads::get_thread_data(::hpx::threads::get_outer_self_id()));
}

void
reset_storage() noexcept {
  auto * stg = storage();
  flog_assert(stg != nullptr, "thread local storage should still exist");
  if(--stg->count == 0) {
    ::hpx::threads::set_thread_data(::hpx::threads::get_outer_self_id(), 0);
    delete stg;
  }
}
} // namespace flecsi::detail
