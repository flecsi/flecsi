// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include <hpx/hpx_init.hpp>
#include <hpx/local/mutex.hpp> // unlock_guard
#include <hpx/modules/execution_base.hpp> // yield_while
#include <hpx/modules/runtime_local.hpp> // get_thread_manager

#include "flecsi/data.hh"
#include "flecsi/run/hpx/context.hh"
#include "flecsi/util/mpi.hh"

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace flecsi::run {

//----------------------------------------------------------------------------//
// Implementation of context_t::start.
//----------------------------------------------------------------------------//

int
context_t::start(std::function<int()> const & action) {

  ::hpx::init_params params;
  params.cfg = {// allocate at least two cores
    "hpx.force_min_os_threads!=2",
    // make sure hpx_main is always executed
    "hpx.run_hpx_main!=1",
    // force HPX to use multi-threaded MPI
    "hpx.parcel.mpi.multithreaded!=1",
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

      // guard destroyed after action call
      context::exit_status() = (flecsi::detail::data_guard(), action());

      // free communicators (must happen before hpx::finalize as the
      // cleanup operations require for the runtime system to be up and
      // running)
      p2p_comms_.clear();
      world_comms_.clear();

      // tell the runtime it's ok to exit
      ::hpx::finalize();
      return context::exit_status();
    },
    argv_.size(),
    argv_.data(),
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

  auto comm = get_communicator_data(
    world_comms_, "/flecsi/world_comm/" + std::move(name), [](auto &&... args) {
      return ::hpx::collectives::create_communicator(
        std::forward<decltype(args)>(args)...);
    });
  comm.first.set_info(processes_, process_);
  return comm;
}

context_t::channel_communicator_data
context_t::p2p_comm(std::string name) {

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
             std::int64_t(tm.get_background_thread_count() + 1);
    },
    "termination_detection");
}

} // namespace flecsi::run
