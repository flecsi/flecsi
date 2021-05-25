// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_HPX_CONTEXT_HH
#define FLECSI_RUN_HPX_CONTEXT_HH

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/synchronization.hpp>

#include "flecsi/run/local/context.hh"

#include <cstddef>
#include <map>
#include <utility>

namespace flecsi::run {
/// \defgroup hpx-runtime HPX Runtime
/// Global state.
/// \ingroup runtime
/// \{

struct context_t : local::context {

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//

  int start(const std::function<int()> &);

  Color process() const {
    return process_;
  }

  Color processes() const {
    return processes_;
  }

  Color threads_per_process() const {
    return threads_per_process_;
  }

  Color threads() const {
    return threads_;
  }

  static int task_depth() {
    return 0;
  } // task_depth

  Color color() const {
    return process_;
  }

  Color colors() const {
    return processes_;
  }

  using channel_communicator_data =
    std::pair<::hpx::collectives::channel_communicator, std::size_t>;
  using communicator_data =
    std::pair<::hpx::collectives::communicator, std::size_t>;

  channel_communicator_data p2p_comm(std::string name);
  communicator_data world_comm(std::string name);

  static void termination_detection();

private:
  template<typename Map, typename CreateComm>
  auto
  get_communicator_data(Map & map, std::string name, CreateComm && create_comm);

  ::hpx::spinlock mtx;
  std::map<std::string, channel_communicator_data> p2p_comms_;
  std::map<std::string, communicator_data> world_comms_;
};

/// \}
} // namespace flecsi::run

#endif // FLECSI_RUN_HPX_CONTEXT_HH
