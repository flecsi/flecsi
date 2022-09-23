// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_FUTURE_HH
#define FLECSI_EXEC_LEG_FUTURE_HH

#include <flecsi-config.h>

#include "flecsi/exec/launch.hh"

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include <legion.h>

namespace flecsi {

template<typename Return>
struct future<Return> {

  void wait() {
    legion_future_.wait();
  } // wait

  Return get(bool silence_warnings = false) {
    if constexpr(std::is_same_v<Return, void>)
      return legion_future_.get_void_result(silence_warnings);
    else
      return legion_future_.get_result<Return>(silence_warnings);
  } // get

  Legion::Future legion_future_;
};

template<typename Return>
struct future<Return, exec::launch_type_t::index> {
  void wait(bool silence_warnings = false) {
    legion_future_.wait_all_results(silence_warnings);
  } // wait

  Return get(Color index = 0, bool silence_warnings = false) {
    if constexpr(std::is_same_v<Return, void>)
      return legion_future_.get_void_result(index, silence_warnings);
    else
      return legion_future_.get_result<Return>(index, silence_warnings);
  } // get

  Color size() const {
    return legion_future_.get_future_map_domain().get_volume();
  }

  Legion::FutureMap legion_future_;
};

} // namespace flecsi

#endif
