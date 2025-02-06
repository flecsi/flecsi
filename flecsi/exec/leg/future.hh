// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_FUTURE_HH
#define FLECSI_EXEC_LEG_FUTURE_HH

#include "flecsi/config.hh"
#include "flecsi/exec/launch.hh"

#include <legion.h>

namespace flecsi {

template<typename Return>
struct future<Return> : data::bind_tag {

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

  [[deprecated("pass to a task or use all")]] Return get(Color index = 0,
    bool silence_warnings = false) {
    if constexpr(std::is_same_v<Return, void>)
      return legion_future_.get_void_result(index, silence_warnings);
    else
      return legion_future_.get_result<Return>(index, silence_warnings);
  } // get
  template<class R = Return, class = std::enable_if_t<!std::is_void_v<R>>>
  auto all() {
    const Color n = size();
    std::vector<R> ret;
    ret.reserve(n);
    for(Color i = 0; i < n; ++i)
      ret.push_back(legion_future_.get_result<R>(i));
    return ret;
  }

  Color size() const {
    return legion_future_.get_future_map_domain().get_volume();
  }

  Legion::FutureMap legion_future_;
};

template<class Return>
future<Return>
make_future(Return && val) {
  return {{}, Legion::Future::from_value<Return>(std::forward<Return>(val))};
} // make_future

} // namespace flecsi

#endif
