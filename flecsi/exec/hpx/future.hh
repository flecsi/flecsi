/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Los Alamos National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

/*! @file */

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>

#include "flecsi/exec/launch.hh"

#include <utility>

namespace flecsi {

template<typename R>
struct future<R> {

  explicit future(R result)
    : future_(hpx::make_ready_future(std::move(result))) {}

  void wait() {
    future_.wait();
  }
  R get(bool = false) {
    return future_.get();
  }

  hpx::shared_future<R> future_;
};

template<>
struct future<void> {
  void wait() {}
  void get(bool = false) {}
};

template<typename R>
struct future<R, exec::launch_type_t::index> {

  explicit future(R result)
    : results(hpx::all_gather("hpx_comm_world", std::move(result), size())) {}

  void wait(bool = false) {
      results.wait();
  }

  R get(Color index = 0, bool = false) {
    return results.get().at(index);
  }

  Color size() const {
    return run::context::instance().processes();
  }

private:
  hpx::shared_future<std::vector<R>> results;
};

template<>
struct future<void, exec::launch_type_t::index> {
  void wait(bool = false) {}
  void get(Color = 0, bool = false) {}
  Color size() const {
    return run::context::instance().processes();
  }
};
} // namespace flecsi
