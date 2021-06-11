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

  future() = default;

  explicit future(R result)
    : future_(::hpx::make_ready_future(std::move(result))) {}

  explicit future(::hpx::future<R> && result) : future_(std::move(result)) {}

  future & operator=(R result) {
    future_ = ::hpx::make_ready_future(std::move(result));
    return *this;
  }
  future & operator=(::hpx::future<R> && result) {
    future_ = std::move(result);
    return *this;
  }

  void wait() {
    future_.wait();
  }
  R get(bool = false) {
    return future_.get();
  }

private:
  ::hpx::shared_future<R> future_;
};

template<>
struct future<void> {
  void wait() {}
  void get(bool = false) {}
};

template<typename R>
struct future<R, exec::launch_type_t::index> {

  explicit future(R result)
    : results(::hpx::collectives::all_gather(
        flecsi::run::context::instance().world_comm(),
        std::move(result))) {}

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
  ::hpx::shared_future<std::vector<R>> results;
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
