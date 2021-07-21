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
#include "flecsi/flog.hh"

#include <string>
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
    flog_assert(future_.valid(), "future must be valid");
    future_.wait();
  }
  R get(bool = false) {
    flog_assert(future_.valid(), "future must be valid");
    return future_.get();
  }

private:
  ::hpx::shared_future<R> future_;
};

template<>
struct future<void> {

  future() : future_(::hpx::make_ready_future()) {}

  explicit future(::hpx::future<void> && result) : future_(std::move(result)) {}

  future & operator=(::hpx::future<void> && result) {
    future_ = std::move(result);
    return *this;
  }

  void wait() {
    flog_assert(future_.valid(), "future must be valid");
    future_.wait();
  }
  void get(bool = false) {
    flog_assert(future_.valid(), "future must be valid");
    future_.get();
  }

private:
  ::hpx::shared_future<void> future_;
};

template<typename R>
struct future<R, exec::launch_type_t::index> {

  explicit future(R result, std::string name = std::string())
    : results(::hpx::collectives::all_gather(
        flecsi::run::context::instance().world_comm(std::move(name)),
        std::move(result))) {}

  explicit future(::hpx::future<R> && result, std::string name = std::string())
    : results(result.then(::hpx::launch::sync,
        [name = std::move(name)](auto && f) mutable {
          return ::hpx::collectives::all_gather(
            flecsi::run::context::instance().world_comm(std::move(name)),
            f.get());
        })) {}

  explicit future(::hpx::future<std::vector<R>> && result)
    : results(std::move(result)) {}

  void wait(bool = false) {
    flog_assert(results.valid(), "future must be valid");
    results.wait();
  }

  R get(Color index = 0, bool = false) {
    flog_assert(results.valid(), "future must be valid");
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

  future() = default;

  explicit future(::hpx::future<void> && result) : future_(std::move(result)) {
    flog_assert(future_.valid(), "future must be valid");
    future_.get();
  }

  void wait(bool = false) {
    flog_assert(future_.valid(), "future must be valid");
    future_.wait();
  }
  void get(Color = 0, bool = false) {
    flog_assert(future_.valid(), "future must be valid");
    future_.get();
  }
  Color size() const {
    return run::context::instance().processes();
  }

private:
  ::hpx::shared_future<void> future_;
};
} // namespace flecsi
