// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_FUTURE_HH
#define FLECSI_EXEC_HPX_FUTURE_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>

#include "flecsi/config.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/flog.hh"
#include "flecsi/run/context.hh"

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace flecsi {
namespace detail {

template<typename R>
struct future_impl {

  future_impl() = default;

  future_impl(::hpx::shared_future<R> f) noexcept : future_(std::move(f)) {}

  ::hpx::shared_future<void> depend() {
    return future_;
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

} // namespace detail

template<typename R>
struct future<R> : detail::future_impl<R> {
  using base_type = typename future::future_impl;
  using base_type::base_type;

  explicit future(R result)
    : base_type(::hpx::make_ready_future(std::move(result))) {}

  future & operator=(R result) {
    return *this = future(std::move(result));
  }
};

template<>
struct future<void> : detail::future_impl<void> {
  using base_type = typename future::future_impl;
  using base_type::base_type;

  future() : base_type(::hpx::make_ready_future()) {}
};

namespace detail {

template<typename R>
struct future_index {
  using future = ::hpx::shared_future<R>;

  explicit future_index(future f) noexcept : future_(std::move(f)) {}

  auto mine() {
    return future_;
  }

  void wait(bool = false) {
    flog_assert(future_.valid(), "future must be valid");
    future_.wait();
    ::hpx::distributed::barrier::synchronize();
  }

  R get() {
    flog_assert(future_.valid(), "future must be valid");
    return future_.get();
  }

  Color size() const {
    return run::context::instance().processes();
  }

private:
  future future_;
};

} // namespace detail

template<typename R>
struct future<R, exec::launch_type_t::index> : detail::future_index<R> {
  using base_type = typename future::future_index;
  using base_type::base_type;

  R get(Color index = 0, bool = false) {
    auto & c = run::context::instance();
    if(index == c.process()) {
      R ret = base_type::get();
      ::hpx::collectives::broadcast_to(c.world0.comm(), ret, c.world0.gen());
      return ret;
    }
    return ::hpx::collectives::broadcast_from<R>(
      c.world0.comm(), c.world0.gen())
      .get();
  }
};

template<>
struct future<void, exec::launch_type_t::index> : detail::future_index<void> {
  using base_type = typename future::future_index;
  using base_type::base_type;

  void get(Color = 0, bool = false) {
    wait();
  }
};

} // namespace flecsi

#endif // FLECSI_EXEC_HPX_FUTURE_HH
