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

  explicit future_impl(::hpx::shared_future<R> f) noexcept
    : future_(std::move(f)) {}

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

  using result_type =
    std::conditional_t<std::is_void_v<R>, void, std::vector<R>>;

  explicit future_index(::hpx::shared_future<result_type> f) noexcept
    : future_(std::move(f)) {}

  void wait(bool = false) {
    flog_assert(future_.valid(), "future must be valid");
    future_.wait();
  }

  result_type get() {
    flog_assert(future_.valid(), "future must be valid");
    return future_.get();
  }

  Color size() const {
    return run::context::instance().processes();
  }

private:
  ::hpx::shared_future<result_type> future_;
};

} // namespace detail

template<typename R>
struct future<R, exec::launch_type_t::index> : detail::future_index<R> {
  using base_type = typename future::future_index;
  using base_type::base_type;

  explicit future(::hpx::shared_future<R> result)
    : base_type(result.then(::hpx::launch::sync,
        [comm = run::context::instance().world_comm()](auto && f) mutable {
          using namespace ::hpx::collectives;
          return all_gather(comm.comm(), f.get(), comm.gen());
        })) {}

  R get(Color index = 0, bool = false) {
    return this->base_type::get().at(index);
  }
};

template<>
struct future<void, exec::launch_type_t::index> : detail::future_index<void> {
  using base_type = typename future::future_index;
  using base_type::base_type;

  void get(Color = 0, bool = false) {
    this->base_type::get();
  }
};

} // namespace flecsi

#endif // FLECSI_EXEC_HPX_FUTURE_HH
