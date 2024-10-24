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

template<typename Derived>
struct future_impl;

template<typename R>
struct future_impl<future<R>> {

  future_impl() = default;

  explicit future_impl(::hpx::future<R> && result) noexcept
    : future_(::hpx::make_shared_future(std::move(result))) {}

  future<R> & operator=(::hpx::future<R> && result) noexcept {
    return *this = future<R>(std::move(result));
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
struct future<R> : detail::future_impl<future<R>> {

  using base_type = detail::future_impl<future>;
  using detail::future_impl<future>::future_impl;

  explicit future(R result)
    : base_type(::hpx::make_ready_future(std::move(result))) {}

  future & operator=(R result) {
    return *this = future(std::move(result));
  }
};

template<>
struct future<void> : detail::future_impl<future<void>> {

  using base_type = detail::future_impl<future>;
  using detail::future_impl<future>::future_impl;

  future() : base_type(::hpx::make_ready_future()) {}
};

namespace detail {

template<typename R>
struct future_impl<future<R, exec::launch_type_t::index>> {

  using result_type =
    std::conditional_t<std::is_void_v<R>, void, std::vector<R>>;

  explicit future_impl(::hpx::future<result_type> && result) noexcept
    : future_(::hpx::make_shared_future(std::move(result))) {}

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
struct future<R, exec::launch_type_t::index>
  : detail::future_impl<future<R, exec::launch_type_t::index>> {

private:
  static decltype(auto) all_gather_result(
    flecsi::run::context_t::communicator_data && comm_gen,
    R && result) {
    auto const & [comm, generation] = comm_gen;
    using namespace ::hpx::collectives;
    return all_gather(comm, std::move(result), generation_arg(generation));
  }

public:
  using base_type = detail::future_impl<future>;
  using detail::future_impl<future>::future_impl;

  explicit future(::hpx::future<R> && result, std::string name)
    : base_type(result.then(::hpx::launch::sync,
        [comm_gen = flecsi::run::context::instance().world_comm(
           std::move(name))](auto && f) mutable {
          return future::all_gather_result(std::move(comm_gen), f.get());
        })) {}

  explicit future(R result, std::string name)
    : base_type(all_gather_result(std::move(name), std::move(result))) {}

  R get(Color index = 0, bool = false) {
    return this->base_type::get().at(index);
  }
};

template<>
struct future<void, exec::launch_type_t::index>
  : detail::future_impl<future<void, exec::launch_type_t::index>> {

  using base_type = detail::future_impl<future>;
  using detail::future_impl<future>::future_impl;

  future() : base_type(::hpx::make_ready_future()) {}

  void get(Color = 0, bool = false) {
    this->base_type::get();
  }
};

namespace detail {
// Helper comparing HPX futures
inline bool
is_same(::hpx::shared_future<void> const & lhs,
  ::hpx::shared_future<void> const & rhs) {
  return ::hpx::traits::detail::get_shared_state(lhs) ==
         ::hpx::traits::detail::get_shared_state(rhs);
}
} // namespace detail
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_FUTURE_HH
