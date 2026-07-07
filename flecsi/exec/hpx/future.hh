// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_FUTURE_HH
#define FLECSI_EXEC_HPX_FUTURE_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/futures.hpp>

#include "flecsi/flog.hh"
#include "flecsi/run/backend.hh"

#include <utility>

namespace flecsi {
namespace detail {

struct future_base {
  run::comms::ptr get_comms() const {
    return comms;
  }

  void silence() { // for task parameters, where waiting isn't interesting
    comms = {};
  }

protected:
  future_base(run::comms::ptr c = {}) : comms(std::move(c)) {}

  void wait() {
    run::context::instance().depend(std::move(comms));
  }

private:
  run::comms::ptr comms;
};

template<typename R>
struct future_impl : future_base {

  future_impl() = default;

  future_impl(::hpx::shared_future<R> f, run::comms::ptr c = {}) noexcept
    : future_base(std::move(c)), future_(std::move(f)) {}

  inline future<void> depend() const;
  auto backend() const {
    return future_;
  }

  void wait() {
    flog_assert(future_.valid(), "future must be valid");
    future_base::wait();
    future_.wait();
  }
  R get(bool = false) {
    flog_assert(future_.valid(), "future must be valid");
    future_base::wait();
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
struct future_index : future_base {
  using future = ::hpx::shared_future<R>;

  explicit future_index(future f, run::comms::ptr c) noexcept
    : future_base(std::move(c)), future_(std::move(f)) {}

  inline flecsi::future<void, future_kind::index> depend() const;
  auto backend() const {
    return future_;
  }

  void wait(bool = false) {
    flog_assert(future_.valid(), "future must be valid");
    future_base::wait();
    future_.wait();
    ::hpx::distributed::barrier::synchronize();
  }

  R get() {
    flog_assert(future_.valid(), "future must be valid");
    // No future_base::wait(): this isn't reliably called on all processes.
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
    auto & comm = c.world_comms->get();
    if(index == c.process()) {
      R ret = base_type::get();
      ::hpx::collectives::broadcast_to(comm.comm(), ret, comm.gen());
      return ret;
    }
    return ::hpx::collectives::broadcast_from<R>(comm.comm(), comm.gen()).get();
  }
  std::vector<R> all() {
    // c might be used by the task, so we can call gen only after get returns.
    auto r = base_type::get();
    this->future_base::wait();
    auto & c = run::context::instance().world_comms->get();
    return ::hpx::collectives::all_gather(c.comm(), std::move(r), c.gen())
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

template<class R>
future<void>
detail::future_impl<R>::depend() const {
  return future<void>(backend(), get_comms());
}
template<class R>
future<void, future_kind::index>
detail::future_index<R>::depend() const {
  return flecsi::future<void, future_kind::index>(backend(), get_comms());
}

template<class R>
future<std::remove_cvref_t<R>>
make_future(R && r) {
  return future<std::remove_cvref_t<R>>(std::forward<R>(r));
}

} // namespace flecsi

#endif // FLECSI_EXEC_HPX_FUTURE_HH
