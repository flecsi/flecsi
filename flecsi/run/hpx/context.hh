// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_HPX_CONTEXT_HH
#define FLECSI_RUN_HPX_CONTEXT_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/synchronization.hpp>

#include "flecsi/config.hh"
#include "flecsi/run/local/context.hh"

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

namespace flecsi::run {
/// \defgroup hpx-runtime HPX Runtime
/// Global state.
/// \ingroup runtime
/// \{

struct communicator {
  using type = ::hpx::collectives::communicator;
  communicator() = default;
  communicator(type c) : c(std::move(c)) {}
  communicator(communicator &&) = default;
  communicator & operator=(communicator &&) & = default;

  const type & comm() const {
    return c;
  }
  auto gen() {
    return ::hpx::collectives::generation_arg(++g);
  }

private:
  type c;
  std::size_t g = 0;
};

struct config : config_base {
  std::vector<std::string> hpx;
};

struct context_t : local::context {

  //--------------------------------------------------------------------------//
  //  Runtime.
  //--------------------------------------------------------------------------//
  context_t(const config &);

  int start(const std::function<int()> &, bool);

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

  using p2p = ::hpx::collectives::channel_communicator;

  const p2p & p2p_comm() const {
    return channel;
  }
  auto p2p_tag() {
    return ::hpx::collectives::tag_arg(++tag);
  }
  communicator world_comm();
  communicator world0;

private:
  struct outstanding_guard {
    outstanding_guard(context_t * c) : c(c) {
      c->out.fetch_add(1, std::memory_order_relaxed);
    }
    outstanding_guard(outstanding_guard && o) noexcept
      : c(std::exchange(o.c, {})) {}
    ~outstanding_guard() {
      if(c && c->out.fetch_sub(1, std::memory_order_release) == 1) {
        (std::lock_guard(c->out_mutex));
        c->out_cv.notify_one();
      }
    }
    outstanding_guard operator()() {
      return std::move(*this);
    }

  private:
    context_t * c;
  };

public:
  outstanding_guard outstanding() {
    return this;
  }
  void termination_detection();

private:
  std::vector<std::string> cfg;
  p2p channel;
  std::size_t tag = 0, world = 0;
  std::atomic<std::size_t> out = 0;
  ::hpx::mutex out_mutex;
  ::hpx::condition_variable out_cv;
};

/// \}
} // namespace flecsi::run

namespace flecsi {
namespace detail {

struct task_local_data {

  auto begin() {
    return data.begin();
  }
  auto end() {
    return data.end();
  }

  template<typename T>
  auto emplace(void * key) {
    return data.emplace(key, new T());
  }

  auto find(void * key) noexcept {
    auto ret = data.find(key);
    flog_assert(
      ret != data.end(), "task local storage element should have been created");
    return ret;
  }

  constexpr bool outermost() const noexcept {
    return count == 1;
  }

  std::map<void *, void *> data;
  std::int16_t count = 1;
};

// manage task local storage for this task
void create_storage();
task_local_data * storage() noexcept;
void reset_storage() noexcept;

} // namespace detail

template<typename T>
struct task_local : private run::task_local_base {
  T & operator*() noexcept {
    return *get();
  }
  T * operator->() noexcept {
    return get();
  }

private:
  void emplace() override {
    [[maybe_unused]] auto p = detail::storage()->emplace<T>(this);
    flog_assert(p.second || !detail::storage()->outermost(),
      "task local storage element should not have been created yet");
  }
  void reset() noexcept override {
    auto * stg = detail::storage();
    auto it = stg->find(this);
    if(stg->outermost()) {
      delete static_cast<T *>((*it).second);
      (*it).second = nullptr;
    }
  }
  void create_storage() override {
    detail::create_storage();
  }
  void reset_storage() noexcept override {
    detail::reset_storage();
  }

  T * get() noexcept {
    return static_cast<T *>(detail::storage()->find(this)->second);
  }
};
} // namespace flecsi

#endif // FLECSI_RUN_HPX_CONTEXT_HH
