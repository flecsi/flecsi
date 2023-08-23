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

  using channel_communicator_data =
    std::pair<::hpx::collectives::channel_communicator, std::size_t>;
  using communicator_data =
    std::pair<::hpx::collectives::communicator, std::size_t>;

  channel_communicator_data p2p_comm(std::string name);
  communicator_data world_comm(std::string name);

  static void termination_detection();

private:
  template<typename Map, typename CreateComm>
  auto
  get_communicator_data(Map & map, std::string name, CreateComm && create_comm);

  std::vector<std::string> cfg;
  ::hpx::spinlock mtx;
  std::map<std::string, channel_communicator_data> p2p_comms_;
  std::map<std::string, communicator_data> world_comms_;
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
    return data.find(key);
  }

  constexpr bool outermost() const noexcept {
#if HPX_VERSION_FULL >= 0x011000
    return count == 1;
#else
    return true;
#endif
  }

  std::map<void *, void *> data;

#if HPX_VERSION_FULL >= 0x011000
  std::int16_t count = 1;
#endif
};

// manage task local storage for this task
void create_storage();
task_local_data * storage() noexcept;
void reset_storage() noexcept;

template<typename T>
void
add(void * key) {
  [[maybe_unused]] auto p = storage()->emplace<T>(key);
  flog_assert(p.second || !storage()->outermost(),
    "task local storage element should not have been created yet");
}

template<typename T>
void
erase(void * key) noexcept {
  auto * stg = storage();
  auto it = stg->find(key);
  flog_assert(
    it != stg->end(), "task local storage element should have been created");
  if(stg->outermost()) {
    delete static_cast<T *>((*it).second);
    (*it).second = nullptr;
  }
}

template<typename T>
T *
get(void * key) noexcept {
  auto * stg = storage();
  auto it = stg->find(key);
  flog_assert(
    it != stg->end(), "task local storage element should have been created");
  return static_cast<T *>((*it).second);
}
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
    detail::add<T>(this);
  }
  void reset() noexcept override {
    detail::erase<T>(this);
  }
  void create_storage() override {
    detail::create_storage();
  }
  void reset_storage() noexcept override {
    detail::reset_storage();
  }

  T * get() noexcept {
    return detail::get<T>(this);
  }
};
} // namespace flecsi

#endif // FLECSI_RUN_HPX_CONTEXT_HH
