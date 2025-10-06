// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_LEG_CONTEXT_HH
#define FLECSI_RUN_LEG_CONTEXT_HH

#include "flecsi/config.hh"
#include "flecsi/run/context.hh"
#include "flecsi/util/mpi.hh"

#include <legion.h>

#include <functional>
#include <map>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace flecsi {
namespace run {
/// \defgroup legion-runtime Legion Runtime
/// State for and control of the Legion runtime.
/// \ns{run::leg}.
/// \ingroup runtime
/// \{

template<class T>
auto
get1(const Legion::Task & t) {
  const auto p = static_cast<const std::byte *>(t.args);
  return util::serial::get1<T>(p, p + t.arglen);
}

namespace mapper {
/// \addtogroup legion-runtime
/// \{

/// \name Mapper tags
/// Flags used to request custom mapper features.
/// \ns::run::mapper.
/// \{

inline constexpr Legion::MappingTagID
  force_rank_match = 0x00001000, ///< Put colors on corresponding MPI ranks.
#if 0
  compacted_storage = 0x00002000, ///< Combine exclusive, shared, and ghosts.
  subrank_launch = 0x00003000, ///< For nested tasks.
  exclusive_lr = 0x00004000, ///< Indicate first region in compacted set.
#endif
  prefer_gpu = 0x11000001, ///< Request GPU execution.
  prefer_omp = 0x11000002; ///< Request OpenMP execution.
/// \}
/// \}
} // namespace mapper

using task_idx = std::size_t;

class param_locker;

// The number of point tasks for a process to execute may become known only
// after several task launches that share it (via tracing).
struct task_count {
  using type = Color;
  using ptr = std::shared_ptr<task_count>; // to outlive trace as needed

  type set(type n) {
    // If count is engaged, this is mapper output ignored by a trace.
    return count ? *count : count.emplace(n);
  }

private:
  std::optional<type> count;
};

// A move-only subset of std::any.
struct any_base {
  virtual ~any_base() = default;
  virtual void * get(const std::type_info &) = 0;
};

template<class T>
struct any_impl : any_base {
  any_impl(T t) : t(std::move(t)) {}
  void * get(const std::type_info & i) override {
    if(i != typeid(T))
      throw std::bad_cast();
    return &t;
  }
  T t;
};

struct any {
  template<class T>
  std::decay_t<T> & emplace(T && t) {
    auto * const q = new any_impl<std::decay_t<T>>(std::forward<T>(t));
    p.reset(q);
    return q->t;
  }

  explicit operator bool() const {
    return !!p;
  }
  template<class T>
  T & get() {
    return *static_cast<T *>(p->get(typeid(T)));
  }
  template<class T>
  const T & get() const {
    return const_cast<any &>(*this).get<T>();
  }
  template<class T>
  T && get() && {
    return std::move(get<T>());
  }

private:
  std::unique_ptr<any_base> p;
};

class param_locker
{
  struct task {
    task(task_count::ptr tc, any && p)
      : tc(std::move(tc)), params(std::move(p)) {}

    task_count::ptr tc;
    util::ref_count<task_count::type> ref{1};
    any params;
  };

  using Map = std::map<task_idx, task>;
  std::mutex lock;
  Map tasks;
  task_idx id = 0;

  auto lease() {
    return std::unique_lock(lock);
  }

  class guard
  {
    param_locker & lk;
    Map::iterator it;

  public:
    guard(param_locker & lk, task_idx i) : lk(lk), it(lk.tasks.find(i)) {
      flog_assert(it != lk.tasks.end(), "no such task");
    }
    guard(guard &&) = delete;

    ~guard() {
      if(--it->second.ref)
        lk.lease(), lk.tasks.erase(it);
    }

    void post(task_count::type n) const {
      const auto & p = it->second.tc;
      it->second.ref += p ? p->set(n) : n;
    }
    template<class T>
    T & get() const {
      return it->second.params.get<T>();
    }
  };

public:
  [[nodiscard]] task_idx add(any && a, task_count::ptr tc) {
    return lease(), tasks.try_emplace(id, std::move(tc), std::move(a)), id++;
  }
  guard at(task_idx i) {
    return lease(), guard(*this, i);
  }
};

namespace leg {
template<class R = void>
using task = R(const Legion::Task *,
  const std::vector<Legion::PhysicalRegion> &,
  Legion::Context,
  Legion::Runtime *) noexcept;
}

struct dependencies_guard : util::mpi::init {
  dependencies_guard(dependencies_config = {});
};

struct config : config_base {
  argv legion;
  argv * backend() & {
    return &legion;
  }
};

struct context_t : context {
  context_t(const config &);

  [[nodiscard]] int start(const std::function<int()> &, bool);

  static int task_depth() {
    return Legion::Runtime::get_runtime()
      ->get_current_task(Legion::Runtime::get_context())
      ->get_depth();
  } // task_depth

  static Color color() {
    flog_assert(
      task_depth() > 0, "this method can only be called from within a task");
    return Legion::Runtime::get_runtime()
      ->get_current_task(Legion::Runtime::get_context())
      ->index_point.point_data[0];
  } // color

  static Color colors() {
    flog_assert(
      task_depth() > 0, "this method can only be called from within a task");
    return Legion::Runtime::get_runtime()
      ->get_current_task(Legion::Runtime::get_context())
      ->index_domain.get_volume();
  } // colors

  param_locker params;

  //--------------------------------------------------------------------------//
  //  MPI interoperability.
  //--------------------------------------------------------------------------//

  void * mpi_params;

  /*!
    Set the MPI user task. When control is given to the MPI runtime
    it will execute whichever function is currently set.
   */

  void mpi_call(std::function<void()> mpi_task) {
    mpi_task_ = std::move(mpi_task);
    mpi_handoff();
    mpi_wait();
  }

  /*!
    Handoff to MPI from Legion.
   */
  void mpi_handoff() {
    handshake_.legion_handoff_to_mpi();
  }

  /*!
    Wait for MPI runtime to complete task execution.
   */

  void mpi_wait() {
    handshake_.legion_wait_on_mpi();
  }

  [[nodiscard]] static Legion::LocalVariableID local_variable() {
    return next_var++;
  }

private:
  static leg::task<> top_level_task;

  /*--------------------------------------------------------------------------*
    Runtime data.
   *--------------------------------------------------------------------------*/

  static inline Legion::LocalVariableID next_var;
  run::argv argv;
  const std::function<int()> * top_level_action_ = nullptr;

  /*--------------------------------------------------------------------------*
    Interoperability data members.
   *--------------------------------------------------------------------------*/

  std::function<void()> mpi_task_;
  Legion::MPILegionHandshake handshake_;
};

/// \}
} // namespace run

template<class T>
struct task_local : run::task_local_base {
  task_local() : var(run::context_t::local_variable()) {}

  T & operator*() noexcept {
    return Run::has_context() ? *Run::get_runtime()->get_local_task_variable<T>(
                                  Run::get_context(), var)
                              : *mpi;
  }
  T * operator->() noexcept {
    return &**this;
  }

private:
  using Run = Legion::Runtime;

  void emplace() override {
    if(Run::has_context())
      Run::get_runtime()->set_local_task_variable(
        Run::get_context(), var, new T(), [](void * p) {
          delete static_cast<T *>(p);
        });
    else
      mpi.emplace();
  }
  void reset() noexcept override {
    if(!Run::has_context())
      mpi.reset();
  }

  std::optional<T> mpi;
  Legion::LocalVariableID var;
};

} // namespace flecsi

#endif
