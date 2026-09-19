// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_LEG_CONTEXT_HH
#define FLECSI_RUN_LEG_CONTEXT_HH

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
  force_rank_match = 0x400, ///< Put colors on corresponding MPI ranks.
#if 0
  compacted_storage = 0x800, ///< Combine exclusive, shared, and ghosts.
  subrank_launch = 0x1000, ///< For nested tasks.
  exclusive_lr = 0x100, ///< Indicate first region in compacted set.
#endif
  proc_mask = 0x300,
  gpu = 0x100, ///< Select GPU execution.
  omp = 0x200; ///< Select OpenMP execution.
/// \}
/// \}
} // namespace mapper

class param_locker {
  struct task {
    // On process 0 only, the number of instances to run (zero until known, if
    // other processes get ahead) less those that have already run.
    Color ref;
    util::any params; // empty if this process is behind
  };

  using Map = std::map<task_idx, task>;
  std::mutex lock;
  util::mpi::comm comm = util::mpi::comm::dup(MPI_COMM_WORLD);
  Map tasks;
  task_idx id = 1;

  auto lease() {
    return std::unique_lock(lock);
  }

  void send(task_idx i) {
    util::mpi::send(i, 0, 0, comm);
  }
  void run();

  class guard {
    param_locker & lk;
    Map::iterator it;

  public:
    guard(param_locker & lk, task_idx i) : lk(lk), it(lk.tasks.find(i)) {
      flog_assert(it != lk.tasks.end(), "no such task");
    }
    guard(guard &&) = delete;

    ~guard() {
      lk.send(it->first);
    }

    template<class T>
    T & get() const {
      return it->second.params.get<T>();
    }
  };

  struct thread {
    explicit thread(param_locker & p)
      : p(p), t(&param_locker::run, std::ref(p)) {}
    ~thread() {
      if(!rank(p.comm))
        p.send(0);
    }

  private:
    param_locker & p;
    std::jthread t;
  };

public:
  ~param_locker() {
    flog_assert(tasks.empty(), "abandoned parameters");
  }

  [[nodiscard]] task_idx add(util::any && a, Color n) {
    if(n) { // otherwise no broadcast ever happens
      if(lease(), [&] {
           const auto [it, nu] = tasks.try_emplace(id, n, std::move(a));
           if(!nu) {
             auto & r = it->second.ref;
             if(!r)
               tasks.erase(it);
             else {
               it->second.params = std::move(a);
               if(!(r += n)) {
                 ++r; // so that send will trigger broadcast
                 return true;
               }
             }
           }
           return false;
         }())
        send(id);
    }
    return id++;
  }
  guard at(task_idx i) {
    return lease(), guard(*this, i);
  }

  thread clean() {
    return thread(*this);
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
