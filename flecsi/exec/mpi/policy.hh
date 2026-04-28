// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_POLICY_HH
#define FLECSI_EXEC_MPI_POLICY_HH

#include "flecsi/exec/future.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/exec/mpi/fold.hh"
#include "flecsi/exec/params.hh"
#include "flecsi/exec/tracer.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/function_traits.hh"

#include <mpi.h>

#include <type_traits>
#include <utility> // forward

namespace flecsi {
namespace exec {
/// \defgroup mpi-execution MPI Execution
/// Direct task execution.
/// \ns{exec::mpi}.
/// \ingroup execution
/// \{
template<auto & F, class Reduction, TaskAttributes Attributes, typename... Args>
auto
reduce_internal(Args &&... args) {
  using util::mpi::test;
  using launch = exec::launch<F, Attributes>;
  using R = typename launch::Return;

  // replace arguments in args, for example, field_reference -> accessor.
  auto params = launch::params(std::forward<Args>(args)...);

  prolog<launch::proc> pro(params, args...);
  auto storage = pro.detach();
  bind_parameters<launch::proc> bp(params, storage);

  run::context_t::depth_guard rg;
  run::task_local_base::guard tlg;

  const auto ds = launch::size(args...);
  const auto task = [&params] { return launch::call(std::move(params)); };
  auto ret = [&] {
    if constexpr(std::is_same_v<decltype(ds), const std::monostate>) {
      const bool root = !flecsi::run::context::instance().process();
      // single launch, only invoke the user task on the Root.
      if constexpr(std::is_void_v<R>) {
        // void return type, just invoke, no return value to broadcast
        if(root) {
          task();
        }
        return future<void>{};
      }
      else {
        auto ret = future<R>::make(task, root);

        test(MPI_Ibcast(ret->data(),
          1,
          flecsi::util::mpi::type<R>(),
          0,
          MPI_COMM_WORLD,
          ret->request()));

        return ret;
      }
    }
    else {
      if(ds != run::context::instance().processes())
        flog_fatal("MPI backend supports only per-process index launches");
      // Index launch (including "mpi task"): invoke user task on every process.
      if constexpr(!std::is_void_v<Reduction>) {
        static_assert(!std::is_void_v<R>, "cannot reduce void results");

        // A real reduce operation: every process needs to be able to access the
        // same result through future<R>::get().
        // 1. Call the F, get the local return value
        auto ret = future<R>::make(task, true);

        // 2. Reduce the local return values with the Reduction (using its
        // corresponding MPI_Op created by register_reduction<>()).
        test(MPI_Iallreduce(MPI_IN_PLACE,
          ret->data(),
          1,
          flecsi::util::mpi::type<R>(),
          fold::wrap<Reduction, R>::op(),
          MPI_COMM_WORLD,
          ret->request()));

        return ret;
      }
      else if constexpr(!std::is_void_v<R>)
        return future<R, exec::launch_type_t::index>{task()};
      else {
        // index launch of void functions, e.g. printf("hello world");
        task();
        return future<void, exec::launch_type_t::index>{};
      }
    }
  }();
  pro.set_future(ret.depend()); // just to get the type right
  return ret;
}

/// \}
} // namespace exec

void
scheduler::wait() {}

} // namespace flecsi

#endif
