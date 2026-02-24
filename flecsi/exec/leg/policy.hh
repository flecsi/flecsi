// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_POLICY_HH
#define FLECSI_EXEC_LEG_POLICY_HH

#include "flecsi/exec/future.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/exec/leg/fold.hh"
#include "flecsi/exec/leg/task.hh"
#include "flecsi/exec/leg/tracer.hh"
#include "flecsi/exec/params.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/function_traits.hh"
#include <flecsi/flog.hh>

#include <functional>
#include <memory>
#include <type_traits>

#include <legion.h>

namespace flecsi {
namespace exec {
/// \defgroup legion-execution Legion Execution
/// Potentially remote task execution.
/// \ns{exec::leg}.
/// \ingroup execution
/// \{

template<auto & F, class Reduction, TaskAttributes Attributes, typename... Args>
auto
reduce_internal(Args &&... args) {
  using namespace Legion;
  using launch = exec::launch<F, Attributes>;
  using return_t = typename launch::Return;

  auto & flecsi_context = run::context::instance();
  auto legion_runtime = Legion::Runtime::get_runtime();
  auto legion_context = Legion::Runtime::get_context();

  const auto domain_size = launch::size(args...);

  util::any any;
  auto & params =
    any.emplace(leg::parameters(launch::params(std::forward<Args>(args)...)));
  prolog<launch::proc> pro(params.params, args...);
  params.which = std::move(pro).bindings();
  std::optional<leg::parameters<typename launch::Params>> mpi_params;
  std::vector<std::byte> buf;
  if constexpr(launch::mpi) {
    // We can own the parameters for a synchronous launch, but we can't store
    // the various pointers to them in the one TaskArgument.
    flecsi_context.mpi_params = &mpi_params.emplace(std::move(params));
  }
  else {
    const auto t = trace::current();
    buf = util::serial::put_tuple(
      flecsi_context.params.add(std::move(any), t ? t->next() : nullptr));
  }

  // Replace the MPI "processor type" with an actual flag:
  const auto task = leg::task_id<leg::task_wrapper<launch>,
    (launch::mpi ? (Attributes & ~processor_mask) | loc : Attributes),
    launch::concurrent>;

  const auto add = [&](auto & l) {
    l.region_requirements = std::move(pro).region_requirements();
    l.futures = std::move(pro).futures();
    switch(launch::proc) {
      case processor::toc:
        l.tag = run::mapper::gpu;
        break;
      case processor::omp:
        l.tag = run::mapper::omp;
        break;
      // Null default is added to suppress warning for other enumerators that
      // do nothing
      default:
        break;
    }
  };

  auto ret = [&] {
    if constexpr(std::is_same_v<decltype(domain_size), const std::monostate>) {
      TaskLauncher launcher(task, TaskArgument(buf.data(), buf.size()));
      add(launcher);

      return future<return_t>{
        {}, legion_runtime->execute_task(legion_context, launcher)};
    }
    else {
      IndexTaskLauncher launcher(task,
        Domain(0, static_cast<coord_t>(domain_size) - 1),
        {buf.data(), buf.size()},
        {});
      add(launcher);
      launcher.point_futures.assign(
        pro.future_maps().begin(), pro.future_maps().end());

      if(launch::matched)
        launcher.tag |= run::mapper::force_rank_match;
      launcher.concurrent = launch::concurrent;
      if(launch::mpi)
        scheduler::instance->wait();

      if constexpr(!std::is_void_v<Reduction>)
        return future<return_t>{{},
          legion_runtime->execute_index_space(legion_context,
            launcher,
            fold::wrap<Reduction, return_t>::REDOP_ID)};
      else
        return future<return_t, launch_type_t::index>{
          legion_runtime->execute_index_space(legion_context, launcher)};
    }
  }();
  pro.set_future(ret.depend());
  if(launch::mpi)
    ret.wait();
  return ret;
} // reduce_internal

/// \}
} // namespace exec

void
scheduler::wait() {
  Legion::Runtime::get_runtime()->issue_execution_fence(
    Legion::Runtime::get_context());
}

} // namespace flecsi

#endif
