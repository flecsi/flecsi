// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_POLICY_HH
#define FLECSI_EXEC_LEG_POLICY_HH

#include "flecsi/config.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/exec/leg/future.hh"
#include "flecsi/exec/leg/reduction_wrapper.hh"
#include "flecsi/exec/leg/task_wrapper.hh"
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
  using traits_t = util::function_t<F>;
  using return_t = typename traits_t::return_type;
  using param_tuple = typename traits_t::arguments_type;

  // Get the FleCSI runtime context
  auto & flecsi_context = run::context::instance();

  // Get the processor type.
  constexpr auto processor_type = mask_to_processor_type(Attributes);

  // Get the Legion runtime and context from the current task.
  auto legion_runtime = Legion::Runtime::get_runtime();
  auto legion_context = Legion::Runtime::get_context();

  constexpr bool mpi_task = processor_type == processor::mpi;
  static_assert(processor_type == processor::toc ||
                  processor_type == processor::loc ||
                  processor_type == processor::omp || mpi_task,
    "Unknown launch type");
  const auto domain_size = launch_size<Attributes, param_tuple>(args...);

  // We do not generate a separate task_wrapper specialization for each set of
  // argument types, so we construct a tuple whose type is independent of the
  // those types.  Since an MPI task can use references to the
  // original arguments, we have to provide references, which in turn requires
  // separate storage for any objects created by argument conversions (absent
  // excessive variadic aggregate gymnastics to create lifetime-extended
  // temporaries).

  run::any any;
  auto & params = any.emplace(leg::parameters(
    make_parameters<mpi_task, param_tuple>(std::forward<Args>(args)...)));
  prolog<mask_to_processor_type(Attributes)> pro(params.params, args...);
  params.which = std::move(pro).bindings();
  std::optional<leg::parameters<param_tuple>> mpi_params;
  std::vector<std::byte> buf;
  if constexpr(mpi_task) {
    // MPI tasks must be invoked collectively from one task on each rank.
    // We therefore can transmit merely a pointer to a tuple of the arguments.
    // The TaskArgument must be identical on every shard, so use the context.
    flecsi_context.mpi_params = &mpi_params.emplace(std::move(params));
  }
  else {
    const auto t = trace::current();
    buf = util::serial::put_tuple(
      flecsi_context.params.add(std::move(any), t ? t->next() : nullptr));
  }

  using wrap = leg::task_wrapper<F, processor_type>;
  // Replace the MPI "processor type" with an actual flag:
  const auto task = leg::task_id<wrap::execute,
    (Attributes & ~processor_mask) | as_mask(wrap::LegionProcessor)>;

  const auto add = [&](auto & l) {
    l.region_requirements = std::move(pro).region_requirements();
    l.futures = std::move(pro).futures();
    switch(processor_type) {
      case processor::toc:
        l.tag = run::mapper::prefer_gpu;
        break;
      case processor::omp:
        l.tag = run::mapper::prefer_omp;
        break;
      // Null default is added to suppress warning for other enumerators that
      // do nothing
      default:
        break;
    }
  };

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

    if(mpi_task) {
      launcher.tag = run::mapper::force_rank_match;
      legion_runtime->issue_execution_fence(legion_context);
    }

    if constexpr(!std::is_void_v<Reduction>) {
      auto ret = future<return_t, launch_type_t::single>{{},
        legion_runtime->execute_index_space(
          legion_context, launcher, fold::wrap<Reduction, return_t>::REDOP_ID)};
      if(mpi_task)
        ret.wait();
      return ret;
    }
    else {
      auto ret = future<return_t, launch_type_t::index>{
        legion_runtime->execute_index_space(legion_context, launcher)};
      if(mpi_task)
        ret.wait();

      return ret;
    } // if reduction

  } // if constexpr

} // reduce_internal

/// \}
} // namespace exec
} // namespace flecsi

#endif
