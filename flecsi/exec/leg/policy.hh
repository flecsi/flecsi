/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include <flecsi-config.h>

#include "flecsi/exec/launch.hh"
#include "flecsi/exec/leg/future.hh"
#include "flecsi/exec/leg/reduction_wrapper.hh"
#include "flecsi/exec/leg/task_wrapper.hh"
#include "flecsi/exec/prolog.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/function_traits.hh"
#include <flecsi/flog.hh>

#include <functional>
#include <memory>
#include <type_traits>

#if !defined(FLECSI_ENABLE_LEGION)
#error FLECSI_ENABLE_LEGION not defined! This file depends on Legion!
#endif

#include <legion.h>

namespace flecsi {

inline flog::devel_tag execution_tag("execution");

namespace exec {
/// \defgroup legion-execution Legion Execution
/// Potentially remote task execution.
/// \ingroup execution
/// \{
namespace detail {

// Remove const from under a reference, if there is one.
template<class T>
struct nonconst_ref {
  using type = T;
};

template<class T>
struct nonconst_ref<const T &> {
  using type = T &;
};

template<class T>
using nonconst_ref_t = typename nonconst_ref<T>::type;

template<class T>
constexpr bool mpi_accessor = false;
template<data::layout L, class T, Privileges P>
constexpr bool mpi_accessor<data::accessor<L, T, P>> = !data::portable_v<T>;

// Construct a tuple of converted arguments (or references to existing
// arguments where possible).  Note that is_constructible_v<const
// float&,const double&> is true, so we have to check
// is_constructible_v<float&,double&> instead.
template<bool M, class... PP, class... AA>
std::conditional_t<M,
  std::tuple<PP...>,
  std::tuple<std::conditional_t<
    std::is_constructible_v<nonconst_ref_t<PP> &, nonconst_ref_t<AA>>,
    const PP &,
    std::decay_t<PP>>...>>
make_parameters(std::tuple<PP...> * /* to deduce PP */, AA &&... aa) {
  if constexpr(!M) {
    static_assert((std::is_const_v<std::remove_reference_t<const PP>> && ...),
      "only MPI tasks can accept non-const references");
    static_assert((!mpi_accessor<std::decay_t<PP>> && ...),
      "only MPI tasks can accept accessors for non-portable fields");
  }
  return {exec::replace_argument<PP>(std::forward<AA>(aa))...};
}

template<bool M, class P, class... AA>
auto
make_parameters(AA &&... aa) {
  return make_parameters<M>(static_cast<P *>(nullptr), std::forward<AA>(aa)...);
}
} // namespace detail

template<auto & F, class Reduction, TaskAttributes Attributes, typename... Args>
auto
reduce_internal(Args &&... args) {
  using namespace Legion;
  using traits_t = util::function_traits<decltype(F)>;
  using return_t = typename traits_t::return_type;
  using param_tuple = typename traits_t::arguments_type;

  // This will guard the entire method
  flog::devel_guard guard(execution_tag);

  // Get the FleCSI runtime context
  auto & flecsi_context = run::context::instance();

  // Get the processor type.
  constexpr auto processor_type = mask_to_processor_type(Attributes);

  // Get the Legion runtime and context from the current task.
  auto legion_runtime = Legion::Runtime::get_runtime();
  auto legion_context = Legion::Runtime::get_context();

  constexpr bool mpi_task = processor_type == task_processor_type_t::mpi;
  static_assert(processor_type == task_processor_type_t::toc ||
                  processor_type == task_processor_type_t::loc ||
                  processor_type == task_processor_type_t::omp || mpi_task,
    "Unknown launch type");
  const auto domain_size =
    launch_size<Attributes, param_tuple>(std::forward<Args>(args)...);

  auto params =
    detail::make_parameters<mpi_task, param_tuple>(std::forward<Args>(args)...);
  prolog<mask_to_processor_type(Attributes)> pro(params, args...);

  std::vector<std::byte> buf;
  if constexpr(mpi_task) {
    // MPI tasks must be invoked collectively from one task on each rank.
    // We therefore can transmit merely a pointer to a tuple of the arguments.
    // The TaskArgument must be identical on every shard, so use the context.
    flecsi_context.mpi_params = &params;
  }
  else {
    buf = std::apply(
      [](const auto &... pp) { return util::serial::put_tuple(pp...); },
      params);
  }

  using wrap = leg::task_wrapper<F, processor_type>;
  // Replace the MPI "processor type" with an actual flag:
  const auto task = leg::task_id<wrap::execute,
    Attributes & ~mpi | as_mask(wrap::LegionProcessor)>;

  const auto add = [&](auto & l) {
    for(auto & req : pro.region_requirements())
      l.add_region_requirement(req);
    l.futures = std::move(pro).futures();
    switch(processor_type) {
      case task_processor_type_t::toc:
        l.tag = run::mapper::prefer_gpu;
        break;
      case task_processor_type_t::omp:
        l.tag = run::mapper::prefer_omp;
        break;
      // Null default is added to suppress warning for other enumerators that
      // do nothing
      default:
        break;
    }
  };

  if constexpr(std::is_same_v<decltype(domain_size), const std::monostate>) {
    {
      flog::devel_guard guard(execution_tag);
      flog_devel(info) << "Executing single task" << std::endl;
    }

    TaskLauncher launcher(task, TaskArgument(buf.data(), buf.size()));
    add(launcher);

    return future<return_t>{
      legion_runtime->execute_task(legion_context, launcher)};
  }
  else {
    {
      flog::devel_guard guard(execution_tag);
      flog_devel(info) << "Executing index task" << std::endl;
    }

    LegionRuntime::Arrays::Rect<1> launch_bounds(
      LegionRuntime::Arrays::Point<1>(0),
      LegionRuntime::Arrays::Point<1>(domain_size - 1));
    Domain launch_domain = Domain::from_rect<1>(launch_bounds);

    Legion::ArgumentMap arg_map;
    Legion::IndexLauncher launcher(
      task, launch_domain, TaskArgument(buf.data(), buf.size()), arg_map);
    add(launcher);
    launcher.point_futures.assign(
      pro.future_maps().begin(), pro.future_maps().end());

    if(mpi_task)
      launcher.tag = run::mapper::force_rank_match;

    if constexpr(!std::is_void_v<Reduction>) {
      flog_devel(info) << "executing reduction logic for "
                       << util::type<Reduction>() << std::endl;

      auto ret = future<return_t, launch_type_t::single>{
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
