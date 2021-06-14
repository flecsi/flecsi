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

/*! @file */

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif

#include <hpx/modules/collectives.hpp>

#include "flecsi/exec/buffers.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/exec/prolog.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/function_traits.hh"

#include <type_traits>
#include <utility> // forward

namespace flecsi {
namespace exec {
namespace detail {

// AA is what the user gives us when calling execute(), PP is what
// the user defined function/task expects. PP may not be the same
// as AA, for example, user pass a field_reference as an argument
// to execute() but the task expects an data accessor as its formal
// parameter. In this case replace_argument replaces a
// field_reference with an accessor. This is done through various
// specialization of the exec::detail::task_param<> template.
template<class... PP, class... AA>
auto
replace_arguments(std::tuple<PP...> * /* to deduce PP */, AA &&... aa) {
  // Specify the template arguments explicitly to produce references to
  // unchanged arguments.
  return std::tuple<decltype(exec::replace_argument<PP>(std::forward<AA>(
    aa)))...>(exec::replace_argument<PP>(std::forward<AA>(aa))...);
}

template<typename R>
struct reduction_helper {

  template<typename T>
  auto operator()(T const & lhs, T const & rhs) const {
    return R::combine(lhs, rhs);
  }
};

} // namespace detail

template<auto & F, class Reduction, TaskAttributes Attributes, typename... Args>
auto
reduce_internal(Args &&... args) {
  using Traits = util::function_traits<decltype(F)>;
  using R = typename Traits::return_type;

  // replace arguments in args, for example, field_reference -> accessor.
  auto params = exec::detail::replace_arguments(
    static_cast<typename Traits::arguments_type *>(nullptr),
    std::forward<Args>(args)...);

  auto task_name = util::symbol<F>();

  // Now we have accessors, we need to bind the accessor to real memory
  // for the data field. We also need to patch up default conversion
  // from args to params, especially for the future<>.
  prolog _(params, args...);

  // Different kinds of task invocation with flecsi::execute():
  // 1. domain_size is a std::monostate: a single task launch. On a single
  //    rank (0) of our choice, we apply F to ARGS. Given the return type R of
  //    F, we return a future<R, single>. The client can later call
  //    future<>.get() on any rank to get the result. The value returned by
  //    .get should be the same on all ranks, implying a Broadcast is needed.
  // 2. Reduction is not void: a true reduction task.
  //    We apply F to ARGS and reduce the return values with the Reduction
  //    operation. We then put the reduced value into a future<R,
  //    launch_type::single> and return it. Again, the client could call
  //    .get() to get the (same) result on any rank, implying either an
  //    Allreduce or Reduce/Broadcast is needed.
  // 3. Reduction is void: an index launch. We apply F to ARGS on every rank,
  //    each will return a value r_i. The value r_i will be put into the the
  //    slot i in the future<R, index>. The client can call future<R,
  //    index>.get(j) to get the return value on any rank j. This implies an
  //    Allgather is needed.
  const auto ds =
    launch_size<Attributes, decltype(params)>(std::forward<Args>(args)...);
  util::annotation::rguard<util::annotation::execute_task_user> ann{task_name};
  if constexpr(std::is_same_v<decltype(ds), const std::monostate>) {
    const bool root = flecsi::run::context::instance().process() == 0;

    // single launch, only invoke the user task on the Root.
    if constexpr(std::is_void_v<R>) {
      if(root) {
        auto result = _.delay_execution(
          std::move(params), std::move(task_name), [&](auto && params) {
            // void return type, just invoke, no return value to broadcast
            std::apply(F, std::move(params));
          });

        // make sure the future that is associated with the result of this
        // task is made ready once the task finishes executing
        _.attach_result(result);

        return future<void>{std::move(result)};
      }
      else {
        // see comment about param_buffers above
        auto finalize = param_buffers{params, task_name};
        return future<void>{};
      }
    }
    else {
      auto result = _.delay_execution(
        std::move(params), std::move(task_name), [&](auto && params) {
          // Broadcast the result from root to the rest of ranks return
          // future<R, launch_type::single> where clients on every rank
          // will get the same value when calling .get().
          if(root) {
            return ::hpx::collectives::broadcast_to(
              flecsi::run::context::instance().world_comm(),
              std::apply(F, std::move(params)));
          }
          else {
            return ::hpx::collectives::broadcast_from<R>(
              flecsi::run::context::instance().world_comm());
          }
        });

      // make sure the future that is associated with the result of this
      // task is made ready once the task finishes executing
      _.attach_result(result);

      return future<R>{std::move(result)};
    }
  }
  else {
    flog_assert(ds == run::context::instance().processes(),
      "HPX backend supports only per-rank index launches");

    // index launch (including "mpi task"), invoke the user task on all ranks.
    if constexpr(!std::is_void_v<Reduction>) {
      static_assert(!std::is_void_v<R>, "can not reduce results of void task");

      auto result = _.delay_execution(
        std::move(params), std::move(task_name), [&](auto && params) {
          // A real reduce operation, every rank needs to be able to access
          // the same result through future<R>::get().
          // 1. Call the F, get the local return value
          // 2. Reduce the local return values with the Reduction
          // 3. Put the reduced value in a future<R, single> (since there is
          // only one final value) and return it.
          return ::hpx::collectives::all_reduce(
            flecsi::run::context::instance().world_comm(),
            std::apply(F, std::move(params)),
            detail::reduction_helper<Reduction>{});
        });

      // make sure the future that is associated with the result of this
      // task is made ready once the task finishes executing
      _.attach_result(result);

      return future<R>{std::move(result)};
    }
    else if constexpr(!std::is_void_v<R>) {
      // There is an Allgather happening in the constructor of future<R,
      // index> where the results from ranks are redistributed such that
      // clients on every rank i can get the return value of rank j by calling
      // get(j).
      auto result = _.delay_execution(std::move(params),
        std::move(task_name),
        [&](auto && params) { return std::apply(F, std::move(params)); });

      // make sure the future that is associated with the result of this
      // task is made ready once the task finishes executing
      _.attach_result(result);

      return future<R, exec::launch_type_t::index>{std::move(result)};
    }
    else {
      // index launch of void functions, e.g. printf("hello world");
      auto result = _.delay_execution(std::move(params),
        std::move(task_name),
        [&](auto && params) { std::apply(F, std::move(params)); });

      // make sure the future that is associated with the result of this
      // task is made ready once the task finishes executing
      _.attach_result(result);

      return future<void, exec::launch_type_t::index>{std::move(result)};
    }
  }
}
} // namespace exec
} // namespace flecsi
