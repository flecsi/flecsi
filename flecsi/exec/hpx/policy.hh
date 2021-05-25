// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_POLICY_HH
#define FLECSI_EXEC_HPX_POLICY_HH

#include <flecsi-config.h>

#if !defined(FLECSI_ENABLE_HPX)
#error FLECSI_ENABLE_HPX not defined! This file depends on HPX!
#endif
#if !defined(FLECSI_ENABLE_MPI)
#error FLECSI_ENABLE_MPI not defined! This file depends on MPI!
#endif

#include <hpx/modules/collectives.hpp>

#include "flecsi/exec/buffers.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/exec/hpx/reduction_wrapper.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/exec/local/bind_parameters.hh"
#include "flecsi/exec/prolog.hh"
#include "flecsi/exec/tracer.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/function_traits.hh"

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility> // forward

namespace flecsi {
namespace exec {
namespace detail {

template<class T>
constexpr bool is_accessor_v = false;

template<data::layout L, class T, Privileges P>
constexpr bool is_accessor_v<data::accessor<L, T, P>> = true;

template<class T>
std::decay_t<T>
decay_copy(T && v) {
  return std::forward<T>(v);
}

template<bool M, class P, class A>
decltype(auto)
convert_argument(A && a) {
  const auto gen = [&a]() -> decltype(auto) {
    return exec::replace_argument<P>(std::forward<A>(a));
  };
  using PD = std::decay_t<P>;
  if constexpr(std::is_same_v<std::decay_t<decltype(gen())>, PD>) {
    if constexpr(!M && !is_accessor_v<PD>) {
      // non-MPI tasks need to decay-copy their non-accessor arguments
      return decay_copy(gen());
    }
    else {
      return gen();
    }
  }
  else {
    // This backend only must perform implicit conversions early:
    return [&gen]() -> PD { return gen(); }();
  }
}

template<class T>
constexpr bool mpi_accessor = false;
template<data::layout L, class T, Privileges P>
constexpr bool mpi_accessor<data::accessor<L, T, P>> = !data::portable_v<T>;

// Construct a tuple of converted arguments (all non-references).
template<bool M, class... PP, class... AA>
auto
make_parameters(std::tuple<PP...> * /* to deduce PP */, AA &&... aa) {
  if constexpr(!M) {
    static_assert((std::is_const_v<std::remove_reference_t<const PP>> && ...),
      "only MPI tasks can accept non-const references");
    static_assert((!mpi_accessor<std::decay_t<PP>> && ...),
      "only MPI tasks can accept accessors for non-portable fields");
  }
  return std::tuple<decltype(convert_argument<M, PP>(std::forward<AA>(aa)))...>(
    convert_argument<M, PP>(std::forward<AA>(aa))...);
}

template<bool M, class P, class... AA>
auto
make_parameters(AA &&... aa) {
  return make_parameters<M>(static_cast<P *>(nullptr), std::forward<AA>(aa)...);
}

template<typename R>
struct reduction_helper {

  template<typename T>
  auto operator()(T const & lhs, T const & rhs) const {
    return R::combine(lhs, rhs);
  }
};

template<auto & F, class Reduction, TaskAttributes Attributes, typename... Args>
auto
reduce_internal(Args &&... args) {
  using Traits = util::function_traits<decltype(F)>;
  using R = typename Traits::return_type;

  // replace arguments in args, for example, field_reference -> accessor.
  constexpr auto processor_type = mask_to_processor_type(Attributes);
  constexpr bool mpi_task = processor_type == task_processor_type_t::mpi;
  static_assert(processor_type == task_processor_type_t::toc ||
                  processor_type == task_processor_type_t::loc ||
                  processor_type == task_processor_type_t::omp || mpi_task,
    "Unknown launch type");

  auto params =
    detail::make_parameters<mpi_task, typename Traits::arguments_type>(
      std::forward<Args>(args)...);

  auto task_name = util::symbol<F>();

  // Now we have accessors, we need to bind the accessor to real memory for the
  // data field. We also need to patch up default conversion from args to
  // params, especially for the future<>. This is being achieved by creating the
  // prolog<> instances below.

  // For an explanation of the different kinds of task invocation with
  // flecsi::execute() see flexci/exec/mpi/policy.hh.
  const auto ds = exec::launch_size<Attributes, decltype(params)>(args...);
  util::annotation::rguard<util::annotation::execute_task_user> ann{task_name};

  // The prolog will calculate dependencies between tasks based on the
  // attributes associated with the arguments.
  prolog<processor_type> bound_params(params, args...);

  // 'delay' wraps the given task f such that its execution can be postponed
  // until all dependencies have been satisfied. The dependencies are derived
  // from the access Attributes for each of the arguments. See
  // hpx/task_prologue.hh for more details.
  const auto delay = [&](auto && f) {
    // The apply_delayed_prolog is run after all dependencies for the embedded
    // task f have been satisfied.
    auto apply_delayed_prolog = [f = std::forward<decltype(f)>(f)](
                                  auto & regions_partitions, auto && params) {
      // The bind_parameters constructor will possibly execute additional steps
      // during destruction that require execution after the task finished
      // running (reduction operations).
      bind_parameters<processor_type> provide_storage(
        params, regions_partitions);
      return f(std::forward<decltype(params)>(params));
    };
    return std::move(bound_params)
      .template delay_execution<R>(std::move(params),
        std::move(task_name),
        std::move(apply_delayed_prolog));
  };

  constexpr auto delayed_apply = [](auto && params) {
    return std::apply(F, std::forward<decltype(params)>(params));
  };

  if constexpr(std::is_same_v<decltype(ds), const std::monostate>) {
    const bool root = flecsi::run::context::instance().process() == 0;

    // single launch, only invoke the user task on the Root.
    if constexpr(std::is_void_v<R>) {
      if(root) {
        return future<void>{delay(delayed_apply)};
      }
      else {
        return future<void>{delay([](auto &&) {})};
      }
    }
    else {
      auto comm_gen = flecsi::run::context::instance().world_comm(task_name);
      return future<R>{delay([root, comm_gen](auto && params) {
        // Broadcast the result from root to the rest of ranks return future<R,
        // launch_type::single> where clients on every rank will get the same
        // value when calling .get().
        using namespace ::hpx::collectives;
        auto & [comm, generation] = comm_gen;
        if(root) {
          return broadcast_to(comm,
            std::apply(F, std::forward<decltype(params)>(params)),
            generation_arg(generation));
        }
        else {
          return broadcast_from<R>(comm, generation_arg(generation));
        }
      })};
    }
  }
  else {
    flog_assert(ds == run::context::instance().processes(),
      "HPX backend supports only per-rank index launches");

    // index launch (including "mpi task"), invoke the user task on all ranks.
    if constexpr(!std::is_void_v<Reduction>) {
      static_assert(!std::is_void_v<R>, "can not reduce results of void task");

      auto comm_gen = flecsi::run::context::instance().world_comm(task_name);
      return future<R>{delay([comm_gen](auto && params) {
        // A real reduce operation, every rank needs to be able to access
        // the same result through future<R>::get().
        // 1. Call the F, get the local return value
        // 2. Reduce the local return values with the Reduction
        // 3. Put the reduced value in a future<R, single> (since there is
        // only one final value) and return it.
        using namespace ::hpx::collectives;
        auto & [comm, generation] = comm_gen;
        return all_reduce(comm,
          std::apply(F, std::forward<decltype(params)>(params)),
          exec::fold::wrap<Reduction>{},
          generation_arg(generation));
      })};
    }
    else if constexpr(!std::is_void_v<R>) {
      // There is an Allgather happening in the constructor of future<R, index>
      // where the results from ranks are redistributed such that clients on
      // every rank i can get the return value of rank j by calling get(j).
      return future<R, exec::launch_type_t::index>{
        delay(delayed_apply), task_name};
    }
    else {
      // index launch of void functions, e.g. printf("hello world");
      return future<void, exec::launch_type_t::index>{delay(delayed_apply)};
    }
  }
}

} // namespace detail

template<auto & F, class Reduction, TaskAttributes Attributes, typename... Args>
auto
reduce_internal(Args &&... args) {

  constexpr bool mpi_task =
    mask_to_processor_type(Attributes) == exec::task_processor_type_t::mpi;

  if constexpr(mpi_task) {
    // wait for all current tasks to finish executing before running the MPI
    // task
    flecsi::run::context::instance().termination_detection();
  }

  auto result = detail::reduce_internal<F, Reduction, Attributes>(
    std::forward<Args>(args)...);

  if constexpr(mpi_task) {
    // MPI tasks are always synchronous
    result.wait();
  }

  return result;
}

} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_POLICY_HH
