// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_POLICY_HH
#define FLECSI_EXEC_HPX_POLICY_HH

#include <hpx/modules/collectives.hpp>

#include "flecsi/config.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/exec/hpx/reduction_wrapper.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/exec/params.hh"
#include "flecsi/exec/tracer.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/function_traits.hh"

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility> // forward
#include <variant> // monostate, etc.

namespace flecsi {
namespace exec {
namespace detail {
template<class T>
struct reset_guard { // Clang 17.0.6 warns about this as a local class
  ~reset_guard() {
    t.reset();
  }
  T & t;
};
} // namespace detail

template<auto & F, class Reduction, TaskAttributes Attributes, typename... Args>
auto
reduce_internal(Args &&... args) {
  using Traits = util::function_t<F>;
  using R = typename Traits::return_type;
  using P = typename Traits::arguments_type;

  constexpr auto processor_type = mask_to_processor_type(Attributes);
  static constexpr bool mpi_task = processor_type == processor::mpi;
  static_assert(processor_type == processor::toc ||
                  processor_type == processor::loc ||
                  processor_type == processor::omp || mpi_task,
    "Unknown launch type");

  // replace arguments in args, for example, field_reference -> accessor.
  auto params = exec::make_parameters<mpi_task, P>(std::forward<Args>(args)...);

  auto task_name = util::symbol<F>();

  // Now we have accessors, we need to bind the accessor to real memory for the
  // data field. We also need to patch up default conversion from args to
  // params, especially for the future<>. This is being achieved by creating the
  // prolog<> instances below.

  const auto ds = exec::launch_size<Attributes, P>(args...);
  static constexpr bool single =
    std::is_same_v<decltype(ds), const std::monostate>;
  util::annotation::rguard<util::annotation::execute_task_user> ann{task_name};

  // The prolog will calculate dependencies between tasks based on the
  // attributes associated with the arguments.
  prolog<processor_type> bound_params(params, args...);

  // Drain all current tasks before scheduling a flecsi::mpi task (the prolog
  // handling may schedule additional tasks, like ghost-copy operations that
  // should finish running as well).
  if constexpr(mpi_task) {
    flecsi::run::context::instance().termination_detection();
  }

  const auto delay = [&](auto && f) {
    static constexpr bool need_comm =
      !std::is_invocable_v<decltype(f), decltype(params) &&>;
    // The apply_delayed_prolog is run after all dependencies for the embedded
    // task f have been satisfied.
    auto apply_delayed_prolog =
      [f = std::forward<decltype(f)>(f),
        params = std::optional(std::forward<decltype(params)>(params))](
        auto & regions_partitions, run::communicator * comm) mutable noexcept {
        // Destroy parameters (especially mutators) deterministically:
        detail::reset_guard<decltype(params)> g{params};

        // The bind_parameters constructor will possibly schedule additional
        // steps to run during destruction that require execution after the task
        // finished running (reduction operations).
        bind_parameters<processor_type> provide_storage(
          *params, comm, regions_partitions);

        if(mpi_task) // after possibly creating a communicator
          ::hpx::distributed::barrier::synchronize();
        if constexpr(need_comm)
          return std::forward<decltype(f)>(f)(*comm, std::move(*params));
        else
          return std::forward<decltype(f)>(f)(std::move(*params));
      };
    if(need_comm)
      bound_params.request_comm();
    auto ret = std::make_from_tuple<future<std::remove_cv_t<R>,
      std::is_void_v<Reduction> && !single ? launch_type_t::index
                                           : launch_type_t::single>>(
      std::move(bound_params)
        .template delay_execution<R>(
          util::symbol<F>(), std::move(apply_delayed_prolog)));
    if(mpi_task)
      ret.wait();
    return ret;
  };

  constexpr auto delayed_apply = [](auto && params) {
    return std::apply(F, std::forward<decltype(params)>(params));
  };

  if constexpr(single) {
    const bool root = flecsi::run::context::instance().process() == 0;
    if constexpr(std::is_void_v<R>) {
      if(root) {
        return delay(delayed_apply);
      }
      else {
        return delay([](auto &&) {});
      }
    }
    else {
      return delay([root](run::communicator & comm, auto && params) {
        using namespace ::hpx::collectives;
        if(root) {
          return broadcast_to(comm.comm(),
            std::apply(F, std::forward<decltype(params)>(params)),
            comm.gen())
            .get();
        }
        else {
          return broadcast_from<R>(comm.comm(), comm.gen()).get();
        }
      });
    }
  }
  else {
    flog_assert(ds == run::context::instance().processes(),
      "HPX backend supports only per-process index launches");

    if constexpr(!std::is_void_v<Reduction>) {
      static_assert(!std::is_void_v<R>, "can not reduce results of void task");

      return delay([](run::communicator & comm, auto && params) {
        using namespace ::hpx::collectives;
        return all_reduce(comm.comm(),
          std::apply(F, std::forward<decltype(params)>(params)),
          exec::fold::wrap<Reduction>{},
          comm.gen())
          .get();
      });
    }
    else
      return delay(delayed_apply);
  }
}

} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_POLICY_HH
