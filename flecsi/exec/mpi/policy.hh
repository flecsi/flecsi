// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_POLICY_HH
#define FLECSI_EXEC_MPI_POLICY_HH

#include "flecsi/exec/buffers.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/exec/mpi/future.hh"
#include "flecsi/exec/mpi/reduction_wrapper.hh"
#include "flecsi/exec/prolog.hh"
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
/// \ingroup execution
/// \{
template<auto & F, class Reduction, TaskAttributes Attributes, typename... Args>
auto
reduce_internal(Args &&... args) {
  using util::mpi::test;
  using Traits = util::function_t<F>;
  using R = typename Traits::return_type;

  // replace arguments in args, for example, field_reference -> accessor.
  auto params = exec::detail::replace_arguments(
    static_cast<typename Traits::arguments_type *>(nullptr),
    std::forward<Args>(args)...);

  // TIP: param_buffers is an RAII type. We create an instance and give
  // the object a reference to the parameters and name of the task. We then
  // assign it to the `finalize` variable so it is not destroyed immediately.
  // The object will be destroyed when reduce_internal() returns. The
  // ~param_buffers() will then perform the necessary clean up on the
  // parameters (mostly calling mutator.commit()).
  auto task_name = util::symbol<F>();
  auto finalize = param_buffers{params, task_name};

  // Now we have accessors, we need to bind the accessor to real memory
  // for the data field. We also need to patch up default conversion
  // from args to params, especially for the future<>. Ghost copy for
  // the fields is also done in the prolog.
  const prolog<mask_to_processor_type(Attributes)> pr(params, args...);

  run::context_t::depth_guard rg;
  run::task_local_base::guard tlg;

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
  //
  const auto ds = launch_size<Attributes, decltype(params)>(args...);
  const auto task = [&params] { return std::apply(F, std::move(params)); };
  util::annotation::rguard<util::annotation::execute_task_user> ann{task_name};
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

      // Initiate Ibroadcast to broadcast the result from root to the rest of
      // ranks
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
      flog_fatal("MPI backend supports only per-rank index launches");
    // index launch (including "mpi task"), invoke the user task on all ranks.
    if constexpr(!std::is_void_v<Reduction>) {
      static_assert(!std::is_void_v<R>, "can not reduce results of void task");

      // A real reduce operation, every rank needs to be able to access the
      // same result through future<R>::get().
      // 1. Call the F, get the local return value
      auto ret = future<R>::make(task, true);

      // 2. Reduce the local return values with the Reduction (using its
      // corresponding MPI_Op created by register_reduction<>()).
      test(MPI_Iallreduce(MPI_IN_PLACE,
        ret->data(),
        1,
        flecsi::util::mpi::type<R>(),
        flecsi::exec::fold::wrap<Reduction, R>::op,
        MPI_COMM_WORLD,
        ret->request()));

      return ret;
    }
    else if constexpr(!std::is_void_v<R>)
      // There is an Allgather happening in the constructor of future<R, index>
      // where the results from ranks are redistributed such that clients on
      // every rank i can get the return value of rank j by calling get(j).
      return future<R, exec::launch_type_t::index>{task()};
    else {
      // index launch of void functions, e.g. printf("hello world");
      task();
      return future<void, exec::launch_type_t::index>{};
    }
  }
}

/// \}
} // namespace exec
} // namespace flecsi

#endif
