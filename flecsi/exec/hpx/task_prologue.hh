// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_TASK_PROLOGUE_HH
#define FLECSI_EXEC_HPX_TASK_PROLOGUE_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/serialization.hpp>

#include "flecsi/config.hh"
#include "flecsi/data/hpx/copy.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/hpx/bind_accessors.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/demangle.hh"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace flecsi {
namespace topo {
struct global_base;
}

namespace exec {

struct task_prologue_base {

private:
  // Return whether a given future may be used as a dependency for this task
  bool may_be_used_as_dependency(::hpx::shared_future<void> const & f) const {
    auto it = std::find_if(no_dependencies.begin(),
      no_dependencies.end(),
      [&](auto const & future) { return flecsi::detail::is_same(f, future); });
    return it == no_dependencies.end();
  }

protected:
  // Those methods are "protected" because they are *only* called by
  // flecsi::exec::prolog which inherits from task_prologue.

  // Patch up the "un-initialized" type conversion from future<R, index> to
  // future<R, single> in the generic code.
  template<typename R>
  static void visit(future<R, exec::launch_type_t::single> & single,
    future<R, exec::launch_type_t::index> & index) {
    single = index.get(flecsi::run::context::instance().color());
  }

  // visit generic topology
  template<typename T,
    Privileges P,
    typename Topo,
    typename Topo::index_space Space>
  void visit(data::accessor<data::raw, T, P> &,
    data::field_reference<T, data::raw, Topo, Space> const & ref) {

    const field_id_t f = ref.fid();
    constexpr bool glob =
      std::is_same_v<typename Topo::base, topo::global_base>;

    auto & t = ref.topology();
    auto & r_or_p = [&]() -> auto & {
      if constexpr(glob) {
        return t;
      }
      else {
        // The partition controls how much memory is allocated.
        return t.template get_partition<Space>();
      }
    }
    ();

    auto & field = r_or_p[f];

    // store associated partition for bind_accessors
    regions_partitions.push_back(r_or_p.share());

    // Note that this can, even if P is all read-only, reentrantly post a task
    // that writes to the field (because it needs to update ghost values before
    // the user task reads them).
    data::region & reg = t.template get_region<Space>();
    if constexpr(!glob) {
      // Create a new task that performs the required ghost-copy and make the
      // task currently being created depend on the results of the ghost-copy
      // operation. This happens inside ghost_copy (see the implementation
      // copy_engine::operator()()).
      reg.ghost_copy<P>(ref);
    }
    else {
      // Perform ghost copy using host side storage if required. This might copy
      // data from the device side first.
      if(reg.ghost<privilege_pack<get_privilege(0, P), ro>>(f)) {
        // Create a new task that performs the required ghost-copy and make the
        // task currently being created depend on the results of the ghost-copy
        // operation.
        auto comm_gen =
          flecsi::run::context::instance().world_comm(std::to_string(f));

        auto && delayed_ghost_copy = [r = t.share(), f, comm_gen]() mutable {
          using data_type = ::hpx::serialization::serialize_buffer<T>;
          // This is a special case of ghost_copy thus we need the storage in
          // HostSpace rather than ExecutionSpace.
          using namespace ::hpx::collectives;
          auto & [comm, generation] = comm_gen;
          if(comm.is_root()) {
            auto host_storage =
              r->template get_storage<T, task_processor_type_t::loc, ro>(f);
            broadcast_to(comm,
              data_type(
                host_storage.data(), host_storage.size(), data_type::reference),
              generation_arg(generation))
              .get();
          }
          else {
            auto host_storage =
              r->template get_storage<T, task_processor_type_t::loc, wo>(f);
            auto && data =
              broadcast_from<data_type>(comm, generation_arg(generation)).get();
            assert(data.size() == host_storage.size());
            std::move(
              data.begin(), data.begin() + data.size(), host_storage.data());
          }
        };

        data::init_delayed_ghost_copy(field, delayed_ghost_copy);
      }
    }

    // If the current argument has a valid future associated with it (that has
    // not been marked as ready yet), then that future has to be used as a
    // dependency for executing the current task, but only if it represents
    // either a "read after write" or a "write after read or write" dependency.
    // "read after read" dependencies are not considered here.
    // Note that it might derive from the ghost copies just above.
    const bool pending = field.future.valid() && !field.future.is_ready();
    if(pending &&
       (privilege_write(P) || field.dep == data::dependency::write) &&
       may_be_used_as_dependency(field.future)) {
      dependencies.push_back(field.future);
    }

    // The dependencies of this task can be either "read after write" (if this
    // task reads from a field, then any known write to the same field has to
    // finish first), and "write after read" (if this task writes to a field,
    // then all known reads from the field have to finish before the write
    // operation).
    if constexpr(privilege_write(P)) {
      // request future from promise only if required (once for all arguments)
      if(!future.valid()) {
        future = promise.get_future();
        no_dependencies.push_back(future);
      }

      // If the task writes to the current argument then we must associate the
      // future that represents the end of the task execution with the current
      // argument.
      field.future = future;
      field.dep = flecsi::data::dependency::write;
    }
    else if constexpr(privilege_read(P)) {
      // request future from promise only if required (once for all arguments)
      if(!future.valid()) {
        future = promise.get_future();
        no_dependencies.push_back(future);
      }

      // If the task reads from the current argument then we must associate
      // the future that represents the end of the task execution with the
      // current argument. We have to make sure that possibly more than one
      // read operation may have to finish before other operations are allowed
      // to go ahead.
      if(!pending) {
        // This is either the first operation using the given field or any
        // previous operations have already finished.
        field.future = future;
        field.dep = flecsi::data::dependency::read;
      }
      else if(!flecsi::detail::is_same(field.future, future)) {
        // This read operation needs to be added to the list of dependencies
        // already existing for the given field.
        field.future = ::hpx::when_all(std::move(field.future), future).share();
        no_dependencies.push_back(field.future);

        // Leave the dependency type unchanged. If it is currently a write
        // dependency, then overwriting it here would cause the next task that
        // also reads from this field only see a read dependency, leading to its
        // unsequenced execution.
        flog_assert(field.dep != flecsi::data::dependency::none,
          "field reference must represent a valid dependency");
      }
    }
  }

  // visit for reduction operation
  template<typename R,
    typename T,
    typename Topo,
    typename Topo::index_space Space>
  void visit(data::reduction_accessor<R, T> &,
    const data::field_reference<T, data::dense, Topo, Space> & ref) {
    static_assert(std::is_same_v<typename Topo::base, topo::global_base>);
    const field_id_t f = ref.fid();
    auto & r = ref.topology();

    // Reduction input must be treated as a dependency
    auto & field = r[f];

    // store associated region for bind_accessors
    regions_partitions.push_back(r.share());

    // Any possibly valid field-future must be added as a dependency for this
    // task.
    if(field.future.valid() && !field.future.is_ready() &&
       may_be_used_as_dependency(field.future)) {
      dependencies.push_back(std::move(field.future));
    }

    // request future from promise only if required (once for all arguments)
    if(!future.valid()) {
      future = promise.get_future();
      no_dependencies.push_back(future);
    }

    // Reductions write to the current argument, thus we must associate the
    // future that represents the end of the task execution with the current
    // argument.
    field.future = future;
    field.dep = flecsi::data::dependency::write;
  }

  // Make sure HPX futures will be unwrapped (i.e. extract the inner future from
  // a compound one), if needed.
  template<typename T>
  static ::hpx::future<T> && unwrap(::hpx::future<T> && f) noexcept {
    return std::move(f);
  }

  template<typename T>
  static ::hpx::future<T> unwrap(::hpx::future<::hpx::future<T>> && f) {
    return {std::move(f)};
  }

public:
  // Delay the execution of the given task until all dependencies have been
  // satisfied (if any).
  template<typename R, typename Params, typename Task>
  ::hpx::future<R>
  delay_execution(Params && params, std::string task_name, Task && task) && {
    auto f = [regions_partitions = std::move(regions_partitions),
               task = std::forward<Task>(task),
               params = std::forward<Params>(params),
               task_name = std::move(task_name)](auto && deps) mutable {
      // manage task_local variables for this task
      run::task_local_base::guard tlg;

      // annotate new HPX thread
      ::hpx::scoped_annotation _(task_name);

      // set up execution environment
      auto finalize = param_buffers(params, task_name);

      // rethrow exceptions propagated from dependencies
      for(auto && f : std::forward<decltype(deps)>(deps))
        f.get();

      // invoke actual task, 'regions_partitions' needs to outlive the task
      // execution
      return task(regions_partitions, std::move(params));
    };
    return std::move(*this).attach_dependencies(
      unwrap(dependencies.empty()
               ? ::hpx::async(std::move(f), std::move(dependencies))
               : ::hpx::dataflow(std::move(f), std::move(dependencies))));
  }

private:
  // Make sure the future that is associated with the result of this task is
  // made ready once the task finishes executing.
  template<typename R>
  ::hpx::future<R> attach_dependencies(::hpx::future<R> && f) && {
    if(future.valid()) {
      if(f.is_ready()) {
        // task has already finished running
        promise.set_value();
      }
      else {
        // attach continuation to the task that makes the future ready
        ::hpx::traits::detail::get_shared_state(f)->set_on_completed(
          [p = std::move(promise)]() mutable { p.set_value(); });
      }
    }
    return std::move(f);
  }

  // The futures that represent the dependencies of the current task on its
  // arguments
  std::vector<::hpx::shared_future<void>> dependencies;

  // The futures that should not be used as dependencies as they depend on the
  // result of this task. We have to make sure to avoid circular dependencies
  // where the future representing the result of the current task is indirectly
  // used as a dependency for the task. This may happen if several arguments to
  // this task refer to the same field.
  std::vector<::hpx::shared_future<void>> no_dependencies;

  // This future is used as a dependency for all arguments, if needed. It
  // is used to convey the availability of this task's result.
  ::hpx::promise<void> promise;
  ::hpx::shared_future<void> future;

  // collect regions and partitions each of the arguments is associated with
  std::vector<region_or_partition> regions_partitions;
};

template<task_processor_type_t ProcessorType>
using task_prologue = task_prologue_base;

} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_TASK_PROLOGUE_HH
