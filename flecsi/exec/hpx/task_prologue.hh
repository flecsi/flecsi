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
  auto dep() {
    return [this](auto && d) {
      dependencies(std::forward<decltype(d)>(d));
      return get_future();
    };
  }

protected:
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

        data::init_delayed_ghost_copy(field, field, delayed_ghost_copy);
      }
    }

    if constexpr(privilege_write(P)) {
      field.do_write(dep());
    }
    else if constexpr(privilege_read(P)) {
      field.do_read(dep());
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
    auto & r = ref.topology();

    // store associated region for bind_accessors
    regions_partitions.push_back(r.share());

    r[ref.fid()].do_write(dep());
  }

public:
  // Delay the execution of the given task until all dependencies have been
  // satisfied (if any).
  template<typename R, typename Params, typename Task>
  ::hpx::shared_future<R>
  delay_execution(Params && params, std::string task_name, Task && task) && {
    auto f = ::hpx::dataflow(
      [out = run::context::instance().outstanding(),
        regions_partitions = std::move(regions_partitions),
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
        return (void)out(), task(regions_partitions, std::move(params));
      },
      dependencies.detach())
               .share();
    // Publish to the fields used.  There is no race with the task, since
    // tasks never access any field futures.
    if(future)
      future.send(f);
    return f;
  }

private:
  const data::fate & get_future() {
    if(!future)
      future = data::fate::make();
    return future;
  }

  // The futures that represent the dependencies of the current task on its
  // arguments
  data::dependencies dependencies;

  // This future is used as a dependency for all arguments, if needed. It
  // is used to convey the availability of this task's result.
  data::fate future;

  // collect regions and partitions each of the arguments is associated with
  std::vector<region_or_partition> regions_partitions;
};

template<task_processor_type_t ProcessorType>
using task_prologue = task_prologue_base;

} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_TASK_PROLOGUE_HH
