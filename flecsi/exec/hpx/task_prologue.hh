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

struct task_prologue_base : prolog_base {
  using prolog_base::prolog_base;

protected:
  template<typename R>
  void visit(future<R> &, future<R> & f) {
    dependencies(f.depend());
  }
  template<typename R>
  void visit(future<R, exec::launch_type_t::single> & single,
    future<R, exec::launch_type_t::index> & index) {
    auto f = index.mine();
    dependencies(f);
    single = std::move(f);
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
    if constexpr(!glob) {
      add_copy<P>(ref);
    }
    else if(t.template get_region<Space>()
              .template ghost<privilege_pack<get_privilege(0, P), ro>>(f)) {
      // Create a new task that performs the required ghost-copy and make the
      // task currently being created depend on the results of the ghost-copy
      // operation.
      data::init_delayed_ghost_copy(
        field, field, [r = t.share(), f](run::communicator & comm) mutable {
          using data_type = ::hpx::serialization::serialize_buffer<T>;
          // This is a special case of ghost_copy thus we need the storage in
          // HostSpace rather than ExecutionSpace.
          using namespace ::hpx::collectives;
          if(comm.comm().is_root()) {
            auto host_storage = r->template get_storage<T>(f);
            broadcast_to(comm.comm(),
              data_type(
                host_storage.data(), host_storage.size(), data_type::reference),
              comm.gen())
              .get();
          }
          else {
            auto host_storage = r->template get_storage<T, wo>(f);
            auto && data =
              broadcast_from<data_type>(comm.comm(), comm.gen()).get();
            assert(data.size() == host_storage.size());
            std::move(
              data.begin(), data.begin() + data.size(), host_storage.data());
          }
        });
    }

    if constexpr(privilege_write(P)) {
      write.push_back(&field);
    }
    else if constexpr(privilege_read(P)) {
      read.push_back(&field);
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

    write.push_back(&r[ref.fid()]);
    need_comm = true;
  }

  template<class A>
  void visit(data::detail::save_for_epilog &, A & a) {
    epilog_wrappers.push_back([a] { a.get_elements().set_rsz_required(true); });
  }

public:
  void request_comm() {
    need_comm = true;
  }

  // Delay the execution of the given task until all dependencies have been
  // satisfied (if any).
  template<typename R, typename Params, typename Task>
  ::hpx::shared_future<R>
  delay_execution(Params && params, std::string task_name, Task && task) && {
    data::hold future;
    if(!read.empty() || !write.empty())
      future = data::hold::make();
    for(auto r : read)
      r->do_read([&](data::hold & d) {
        dependencies(future.depend(d));
        return future;
      });
    for(auto w : write)
      w->do_write([&](std::vector<data::hold> v) {
        for(auto & r : v)
          dependencies(future.depend(std::move(r)));
        return future;
      });
    // In the rare case where we do not have anywhere to store a future, we
    // create our own single-use communicator.
    data::comms::comm own;
    if(need_comm && !future)
      own = data::comms::make_comm();
    auto f = ::hpx::dataflow(
      [out = run::context::instance().outstanding(),
        regions_partitions = std::move(regions_partitions),
        task = std::forward<Task>(task),
        params = std::forward<Params>(params),
        task_name = std::move(task_name),
        comm = need_comm && future ? &future.comm() : own.get(),
        own = std::move(own)](data::dependencies::type deps) mutable {
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
        return (void)out(), task(regions_partitions, comm, std::move(params));
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
  // The futures that represent the dependencies of the current task on its
  // arguments
  data::dependencies dependencies;
  // Dependencies on fields are computed after scheduling ghost copies.
  std::vector<data::backend_storage *> read, write;

  // collect regions and partitions each of the arguments is associated with
  std::vector<region_or_partition> regions_partitions;
  bool need_comm = false;
};

template<processor>
using task_prologue = task_prologue_base;

} // namespace exec
} // namespace flecsi

#endif // FLECSI_EXEC_HPX_TASK_PROLOGUE_HH
