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
#include "flecsi/exec/buffers.hh"
#include "flecsi/exec/hpx/future.hh"
#include "flecsi/exec/local/params.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/demangle.hh"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace flecsi::exec {

struct task_prologue_base : local::prolog<task_prologue_base> {
  using prolog::prolog;

  template<class T, class Topo>
  void broadcast(Topo & t, field_id_t f) {
    // Perform the required "ghost copy" on the host and make the
    // task currently being created depend on the results of the ghost-copy
    // operation.
    auto & field = t[f];
    data::init_delayed_ghost_copy(
      field, field, [r = t.share(), f](run::communicator & comm) mutable {
        using data_type = ::hpx::serialization::serialize_buffer<T>;
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

  template<Privileges P>
  void raw(data::backend_storage & field) {
    if constexpr(privilege_write(P)) {
      write.push_back(&field);
    }
    else if constexpr(privilege_read(P)) {
      read.push_back(&field);
    }
  }

  void reduce(data::backend_storage & s) {
    write.push_back(&s);
    need_comm = true;
  }

protected:
  using prolog::visit;

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
        regions_partitions = detach(),
        task = std::forward<Task>(task),
        params = std::forward<Params>(params),
        task_name = std::move(task_name),
        comm = need_comm && future ? &future.comm() : own.get(),
        own = std::move(own)](data::dependencies::type deps) mutable {
        // manage task_local variables for this task
        run::task_local_base::guard tlg;

        // annotate new HPX thread
        ::hpx::scoped_annotation _(task_name);

        // regions_partitions must outlive this:
        auto finalize = param_buffers(params, task_name);

        // rethrow exceptions propagated from dependencies
        for(auto && f : std::forward<decltype(deps)>(deps))
          f.get();

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
  bool need_comm = false;
};

template<processor>
using task_prologue = task_prologue_base;

} // namespace flecsi::exec

#endif // FLECSI_EXEC_HPX_TASK_PROLOGUE_HH
