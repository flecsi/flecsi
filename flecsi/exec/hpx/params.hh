// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_HPX_PARAMS_HH
#define FLECSI_EXEC_HPX_PARAMS_HH

#include <hpx/modules/collectives.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/serialization.hpp>

#include "flecsi/data/hpx/copy.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/future.hh"
#include "flecsi/exec/hpx/reduction_wrapper.hh"
#include "flecsi/exec/local/params.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/demangle.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace flecsi::exec {

struct task_prolog_base : local::prolog<task_prolog_base> {
  using prolog::prolog;

  template<class T>
  void broadcast(const data::local::field & f) {
    // Perform the required "ghost copy" on the host and make the
    // task currently being created depend on the results of the ghost-copy
    // operation.
    data::init_delayed_ghost_copy(
      f.storage(), f.storage(), [f](run::communicator & comm) {
        using data_type = ::hpx::serialization::serialize_buffer<T>;
        using namespace ::hpx::collectives;
        if(comm.comm().is_root()) {
          auto host_storage = f.as<T>();
          broadcast_to(comm.comm(),
            data_type(
              host_storage.data(), host_storage.size(), data_type::reference),
            comm.gen())
            .get();
        }
        else {
          auto host_storage = f.as<T, wo>();
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
  void visit(future<R> & p, future<R> & f) {
    dependencies(f.depend());
    futures.push_back(f.get_comms());
    p.silence();
  }
  template<typename R>
  void visit(future<R> & single,
    future<R, exec::launch_type_t::index> & index) {
    auto f = index.mine();
    dependencies(f);
    futures.push_back(index.get_comms());
    single = {std::move(f), nullptr};
  }

public:
  void request_comm() {
    need_comm = true;
  }

  // Delay the execution of the given task until all dependencies have been
  // satisfied (if any).
  template<typename R, typename Task>
  std::pair<::hpx::shared_future<R>, run::comms::ptr>
  delay_execution(std::string task_name, Task && task) && {
    auto comms = run::comms::make();
    // First dependency has lowest priority:
    comms->depend(run::context::instance().world_comms);
    auto future = data::hold::make(comms);
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
    for(auto & f : futures)
      comms->depend(std::move(f));
    auto f = ::hpx::dataflow(
      [out = run::context::instance().outstanding(),
        regions_partitions = detach(),
        task = std::forward<Task>(task),
        task_name = std::move(task_name),
        comm = need_comm ? &comms->get() : nullptr,
        // Keep comm alive if the hold can't:
        own = need_comm && read.empty() && write.empty() ? comms : nullptr](
        data::dependencies::type deps) mutable {
        const auto done = out(); // HPX doesn't destroy functors promptly
        // manage task_local variables for this task
        run::task_local_base::guard tlg;

        // annotate new HPX thread
        ::hpx::scoped_annotation _(task_name);

        // rethrow exceptions propagated from dependencies
        for(auto && f : std::forward<decltype(deps)>(deps))
          f.get();

        return task(regions_partitions, comm);
      },
      dependencies.detach())
               .share();
    // Publish to the fields used.  There is no race with the task, since
    // tasks never access any field futures.
    future.send(f);
    return {std::move(f), std::move(comms)};
  }

private:
  // The futures that represent the dependencies of the current task on its
  // arguments
  data::dependencies dependencies;
  // Dependencies on fields are computed after scheduling ghost copies.
  std::vector<data::backend_storage *> read, write;
  std::vector<run::comms::ptr> futures;
  bool need_comm = false;
};

template<processor>
using task_prolog = task_prolog_base;

/*!
  The bind_accessors type is called to walk the user task arguments inside of an
  executing HPX task to properly complete the users accessors, i.e., by pointing
  the accessor \em view instances to the appropriate buffers.

  This is the other half of the wire protocol implemented by \c task_prolog.
 */
template<processor Proc>
struct bind_accessors : local::bind<bind_accessors<Proc>, Proc> {
  explicit bind_accessors(run::communicator * comm,
    data::local::storages & regions_partitions)
    : bind_accessors::bind(regions_partitions), comm(comm) {}

  template<typename R, typename T>
  void reduce(data::local::field f) {
    reductions.push_back([f = std::move(f)](run::communicator & comm) {
      using data_type = ::hpx::serialization::serialize_buffer<T>;
      using namespace ::hpx::collectives;
      const auto host_s = f.as<T, rw>();
      auto fut = all_reduce(comm.comm(),
        data_type(host_s.data(), host_s.size()),
        exec::fold::wrap<R>{},
        comm.gen());

      return fut.then(::hpx::launch::sync, [host_s](auto && fut) {
        auto && data = fut.get();
        flog_assert(data.size() == host_s.size(),
          "received size of data must be the same as the storage size");
        std::move(data.begin(), data.begin() + data.size(), host_s.data());
      });
    });
  }

  ~bind_accessors() {
    flog_assert(reductions.empty() || comm, "no communicator for reductions");
    std::vector<data::fate> requests;
    requests.reserve(reductions.size());
    for(auto & f : reductions) {
      requests.push_back(data::fate::make(f(*comm)));
    }
  }

private:
  run::communicator * comm;
  std::vector<std::function<::hpx::future<void>(run::communicator &)>>
    reductions;
};

} // namespace flecsi::exec

#endif
