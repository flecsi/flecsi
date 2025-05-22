// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_PARAMS_HH
#define FLECSI_EXEC_MPI_PARAMS_HH

#include "flecsi/exec/local/params.hh"
#include "flecsi/exec/mpi/future.hh"
#include "flecsi/exec/mpi/reduction_wrapper.hh"
#include "flecsi/util/mpi.hh"

namespace flecsi::exec {

template<processor Proc>
struct task_prologue : local::prolog<task_prologue<Proc>> {
  using task_prologue::prolog::prolog;

  template<class T, class Topo>
  void broadcast(Topo & t, field_id_t f) {
    const auto bcast = [&](auto root) {
      // This "ghost copy" is implemented only for the host:
      auto host_storage = t->template get_storage<T, (root ? ro : wo)>(f);
      util::mpi::test(MPI_Bcast(const_cast<T *>(host_storage.data()),
        host_storage.size(),
        flecsi::util::mpi::type<T>(),
        0,
        MPI_COMM_WORLD));
    };

    if(flecsi::run::context::instance().process())
      bcast(std::false_type());
    else
      bcast(std::true_type());
  }

protected:
  using task_prologue::prolog::visit;

  template<typename R>
  static void visit(future<R, exec::launch_type_t::single> & single,
    const future<R, exec::launch_type_t::index> & index) {
    single = future<R>::make(index.result);
  }
}; // struct task_prologue

template<processor Proc>
struct bind_accessors : local::bind<bind_accessors<Proc>, Proc> {
  using bind_accessors::bind::bind;

  template<class R, typename T>
  void reduce(util::span<T> s) {
    reductions.push_back([s](MPI_Request * r) {
      util::mpi::test(MPI_Iallreduce(MPI_IN_PLACE,
        s.begin(),
        s.size(),
        flecsi::util::mpi::type<T>(),
        exec::fold::wrap<R, T>::op,
        MPI_COMM_WORLD,
        r));
    });
  }

  ~bind_accessors() {
    util::mpi::auto_requests r(reductions.size());
    for(auto & f : reductions) {
      f(r());
    }
  }

private:
  std::vector<std::function<void(MPI_Request *)>> reductions;
};

} // namespace flecsi::exec

#endif
