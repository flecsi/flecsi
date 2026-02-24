// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_PARAMS_HH
#define FLECSI_EXEC_MPI_PARAMS_HH

#include "flecsi/exec/future.hh"
#include "flecsi/exec/local/params.hh"
#include "flecsi/exec/mpi/fold.hh"
#include "flecsi/util/mpi.hh"

namespace flecsi::exec {

template<processor Proc>
struct task_prolog : local::prolog<task_prolog<Proc>, Proc> {
  using task_prolog::prolog::prolog;

  template<class T>
  void broadcast(const data::local::field & f) {
    const auto bcast = [&](auto root) {
      // This "ghost copy" is implemented only for the host:
      const auto host_storage = f.as<T, (root ? ro : wo)>();
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
  using task_prolog::prolog::visit;

  template<typename R>
  static void visit(future<R> & single,
    const future<R, exec::launch_type_t::index> & index) {
    if constexpr(!std::is_void_v<R>)
      single = future<R>::make(index.result);
  }
}; // struct task_prolog

template<processor Proc>
struct bind_accessors : local::bind<bind_accessors<Proc>, Proc> {
  using bind_accessors::bind::bind;

  template<class R, typename T>
  void reduce(data::local::field f) {
    reductions.push_back([f = std::move(f)](MPI_Request * r) {
      // reductions are only implemented on the host
      const auto host_s = f.as<T, rw>();
      util::mpi::test(MPI_Iallreduce(MPI_IN_PLACE,
        host_s.data(),
        host_s.size(),
        flecsi::util::mpi::type<T>(),
        exec::fold::wrap<R, T>::op(),
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
