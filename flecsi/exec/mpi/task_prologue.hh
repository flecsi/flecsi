// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_TASK_PROLOGUE_HH
#define FLECSI_EXEC_MPI_TASK_PROLOGUE_HH

#include "flecsi/config.hh"
#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/mpi/future.hh"
#include "flecsi/util/demangle.hh"

#include <mpi.h>

#include <memory>

namespace flecsi {
namespace topo {
struct global_base;
template<class>
struct borrow_category;
} // namespace topo

namespace exec {

template<task_processor_type_t ProcessorType>
struct task_prologue {
protected:
  // Those methods are "protected" because they are *only* called by
  // flecsi::exec::prolog() which inherits from task_prologue.

  // Patch up the "un-initialized" type conversion from future<R, index>
  // to future<R, single> in the generic code.
  template<typename R>
  static void visit(future<R, exec::launch_type_t::single> & single,
    const future<R, exec::launch_type_t::index> & index) {
    single = future<R>::make(index.result);
  }

  // Note: due to how visitor() is implemented in prolog.hh the first
  // parameter can not be 'const &' here, otherwise template/overload
  // resolution fails (silently).
  template<typename T>
  static void visit(data::detail::scalar_value<T> & s, decltype(nullptr)) {
    s.template copy<ProcessorType>();
  }

  template<typename T,
    Privileges P,
    class Topo,
    typename Topo::index_space Space>
  static void visit(data::accessor<data::raw, T, P> & accessor,
    const data::field_reference<T, data::raw, Topo, Space> & ref) {
    const field_id_t f = ref.fid();
    auto & t = ref.topology();
    data::region & reg = t.template get_region<Space>();
    constexpr bool glob =
      std::is_same_v<typename Topo::base, topo::global_base>;

    // Perform ghost copy using host side storage. This might copy data
    // from the device side first.
    if constexpr(glob) {
      if(reg.ghost<privilege_pack<get_privilege(0, P), ro>>(f)) {
        // This is a special case of ghost_copy thus we need the storage
        // in HostSpace rather than ExecutionSpace.
        auto host_storage = t->template get_storage<T>(f);
        util::mpi::test(MPI_Bcast(host_storage.data(),
          host_storage.size(),
          flecsi::util::mpi::type<T>(),
          0,
          MPI_COMM_WORLD));
      }
    }
    else
      reg.ghost_copy<P>(ref);

    if(!get_selected(t))
      return;
    // Now bind the ExecutionSpace storage to the accessor. This will also
    // trigger a host <-> device copy if needed.
    const auto storage = [&]() -> auto & {
      if constexpr(glob)
        return *t;
      else
        // The partition controls how much memory is allocated.
        return t.template get_partition<Space>();
    }
    ().template get_storage<T, ProcessorType>(f);
    accessor.bind(storage);
  } // visit generic topology

  template<class R, typename T, class Topo, typename Topo::index_space Space>
  void visit(data::reduction_accessor<R, T> & accessor,
    const data::field_reference<T, data::dense, Topo, Space> & ref) {
    static_assert(std::is_same_v<typename Topo::base, topo::global_base>);
    const field_id_t f = ref.fid();
    const auto storage =
      ref.topology()->template get_storage<T, ProcessorType>(f);
    accessor.bind(storage);

    // Reset the storage to identity on all processes except 0
    if(run::context::instance().process() != 0)
      std::fill(storage.begin(), storage.end(), R::template identity<T>);

    reductions.push_back([storage](MPI_Request * r) {
      util::mpi::test(MPI_Iallreduce(MPI_IN_PLACE,
        storage.begin(),
        storage.size(),
        flecsi::util::mpi::type<T>(),
        exec::fold::wrap<R, T>::op,
        MPI_COMM_WORLD,
        r));
    });
  }

public:
  ~task_prologue() {
    util::mpi::auto_requests r(reductions.size());
    for(auto & f : reductions) {
      f(r());
    }
  }

private:
  template<class P>
  static bool get_selected(const topo::borrow_category<P> & b) {
    return b.get_projection().selected();
  }
  template<class T>
  static bool get_selected(const T &) {
    return true;
  }

  std::vector<std::function<void(MPI_Request *)>> reductions;

}; // struct task_prologue
} // namespace exec
} // namespace flecsi

#endif
