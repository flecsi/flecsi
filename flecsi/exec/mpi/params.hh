// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_MPI_PARAMS_HH
#define FLECSI_EXEC_MPI_PARAMS_HH

#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/exec/mpi/future.hh"
#include "flecsi/exec/mpi/reduction_wrapper.hh"
#include "flecsi/util/mpi.hh"

namespace flecsi {
namespace topo {
struct global_base;
template<class>
struct borrow_category;
} // namespace topo

namespace exec {

template<processor Proc>
struct task_prologue : prolog_base {
  data::local::storages storage;

  using prolog_base::prolog_base;

protected:
  template<typename R>
  static void visit(future<R, exec::launch_type_t::single> & single,
    const future<R, exec::launch_type_t::index> & index) {
    single = future<R>::make(index.result);
  }

  template<typename T,
    Privileges P,
    class Topo,
    typename Topo::index_space Space>
  void visit(data::accessor<data::raw, T, P> &,
    const data::field_reference<T, data::raw, Topo, Space> & ref) {
    const field_id_t f = ref.fid();
    auto & t = ref.topology();
    constexpr bool glob =
      std::is_same_v<typename Topo::base, topo::global_base>;

    if constexpr(glob) {
      if(t.template get_region<Space>()
           .template ghost<privilege_pack<get_privilege(0, P), ro>>(f)) {
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
    }
    else
      add_copy<P>(ref);

    storage.push_back(
      get_selected(t)
        ? [&]() -> auto & {
            if constexpr(glob)
              return t;
            else
              // The partition controls how much memory is allocated.
              return t.template get_partition<Space>();
          }().share()
        : nullptr);
  } // visit generic topology

  template<class R, typename T, class Topo, typename Topo::index_space Space>
  void visit(data::reduction_accessor<R, T> &,
    const data::field_reference<T, data::dense, Topo, Space> & ref) {
    static_assert(std::is_same_v<typename Topo::base, topo::global_base>);
    storage.push_back(ref.topology().share());
  }

  // epilog
  template<class A>
  void visit(data::detail::save_for_epilog &, A & a) {
    epilog_wrappers.push_back([a] { a.get_elements().set_rsz_required(true); });
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
}; // struct task_prologue

template<processor Proc>
struct bind_accessors {
  bind_accessors(data::local::storages && storage)
    : storage(std::move(storage)) {}

private:
  template<class T, privilege P = rw, class A>
  void bind(A & a) {
    flog_assert(
      index < storage.size(), "more accessors than regions/partitions");
    std::visit(
      [&](auto && s) {
        if(s) // for borrow
          a.bind(s->template get_storage<T, P, Proc>(a.field()));
      },
      storage[index++]);
  }

protected:
  static void visit(processor_space_t<Proc> & s) {
    auto & c = run::context::instance();
    s.bind(c.colors(), c.color());
  }

  template<typename T, Privileges P>
  void visit(data::accessor<data::raw, T, P> & a) {
    bind<T, privilege_merge(P)>(a);
  } // visit generic topology

  template<class R, typename T>
  void visit(data::reduction_accessor<R, T> & a) {
    bind<T>(a);
    const auto s = a.span();

    // Reset the storage to identity on all processes except 0
    if(run::context::instance().process() != 0)
      std::fill(s.begin(), s.end(), R::template identity<T>);

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

public:
  ~bind_accessors() {
    util::mpi::auto_requests r(reductions.size());
    for(auto & f : reductions) {
      f(r());
    }
  }

private:
  data::local::storages storage;
  data::local::storages::size_type index = 0;
  std::vector<std::function<void(MPI_Request *)>> reductions;
};

} // namespace exec
} // namespace flecsi

#endif
