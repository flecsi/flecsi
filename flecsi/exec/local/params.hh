// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LOCAL_PARAMS_HH
#define FLECSI_EXEC_LOCAL_PARAMS_HH

#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology.hh"

namespace flecsi {
namespace topo {
struct global_base;
} // namespace topo

namespace exec::local {

template<class D>
struct prolog : prolog_base {
  using prolog_base::prolog_base;

  auto detach() {
    return std::move(storage);
  }

protected:
  template<typename T,
    Privileges P,
    class Topo,
    typename Topo::index_space Space>
  void visit(data::accessor<data::raw, T, P> &,
    const data::field_reference<T, data::raw, Topo, Space> & ref) {
    const field_id_t f = ref.fid();
    auto & t = ref.topology();
    auto & s = t.template get_partition<Space>();
    auto fld = s[f];

    if constexpr(std::is_same_v<typename Topo::base, topo::global_base>) {
      if(s.template ghost<privilege_pack<get_privilege(0, P), ro>>(f))
        d().template broadcast<T>(fld);
    }
    else
      add_copy<P>(ref);

    d().template raw<P>(fld.storage());
    if(get_selected(t))
      storage.push_back(std::move(fld));
    else
      storage.emplace_back();
  } // visit generic topology

  template<class R, typename T, class Topo, typename Topo::index_space Space>
  void visit(data::reduction_accessor<R, T> &,
    const data::field_reference<T, data::dense, Topo, Space> & ref) {
    static_assert(std::is_same_v<typename Topo::base, topo::global_base>);
    auto f = ref.topology()[ref.fid()];
    d().reduce(f.storage());
    storage.push_back(std::move(f));
  }

  // epilog
  template<class A>
  void visit(data::detail::save_for_epilog &, A & a) {
    epilog_wrappers.push_back([a] { a.get_elements().set_rsz_required(true); });
  }

private:
  D & d() {
    return static_cast<D &>(*this);
  }

  template<Privileges>
  static void raw(data::backend_storage &) {}
  static void reduce(data::backend_storage &) {}

  template<class P>
  static bool get_selected(const topo::borrow_category<P> & b) {
    return b.get_projection().selected();
  }
  template<class T>
  static bool get_selected(const T &) {
    return true;
  }

  data::local::storages storage;
};

template<class D, processor Proc>
struct bind {
  bind(data::local::storages & storage) : storage(storage) {}
  ~bind() {
    flog_assert(
      index == storage.size(), "more regions/partitions than accessors");
  }

private:
  template<class T, privilege P = rw, class A>
  void accessor(A & a) {
    flog_assert(
      index < storage.size(), "more accessors than regions/partitions");
    if(const auto & f = storage[index++]) // for borrow
      a.bind(f.as<T, P, Proc>());
  }

protected:
  static void visit(processor_space_t<Proc> & s) {
    auto & c = run::context::instance();
    s.bind(c.colors(), c.color());
  }

  template<typename T, Privileges P>
  void visit(data::accessor<data::raw, T, P> & a) {
    accessor<T, privilege_merge(P)>(a);
  } // visit generic topology

  template<class R, typename T>
  void visit(data::reduction_accessor<R, T> & a) {
    accessor<T>(a);
    const auto s = a.span();

    // Reset the storage to identity on all processes except 0
    if(run::context::instance().process() != 0)
      std::fill(s.begin(), s.end(), R::template identity<T>);

    static_cast<D &>(*this).template reduce<R>(s);
  }

private:
  data::local::storages & storage;
  data::local::storages::size_type index = 0;
};

} // namespace exec::local
} // namespace flecsi

#endif
