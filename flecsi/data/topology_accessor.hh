// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_TOPOLOGY_ACCESSOR_HH
#define FLECSI_DATA_TOPOLOGY_ACCESSOR_HH

#include "flecsi/data/privilege.hh"
#include "flecsi/exec/launch.hh"

// 'interface' is defined as a macro on some platforms
#undef interface

namespace flecsi {
namespace topo {
template<class>
struct borrow;
}

namespace data {
/// \addtogroup data
/// \{

/*!
  Topology accessor type. Topology accessors are defined by the interface of
  the underlying, user-defined type, i.e., unlike field accessors,
  the specialization can customize topologies to add types and interfaces
  that are not part of the core FleCSI topology interface.  By inheriting from
  the customized topology interface, we pick up these additions.

  Pass a \c topology to a task that expects a \c topology_accessor.

  \tparam T specialization
  \tparam Priv privilege pack

  \note Usually accessed as \c T::accessor.
 */
template<class T, Privileges Priv>
struct topology_accessor
  : T::template interface<typename T::topology::template access<Priv>>,
    send_tag {
  using core = typename T::topology::template access<Priv>;
  static_assert(sizeof(typename T::template interface<core>) == sizeof(core),
    "topology interfaces may not add data members");

  explicit topology_accessor() = default;
}; // struct topology_accessor

/// \}
} // namespace data

namespace exec::detail {
template<class T, Privileges P>
struct task_param<data::topology_accessor<T, P>> {
  using type = data::topology_accessor<T, P>;
  static type replace(typename T::topology &) {
    return type();
  }
  static type replace(typename topo::borrow<T>::topology &) {
    return type();
  }
  static type replace(typename T::slot &) {
    return type();
  }
};
template<class P, class T>
struct launch<P, data::topology_slot<T>> {
  static Index get(const data::topology_slot<T> & t) {
    return t.get().colors();
  }
};
template<class P, class T>
struct launch<P, topology<T>> {
  static Index get(const topology<T> & t) {
    return t.colors();
  }
};
} // namespace exec::detail
} // namespace flecsi

#endif
