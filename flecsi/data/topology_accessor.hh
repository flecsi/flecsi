/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include "flecsi/data/privilege.hh"
#include "flecsi/exec/launch.hh"

namespace flecsi {
namespace data {
/// \addtogroup data
/// \{

/*!
  Topology accessor type. Topology accessors are defined by the interface of
  the underlying, user-defined type, i.e., unlike field accessors,
  topologies can be customized by the specialization to add types and inerfaces
  that are not part of the core FleCSI topology type. By inheriting from the
  customized topology type, we pick up these additions.

  Pass a \c topology_slot to a task that expects a \c topology_accessor.

  \tparam T topology type
  \tparam Priv privilege pack

  \note Usually accessed as \c T::accessor.
 */
template<class T, Privileges Priv>
struct topology_accessor
  : T::template interface<typename T::core::template access<Priv>>,
    bind_tag,
    send_tag {
  using core = typename T::core::template access<Priv>;
  static_assert(sizeof(typename T::template interface<core>) == sizeof(core),
    "topology interfaces may not add data members");

  explicit topology_accessor() = default;
}; // struct topology_accessor

/// \}
} // namespace data

template<class T, Privileges P>
struct exec::detail::task_param<data::topology_accessor<T, P>> {
  static auto replace(const typename T::slot &) {
    return data::topology_accessor<T, P>();
  }
};
// Defined here to avoid circularity; it matters only with a
// topology_accessor parameter.  There are no global topology accessors,
// thus no specialization for them.
template<class P, class T>
struct exec::detail::launch<P, data::topology_slot<T>> {
  static Color get(const data::topology_slot<T> & t) {
    return t.get().colors();
  }
};
} // namespace flecsi
