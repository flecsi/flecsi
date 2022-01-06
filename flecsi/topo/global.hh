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

#include "flecsi/data/topology.hh"
#include "flecsi/topo/core.hh"

namespace flecsi {
namespace topo {
/// \addtogroup topology
/// \{

struct global_base {
  struct coloring {};
};

template<class P>
struct global_category : global_base, data::region {
  global_category(const coloring &) : region(data::make_region<P>({1, 1})) {}
};
template<>
struct detail::base<global_category> {
  using type = global_base;
};

/// \addtogroup spec
/// \{

/*!
  The \c global type allows users to register data on a
  topology with a single index, i.e., there is one instance of
  the registered field type that is visible to all colors.
  Its \c coloring type is empty and default-constructible.
 */
struct global : specialization<global_category, global> {};

/// \}
/// \}
} // namespace topo

// Defined here to avoid circularity via ragged and execute.
template<data::layout L, class T, Privileges Priv>
struct exec::detail::launch<data::accessor<L, T, Priv>,
  data::field_reference<T, L, topo::global, topo::elements>> {
  static std::
    conditional_t<privilege_write(Priv), std::monostate, std::nullptr_t>
    get(const data::field_reference<T, L, topo::global, topo::elements> &) {
    return {};
  }
};

} // namespace flecsi
