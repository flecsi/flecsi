// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_GLOBAL_HH
#define FLECSI_TOPO_GLOBAL_HH

#include "flecsi/data/field.hh"
#include "flecsi/data/topology.hh"
#include "flecsi/topo/core.hh"

namespace flecsi {
namespace topo {
/// \addtogroup topology
/// \{

struct global_base {
  struct coloring {
    coloring(util::id n = 1) : size(n) {}
    util::id size;
  };
};

template<class P>
struct global_category : global_base, data::region, with_cleanup {
  global_category(const coloring & c)
    : region(data::make_region<P>({1, c.size})) {}
};
template<>
struct detail::base<global_category> {
  using type = global_base;
};

/// \addtogroup spec
/// \{

/*!
  Unpartitioned topology whose fields are readable by all colors.
  Fields must be written by single tasks.
  Its \c coloring type is convertible from an integer size;
  default-constructing it produces a size of 1.
 */
struct global : specialization<global_category, global> {
  /// An \c mpi_coloring can be initialized from an integer size.
  static coloring color(util::id n) {
    return n;
  }
};

/// \}
/// \}
} // namespace topo

template<data::layout L, class T, Privileges Priv>
struct exec::detail::launch<data::accessor<L, T, Priv>,
  data::field_reference<T, L, topo::global, topo::elements>> {
  static std::
    conditional_t<privilege_write(Priv), std::monostate, std::nullptr_t>
    get(const data::field_reference<T, L, topo::global, topo::elements> &) {
    return {};
  }
};

template<class R, typename T>
struct exec::detail::launch<data::reduction_accessor<R, T>,
  data::field_reference<T, data::dense, topo::global, topo::elements>> {
  static std::nullptr_t get(const data::
      field_reference<T, data::dense, topo::global, topo::elements> &) {
    return {};
  }
};

} // namespace flecsi

#endif
