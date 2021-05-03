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

/*! file */

#include "flecsi/data/field_info.hh" // TopologyType
#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology_slot.hh"
#include "flecsi/util/constant.hh"

namespace flecsi {
namespace data {
template<class>
struct coloring_slot; // avoid dependency on flecsi::execute
template<class, Privileges>
struct topology_accessor; // avoid circularity via launch.hh
} // namespace data

namespace topo {
enum single_space { elements };

namespace detail {
template<template<class> class>
struct base;

inline TopologyType next_id;
} // namespace detail

// To obtain the base class without instantiating a core topology type:
template<template<class> class T>
using base_t = typename detail::base<T>::type;

#ifdef DOXYGEN
/// An example topology base that is not really implemented.
struct core_base {
  /// The type, independent of specialization, from which the corresponding
  /// core topology type is constructed.
  using coloring = std::nullptr_t;
};

/// An example core topology that is not really implemented.
/// \tparam P topology specialization, used here as a policy
template<class P>
struct core : core_base { // with_ragged<P> is often another base class
  /// Default-constructible base for topology accessors.
  template<Privileges Priv>
  struct access {
    /// \see send_tag
    template<class F>
    void send(F &&);
  };

  explicit core(coloring);

  Color colors() const;

  template<typename P::index_space>
  data::region & get_region();

  // As a special case, the global topology does not define this.
  template<typename P::index_space>
  const data::partition & get_partition(field_id_t) const;
};
template<>
struct detail::base<core> {
  using type = core_base;
};
#endif

struct specialization_base {
  // For connectivity tuples:
  template<auto V, class T>
  using from = util::key_type<V, T>;
  template<auto V, class T>
  using entity = util::key_type<V, T>;
  template<auto... V>
  using has = util::constants<V...>;
  template<auto... V>
  using to = util::constants<V...>;
  template<class... TT>
  using list = util::types<TT...>;

  // May be overridden by policy:
  using index_space = single_space;
  using index_spaces = util::constants<elements>;
  template<class B>
  using interface = B;

  specialization_base() = delete;
};
/// Convenience base class for specialization class templates.
struct help : specialization_base {}; // intervening class avoids warnings

/// CRTP base for specializations.
/// \tparam C core topology
/// \tparam D derived topology type
template<template<class> class C, class D>
struct specialization : specialization_base {
  using core = C<D>;
  using base = base_t<C>;
  // This is just core::coloring, but core is incomplete here.
  using coloring = typename base::coloring;

  // NB: nested classes would prevent template argument deduction.
  using slot = data::topology_slot<D>;
  using cslot = data::coloring_slot<D>;

  /// The topology accessor to use as a parameter to receive a \c slot.
  /// \tparam Priv the appropriate number of privileges
  template<partition_privilege_t... Priv>
  using accessor = data::topology_accessor<D, privilege_pack<Priv...>>;

  // Use functions because these are needed during non-local initialization:
  static TopologyType id() {
    static auto ret = detail::next_id++;
    return ret;
  }

  // May be overridden by policy:

  // Most compilers eagerly instantiate a deduced static member type, so we
  // have to use a function.
  static constexpr auto default_space() {
    return D::index_spaces::value;
  }
  template<auto S> // we can't use D::index_space here
  static constexpr PrivilegeCount privilege_count =
    std::is_same_v<decltype(S), typename D::index_space> ? 1 : throw;

  static void initialize(slot &, coloring const &) {}
};

#ifdef DOXYGEN
/// An example specialization that is not really implemented.
/// No member is needed in all circumstances.
/// See also the members marked for overriding in \c specialization.
struct topology : specialization<core, topology> {
  static coloring color(...); ///< for coloring_slot

  using connectivities = list<>; ///< for connect_t/connect_access
  using entity_lists = list<>; ///< for lists_t/list_access
};
#endif

} // namespace topo
} // namespace flecsi
