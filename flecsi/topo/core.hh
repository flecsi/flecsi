// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_CORE_HH
#define FLECSI_TOPO_CORE_HH

#include "flecsi/data/field.hh" // cleanup
#include "flecsi/data/privilege.hh"
#include "flecsi/data/topology_slot.hh"
#include "flecsi/util/constant.hh"

namespace flecsi {
namespace data {
template<class>
struct coloring_slot; // avoid dependency on flecsi::execute
template<class, Privileges>
struct topology_accessor;
} // namespace data

namespace topo {
/// \addtogroup topology
/// \{

/// The default, trivial index-space type used by specializations.
enum single_space {
  elements ///< The single index space.
};

/// \cond core
namespace detail {
template<template<class> class>
struct base;
template<class>
struct policy;
template<template<class> class C, class P>
struct policy<C<P>> {
  using type = P;
};
} // namespace detail

// To obtain the base class without instantiating a core topology type:
template<template<class> class T>
using base_t = typename detail::base<T>::type;
// Map P::core to P:
template<class T>
using policy_t = typename detail::policy<T>::type;

struct with_cleanup {
  data::cleanup cleanup;
};

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
  /// Default-constructible base for topology accessors. This struct
  /// provides the interface to the topology and can be used by a
  /// specialization developer to implement tailor-made methods
  /// needed by their applications.
  template<Privileges Priv>
  struct access {
    /// \see send_tag
    template<class F>
    void send(F &&);
  };

  /// A topology can be constructed from its \c coloring type.
  explicit core(coloring);

  /// Return the number of colors over which the topology is partitioned.
  Color colors() const;

  /// Find the region for an index space.
  template<typename P::index_space>
  data::region & get_region();

  /// Find the partition for an index space.
  /// \return a \c repartition if appropriate
  /// \note As a special case, the global topology does not define this.
  template<typename P::index_space>
  data::partition & get_partition();

  /// Prepare a ghost copy.
  /// Required only if multiple privileges are used.
  /// \tparam T data type
  /// \tparam L use to trigger special copies for dynamic fields
  /// \tparam S use to identify relevant copy plan
  /// \param f to deduce the above as well as for the field ID
  /// \return the copy plan to use, if any
  template<class T, data::layout L, typename P::index_space S>
  [[nodiscard]] const data::copy_plan * ghost_copy(
    data::field_reference<T, L, P, S> const & f);
};
/// Each core topology type must register its base type.
template<>
struct detail::base<core> {
  /// The base type.
  using type = core_base;
};
#endif
/// \endcond

/// Utilities and defaults for specializations.
struct specialization_base {
  /// \name Utilities
  /// For specifying policy information.
  /// \{

  /// A connectivity specification.
  /// \tparam V input index space
  /// \tparam T a \c to list
  template<auto V, class T>
  using from = util::key_type<V, T>;
  /// A special-entities specification.
  /// \tparam V subject index space
  /// \tparam T a \c has list
  template<auto V, class T>
  using entity = util::key_type<V, T>;
  /// A list of enumerators for which to store data.
  /// \tparam V often index spaces
  template<auto... V>
  using has = util::constants<V...>;
  /// A list of index spaces to use as output.
  /// \tparam V enumerators
  template<auto... V>
  using to = util::constants<V...>;
  /// Container.
  /// \tparam TT \c from or \c entity constructs
  template<class... TT>
  using list = util::types<TT...>;
  /// \}

  /// \name Defaults
  /// May be overridden by policy.
  /// \{

  /// The index space type.
  /// Must be valid as a template parameter type.  Typically an enumeration;
  /// the enumerator values are unimportant (so long as they are distinct).
  using index_space = single_space;
  /// The set of index spaces, wrapped in \c has.
  /// \note Other values (_e.g._, enumerators) of \c index_space are unused.
  using index_spaces = has<elements>;
  /// The topology interface type.
  /// It must be \a B or inherit from it without adding any data members.
  /// Instances of it will be value-initialized and should default-initialize
  /// \a B.
  /// \tparam B core topology interface (a specialization of \c access from
  ///   the appropriate core topology)
  template<class B>
  using interface = B;
  /// \}

  /// Specializations cannot be constructed.
  /// Use slots to create topology instances.
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
  /// The core topology base type, which can provide specialization utilities.
  using base = base_t<C>;
  // This is just core::coloring, but core is incomplete here.
  using coloring = typename base::coloring; ///< The coloring type.

  // NB: a nested class would prevent template argument deduction.

  /// The slot type for declaring topology instances.
  using slot = data::topology_slot<D>;
  /// The slot type for holding a \c coloring object.
  /// \deprecated Use \c mpi_coloring.
  using cslot [[deprecated("use mpi_coloring")]] = data::coloring_slot<D>;
  /// Constructs a \c coloring in an MPI task.
  /// \note \a D must define\code
  /// static coloring color(/* ... */);
  /// \endcode
  struct mpi_coloring {
    /// Create the coloring object.
    /// \param aa arguments to \c D::color
    template<class... AA>
    explicit mpi_coloring(AA &&... aa) {
      slot.emplace(std::forward<AA>(aa)...);
    }

    /// Get the resulting coloring.
    /// \{
    operator coloring &() {
      return slot.get();
    }
    operator const coloring &() const {
      return slot.get();
    }
    /// \}

  private:
    data::coloring_slot<D> slot;
  };

  /// The topology accessor to use as a parameter to receive a \c slot.
  /// \tparam Priv the appropriate number of privileges
  template<partition_privilege_t... Priv>
  using accessor = data::topology_accessor<D, privilege_pack<Priv...>>;

  /// \name Defaults
  /// May be overridden by policy.
  /// \{

  /// The default index space to use when one is optional.
  /// This implementation is ill-formed if there is more than one defined.
  /// \return \c index_space

  // Most compilers eagerly instantiate a deduced static member type, so we
  // have to use a function.
  static constexpr auto default_space() {
    return D::index_spaces::value;
  }
  /// The number of privileges to use for an accessor.
  /// This implementation produces 1.
  /// \tparam S index space
  template<auto S> // we can't use D::index_space here
  static constexpr PrivilegeCount privilege_count =
    std::is_same_v<decltype(S), typename D::index_space> ? 1 : throw;

  /// Specialization-specific initialization.
  /// Called by \c topology_slot::allocate; specializations may specify
  /// additional parameters to be supplied there.
  ///
  /// This implementation does nothing.
  /// \param s the slot in which the core topology has just been constructed
  static void initialize(slot & s, coloring const &) {
    (void)s;
  }
  /// \}
};

/// \}
} // namespace topo
} // namespace flecsi

#endif
