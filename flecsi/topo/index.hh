// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_INDEX_HH
#define FLECSI_TOPO_INDEX_HH

#include "flecsi/data/accessor.hh"
#include "flecsi/exec/launch.hh"
#include "flecsi/topo/size.hh"

namespace flecsi {
namespace topo {
/// \addtogroup topology
/// \{

namespace zero {
inline std::size_t
function(std::size_t) {
  return 0;
}
inline constexpr auto partial = make_partial<function>();
} // namespace zero

/// A partition with a field for dynamically resizing it.
struct repartition : with_size, data::prefixes, with_cleanup {
  // Construct a partition with an initial size.
  // f is passed as a task argument, so it must be serializable;
  // consider using make_partial.
  template<class F = decltype((zero::partial))>
  repartition(data::region & r, F && f = zero::partial)
    : with_size(r.size().first), prefixes(r, sizes().use([&f](auto ref) {
        execute<fill<std::decay_t<F>>>(ref, std::forward<F>(f));
      })) {}

  /// Apply sizes stored in the field.
  void resize() {
    update(sizes());
  }

  template<class F>
  void resize(F f) {
    const auto r = this->sizes();
    flecsi::execute<repartition::fill<F>>(r, f);
    this->resize();
  }

  template<auto>
  repartition & get_partition() {
    return *this;
  }

protected:
  template<Privileges Priv>
  struct access {
    template<class F>
    void send(F && f) {
      size_.topology_send(
        f, [](auto & a) -> auto & { return a.get_sizes(); });
    }

    auto & size() const {
      return size_;
    }

  private:
    data::scalar_access<topo::resize::field, Priv> size_;
  };

private:
  auto & get_sizes() {
    return sz;
  }

  template<class F>
  static void fill(resize::Field::accessor<wo> a, F f) {
    a = std::move(f)(run::context::instance().color());
  }
};

using repartitioned = data::partitioned<repartition>;

template<class T, typename T::index_space S = T::default_space(), class F>
repartitioned
make_repartitioned(Color r, F f) {
  return {data::make_region<T, S>({r, data::logical_size}), std::move(f)};
}

// Stores the flattened elements of the ragged fields on an index space.
struct ragged_partition_base : repartition {
  using coloring = data::region &;
  static constexpr single_space space = elements; // for run::context

  ragged_partition_base(coloring c) : repartition(c), reg(&c) {}

  template<single_space>
  data::region & get_region() const {
    return *reg;
  }

  // Ragged ghost copies must be handled at the level of the host topology.
  template<class R>
  void ghost_copy(const R &) {}

private:
  data::region * reg; // note the address stability assumption
};
template<class>
struct ragged_partition_category : ragged_partition_base {
  using ragged_partition_base::ragged_partition_base;

  using repartition::access;
};
template<>
struct detail::base<ragged_partition_category> {
  using type = ragged_partition_base;
};
template<PrivilegeCount N>
struct ragged_partition
  : specialization<ragged_partition_category, ragged_partition<N>> {
  template<single_space>
  static constexpr PrivilegeCount privilege_count = N;
};

template<class>
struct ragged;

namespace detail {
template<class T>
struct ragged_partitions {
  using base_type = T;
  using Partition = typename T::core;

  ragged_partitions() = default;
  // std::map doesn't propagate copyability:
  ragged_partitions(ragged_partitions &&) = default;
  ragged_partitions & operator=(ragged_partitions &&) = default;

  Partition & operator[](field_id_t i) {
    return part.at(i);
  }
  const Partition & operator[](field_id_t i) const {
    return part.at(i);
  }

protected:
  std::map<field_id_t, Partition> part;
};
} // namespace detail

template<class Topo, typename Topo::index_space S>
struct ragged_partitioned
  : data::region,
    detail::ragged_partitions<
      ragged_partition<Topo::template privilege_count<S>>> {
  explicit ragged_partitioned(Color r)
    : region({r, data::logical_size}, util::key_type<S, ragged<Topo>>()) {
    for(const auto & fi : run::context::field_info_store<ragged<Topo>, S>())
      this->part.try_emplace(fi->fid, *this);
  }
  ragged_partitioned(ragged_partitioned &&) = delete; // we store 'this'
};

namespace detail {
template<template<class P, typename P::index_space> class, class, class>
struct ragged_tuple;
template<template<class P, typename P::index_space> class R,
  class T,
  typename T::index_space... SS>
struct ragged_tuple<R, T, util::constants<SS...>> {
  using type = util::key_tuple<util::key_type<SS, R<T, SS>>...>;
};
template<template<class P, typename P::index_space> class R, class P>
using ragged_tuple_t =
  typename ragged_tuple<R, P, typename P::index_spaces>::type;
} // namespace detail

struct ragged_base {
  using coloring = std::nullptr_t;
};
template<class>
struct ragged_category : ragged_base {
  ragged_category() = delete; // used only for registering fields
};

template<class P>
struct ragged_elements {
  using index_spaces = typename P::index_spaces;
  using index_space = typename P::index_space;

  explicit ragged_elements(Color c) : ragged_elements(c, index_spaces()) {}

  template<index_space S>
  ragged_partitioned<P, S> & get() {
    return part.template get<S>();
  }

private:
  template<auto... VV>
  ragged_elements(Color n,
    util::constants<VV...> /* index_spaces, to deduce a pack */
    )
    : part((void(VV), n)...) {}

  typename detail::ragged_tuple_t<ragged_partitioned, P> part;
};

template<class T>
struct ragged : specialization<ragged_category, ragged<T>> {
  using index_space = typename T::index_space;
  using index_spaces = typename T::index_spaces;
};

// shared base, needed for metaprogramming detection of ragged fields
struct with_ragged_base {};

// Standardized interface for use by fields and accessors:
template<class P>
struct with_ragged : with_ragged_base {
  with_ragged(Color n) : ragged(n) {}

  ragged_elements<P> ragged;
};

template<>
struct detail::base<ragged_category> {
  using type = ragged_base;
};

// The user-facing variant of the color category supports ragged fields.
struct index_base : column_base {};

template<class P>
struct index_category : index_base, column<P>, with_ragged<P>, with_cleanup {
  explicit index_category(coloring c) : column<P>(c), with_ragged<P>(c) {}
};
template<>
struct detail::base<index_category> {
  using type = index_base;
};

// A subtopology for holding internal arrays without ragged support.
struct array_base {
  using coloring = std::vector<std::size_t>;

protected:
  static std::size_t index(const coloring & c, std::size_t i) {
    return c[i];
  }
};
template<class P>
struct array_category : array_base, repartitioned {
  explicit array_category(const coloring & c)
    : partitioned(make_repartitioned<P>(c.size(), make_partial<index>(c))) {}

  using repartition::access;
};

template<>
struct detail::base<array_category> {
  using type = array_base;
};

// Specializations of this template are used for distinguished entity lists.
template<class P>
struct array : topo::specialization<array_category, array<P>> {};

//---------------------------------------------------------------------------//
// User topology.
//---------------------------------------------------------------------------//

// The simplest topology that behaves as expected by application code.
struct user_base : array_base {};

/// The User topology is a bare-bones topology supporting a single
/// index space.  It implements arguably the simplest topology that
/// provides useful functionality for parallel data accesses.  The
/// User topology represents a single index space whose size can vary
/// by color.  Ghost copies are not supported.
///
/// \tparam P the specialization
///
/// The User topology's `coloring` type is a vector (specifically, a
/// ``std::vector<std::size_t>``) indicating the number of indices per
/// color in the single index space.  Because the User topology is
/// hard-wired for a single index space, specializations defining
/// their own `index_space` and `index_spaces` are not supported.
template<class P>
struct user : user_base, array_category<P>, with_ragged<P> {
  explicit user(const coloring & c)
    : user::array_category(c), user::with_ragged(c.size()) {}
};
template<>
struct detail::base<user> {
  using type = user_base;
};

// A subtopology for holding topology-specific metadata per color.
template<class P>
struct meta : specialization<user, meta<P>> {};

template<class P>
struct with_meta { // for interface consistency
  with_meta(Color n) : meta(user_base::coloring(n, 1)) {}
  typename topo::meta<P>::core meta;
};

/// \defgroup spec Predefined specializations
/// Specializations for topologies so simple that no others are needed.
/// \{

/*!
  The \c index type allows users to register data on an
  arbitrarily-sized set of indices that have an implicit one-to-one coloring.
  Its \c coloring type is just the size of that set.
 */
struct index : specialization<index_category, index> {
  static coloring color(Color size) {
    return size;
  } // color

}; // struct index
/// \}

/// \cond core

namespace detail {
// Q is the underlying topology, not to be confused with P which is borrow<Q>.
template<class Q>
struct borrow_ragged_partition {
  borrow_ragged_partition(typename Q::core &, const data::borrow &, bool) {}
};
template<class Q, bool = std::is_base_of_v<with_ragged<Q>, typename Q::core>>
struct borrow_ragged {
  borrow_ragged(typename Q::core &, const data::borrow &, bool) {}
};
template<class Q, bool = std::is_base_of_v<with_meta<Q>, typename Q::core>>
struct borrow_meta {
  borrow_meta(typename Q::core &, const data::borrow &, bool) {}
};
} // namespace detail
/// Topology-specific extension to support multi-color topology accessors.
/// Befriended by the \c borrow_category specialiation that inherits from it.
/// \tparam T core topology type
template<class T>
struct borrow_extra {
  /// Constructor invoked by \c borrow_category with its arguments.
  borrow_extra(T &, const data::borrow &, bool) {}
};
template<class P, typename P::index_space S>
struct borrow_ragged_partitions
  : detail::ragged_partitions<
      borrow<ragged_partition<P::template privilege_count<S>>>> {
  borrow_ragged_partitions(ragged_partitioned<P, S> & r,
    const data::borrow & b,
    bool f) {
    for(const auto & fi :
      run::context::instance().field_info_store<ragged<P>, S>())
      this->part.try_emplace(fi->fid, r[fi->fid], b, f);
  }
};
template<class P>
struct borrow_ragged_elements {
  borrow_ragged_elements(ragged_elements<P> & r, const data::borrow & b, bool f)
    : borrow_ragged_elements(r, b, f, typename P::index_spaces()) {}

  template<typename P::index_space S>
  borrow_ragged_partitions<P, S> & get() {
    return part.template get<S>();
  }

private:
  // TIP: std::tuple<TT...> can be initialized from const TT&... (which
  // requires copying) or from UU&&... (which cannot use list-initialization).
  template<auto... VV>
  borrow_ragged_elements(ragged_elements<P> & r,
    const data::borrow & b,
    bool f,
    util::constants<VV...> /* index_spaces, to deduce a pack */
    )
    : part(borrow_ragged_partitions<P, VV>(r.template get<VV>(), b, f)...) {}

  typename detail::ragged_tuple_t<borrow_ragged_partitions, P> part;
};
template<class>
struct borrow;

/// Specialization-independent definitions.
struct borrow_base {
  struct coloring {
    void * topo;
    const data::borrow * proj;
    bool first;
  };

  /// The core borrow topology for a subtopology.
  /// \tparam core topology type
  template<class C>
  using wrap = typename borrow<policy_t<C>>::core;

  /// Get the derived object from a \c borrow_extra specialization.
  /// \param e usually \c *this
  /// \param the \c borrow_category specialization to which \a e refers
  template<template<class> class C, class T>
  static auto & derived(borrow_extra<C<T>> & e) {
    return static_cast<typename borrow<T>::core &>(e);
  }
};

/// A selection from an underlying topology.  In general, it may have a
/// different number of colors and be partial or non-injective.  Several may
/// be used in concert for many-to-many mappings.
template<class P>
struct borrow_category : borrow_base,
                         detail::borrow_ragged_partition<typename P::Base>,
                         detail::borrow_ragged<typename P::Base>,
                         detail::borrow_meta<typename P::Base>,
                         borrow_extra<typename P::Base::core> {
  using index_space = typename P::index_space;
  using index_spaces = typename P::index_spaces;
  using Base = typename P::Base::core;

  // The underlying topology's accessor is reused, wrapped in a multiplexer
  // that corresponds to more than one instance of this class.

  explicit borrow_category(const coloring & c)
    : borrow_category(*static_cast<Base *>(c.topo), *c.proj, c.first) {}
  /// Borrow a topology.
  /// \param t underlying core topology
  /// \param b selection of colors from \a t
  /// \param f whether this is the first of a set of several borrowings used
  ///   together for many-to-many access
  borrow_category(Base & t, const data::borrow & b, bool f)
    : borrow_category::borrow_ragged_partition(t, b, f),
      borrow_category::borrow_ragged(t, b, f), borrow_category::borrow_meta(t,
                                                 b,
                                                 f),
      borrow_category::borrow_extra(t, b, f), base(&t), proj(&b), first(f) {}

  Color colors() const {
    return proj->size();
  }

  const data::borrow & get_projection() const {
    return *proj;
  }

  template<index_space S>
  data::region & get_region() {
    return base->template get_region<S>();
  }
  template<index_space S>
  auto & get_partition() {
    return base->template get_partition<S>();
  }

  template<class T, data::layout L, index_space S>
  void ghost_copy(data::field_reference<T, L, P, S> const & f) {
    // With (say) <rw,ro> privileges, each round would request a ghost copy.
    if(first)
      base->ghost_copy(
        data::field_reference<T, L, typename P::Base, S>(f.fid(), *base));
  }

private:
  friend typename borrow_category::borrow_ragged_partition;
  friend typename borrow_category::borrow_extra;

  Base * base; ///< The underlying core topology.
  const data::borrow * proj;
  bool first;
};
template<>
struct detail::base<borrow_category> {
  using type = borrow_base;
};

/// Specialization for borrowing.
/// \tparam Q underlying topology
template<class Q>
struct borrow : specialization<borrow_category, borrow<Q>> {
  using Base = Q;
  using index_space = typename Q::index_space;
  using index_spaces = typename Q::index_spaces;

  template<index_space S>
  static constexpr PrivilegeCount privilege_count =
    Q::template privilege_count<S>;

  static TopologyType id() = delete; // prevent ineffectual field registration
};

namespace detail {
template<PrivilegeCount N>
struct borrow_ragged_partition<ragged_partition<N>> {
  using Base = ragged_partition<N>;

  borrow_ragged_partition(typename Base::core & r,
    const data::borrow & b,
    bool f)
    : sz(r.sz, b, f) {}

  auto sizes() {
    return resize::field(sz);
  }
  void resize() {
    const auto & b = static_cast<typename borrow<Base>::core &>(*this);
    if(b.first)
      b.base->resize();
  }

protected:
  borrow<topo::resize>::core sz;
};
template<class Q>
struct borrow_ragged<Q, true> {
  borrow_ragged(typename Q::core & t, const data::borrow & b, bool f)
    : ragged(t.ragged, b, f) {}
  borrow_ragged_elements<Q> ragged;
};
template<class Q>
struct borrow_meta<Q, true> {
  borrow_meta(typename Q::core & t, const data::borrow & b, bool f)
    : meta(t.meta, b, f) {}
  typename borrow<topo::meta<Q>>::core meta;
};
} // namespace detail

// Common utility for borrow_extra specializations.
template<class Q>
struct borrow_sizes {
  borrow_sizes(typename Q::core & t, const data::borrow & b, bool f)
    : borrow_sizes(t, b, f, typename Q::index_spaces()) {}

  auto & get_sizes(std::size_t i) {
    return sz[i];
  }
  auto & get_sizes() {
    static_assert(Q::index_spaces::size == 1);
    return get_sizes(0);
  }

private:
  template<typename Q::index_space... SS>
  borrow_sizes(typename Q::core & t,
    const data::borrow & b,
    bool f,
    util::constants<SS...>)
    : sz{{{{t.template get_partition<SS>().sz, b, f}...}}} {}

  util::key_array<borrow<resize>::core, typename Q::index_spaces> sz;
};

// Specialization of borrow_extra for types defined above
template<class P>
struct borrow_extra<ragged_partition_category<P>> : borrow_sizes<P> {
  using borrow_extra::borrow_sizes::borrow_sizes;
};
template<class P>
struct borrow_extra<array_category<P>> : borrow_sizes<P> {
  using borrow_extra::borrow_sizes::borrow_sizes;
};

/// \endcond
/// \}
} // namespace topo

} // namespace flecsi

#endif
