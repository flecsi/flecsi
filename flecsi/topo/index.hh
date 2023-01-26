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

// A partition with a field for dynamically resizing it.
struct repartition : with_size, data::prefixes, with_cleanup {
  // Construct a partition with an initial size.
  // f is passed as a task argument, so it must be serializable;
  // consider using make_partial.
  template<class F = decltype((zero::partial))>
  repartition(data::region & r, F && f = zero::partial)
    : with_size(r.size().first), prefixes(r, sizes().use([&f](auto ref) {
        execute<fill<std::decay_t<F>>>(ref, std::forward<F>(f));
      })) {}

  void resize() { // apply sizes stored in the field
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
    for(const auto & fi :
      run::context::instance().field_info_store<ragged<Topo>, S>())
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

namespace detail {
// Q is the underlying topology, not to be confused with P which is borrow<Q>.
template<class Q, bool = std::is_base_of_v<with_ragged<Q>, typename Q::core>>
struct borrow_ragged {
  borrow_ragged(typename Q::core &, claims::core &, bool) {}
};
template<class Q, bool = std::is_base_of_v<with_meta<Q>, typename Q::core>>
struct borrow_meta {
  borrow_meta(typename Q::core &, claims::core &, bool) {}
};
} // namespace detail
template<class T> // core topology
struct borrow_extra {
  borrow_extra(T &, claims::core &, bool) {}
};
// An emulation of topo::repartition with borrowed rows.
struct borrow_partition_base : indirect_base {
  struct coloring {
    data::region * r;
    repartition * p;
    claims::core * c;
  };

  borrow_partition_base(const coloring & c)
    : indirect_base(*c.r,
        data::pointers(*c.p, *c.c),
        data::pointers::field.fid),
      sz(c.p->sz, *c.c, claims::field.fid, data::completeness()),
      growth(&c.p->grow()) {}

  const topo::resize::policy & grow() const {
    return *growth;
  }
  template<class T, typename policy_t<T>::index_space S>
  borrow_partition_base(T & t, util::constant<S>, claims::core & c)
    // TODO: change all topologies(!) to provide a more precise
    // get_partition return type
    : borrow_partition_base(
        {&t.template get_region<S>(), &t.template get_partition<S>(), &c}) {}

  auto sizes() {
    return topo::resize::field(sz);
  }
  // TODO: this shouldn't happen depth() times; note that we'd need to
  // recreate the data::borrow base if we did resize here.  There's also
  // the issue of updates to the underlying partition not propagating into
  // ours.
  void resize() {}

  indirect<topo::resize>::core sz;

  // There is no need to use a particular index_space type; only ragged fields
  // use this topology directly.  The default is for I/O code.
  template<single_space = elements>
  data::region & get_region() {
    return indirect_base::get_region();
  }

  template<class R>
  void ghost_copy(const R &) {}

private:
  const topo::resize::policy * growth;
};
template<class>
struct borrow_partition_category : borrow_partition_base {
  using borrow_partition_base::borrow_partition_base;
};
template<>
struct detail::base<borrow_partition_category> {
  using type = borrow_partition_base;
};
template<PrivilegeCount N>
struct borrow_partition
  : specialization<borrow_partition_category, borrow_partition<N>> {
  template<single_space>
  static constexpr PrivilegeCount privilege_count = N;
};
template<class P, typename P::index_space S>
struct borrow_ragged_partitions
  : detail::ragged_partitions<
      borrow_partition<P::template privilege_count<S>>> {
  borrow_ragged_partitions(ragged_partitioned<P, S> & r, claims::core & c) {
    for(const auto & fi :
      run::context::instance().field_info_store<ragged<P>, S>())
      this->part.try_emplace(
        fi->fid, r[fi->fid], util::constant<elements>(), c);
  }
};
template<class P>
struct borrow_ragged_elements {
  borrow_ragged_elements(ragged_elements<P> & r, claims::core & c)
    : borrow_ragged_elements(r, c, typename P::index_spaces()) {}

  template<typename P::index_space S>
  borrow_ragged_partitions<P, S> & get() {
    return part.template get<S>();
  }

private:
  // TIP: std::tuple<TT...> can be initialized from const TT&... (which
  // requires copying) or from UU&&... (which cannot use list-initialization).
  template<auto... VV>
  borrow_ragged_elements(ragged_elements<P> & r,
    claims::core & c,
    util::constants<VV...> /* index_spaces, to deduce a pack */
    )
    : part(borrow_ragged_partitions<P, VV>(r.template get<VV>(), c)...) {}

  typename detail::ragged_tuple_t<borrow_ragged_partitions, P> part;
};
template<class>
struct borrow;

struct borrow_base {
  struct coloring {
    void * topo;
    claims::core * clm;
    bool first;
  };

  template<class C>
  using wrap = typename borrow<policy_t<C>>::core; // for subtopologies

  template<template<class> class C, class T>
  static auto & derived(borrow_extra<C<T>> & e) {
    return static_cast<typename borrow<T>::core &>(e);
  }
};

// A selection from an underlying topology.  In general, it may have a
// different number of colors and be partial or non-injective.  Several may be
// used in concert for many-to-many mappings.  Certain topologies too simple
// to use repartition are not supported.
template<class P>
struct borrow_category : borrow_base,
                         detail::borrow_ragged<typename P::Base>,
                         detail::borrow_meta<typename P::Base>,
                         borrow_extra<typename P::Base::core> {
  using index_space = typename P::index_space;
  using index_spaces = typename P::index_spaces;
  using Base = typename P::Base::core;
  // Using this directly avoids an irrelevant choice of privilege count.
  using Partition = borrow_partition_base;

  // The underlying topology's accessor is reused, wrapped in a multiplexer
  // that corresponds to more than one instance of this class.

  explicit borrow_category(const coloring & c)
    : borrow_category(*static_cast<Base *>(c.topo), *c.clm, c.first) {}
  borrow_category(Base & t, claims::core & c, bool f)
    : borrow_category(t, c, f, index_spaces()) {}

  Color colors() const {
    return clm->colors();
  }

  auto get_claims() {
    return claims::field(*clm);
  }

  template<index_space S>
  data::region & get_region() {
    return get_partition<S>().get_region();
  }
  template<index_space S>
  Partition & get_partition() {
    return spc.template get<S>();
  }

  template<class T, data::layout L, index_space S>
  void ghost_copy(data::field_reference<T, L, P, S> const & f) {
    if(first)
      base->ghost_copy(
        data::field_reference<T, L, typename P::Base, S>(f.fid(), *base));
  }

private:
  template<auto... SS>
  borrow_category(Base & t, claims::core & c, bool f, util::constants<SS...>)
    : borrow_category::borrow_ragged(t, c, f),
      borrow_category::borrow_meta(t, c, f), borrow_category::borrow_extra(t,
                                               c,
                                               f),
      base(&t), spc{{Partition(t, util::constant<SS>(), c)...}}, clm(&c),
      first(f) {}

  friend typename borrow_category::borrow_extra;

  Base * base;
  util::key_array<Partition, index_spaces> spc;
  claims::core * clm;
  bool first;
};
template<>
struct detail::base<borrow_category> {
  using type = borrow_base;
};

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
template<class Q>
struct borrow_ragged<Q, true> {
  borrow_ragged(typename Q::core & t, claims::core & c, bool)
    : ragged(t.ragged, c) {}
  borrow_ragged_elements<Q> ragged;
};
template<class Q>
struct borrow_meta<Q, true> {
  borrow_meta(typename Q::core & t, claims::core & c, bool f)
    : meta(t.meta, c, f) {}
  typename borrow<topo::meta<Q>>::core meta;
};
} // namespace detail

// Specialization of borrow_extra for types defined above
template<class P>
struct borrow_extra<ragged_partition_category<P>> {
  borrow_extra(ragged_partition_category<P> &, claims::core &, bool) {}

  auto & get_sizes() {
    return borrow_base::derived(*this).spc[0].sz;
  }
};

template<class P>
struct borrow_extra<array_category<P>> {
  borrow_extra(array_category<P> &, claims::core &, bool) {}

  auto & get_sizes() {
    return borrow_base::derived(*this).spc[0].sz;
  }
};

/// \}
} // namespace topo

} // namespace flecsi

#endif
