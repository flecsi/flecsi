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
struct repartition : with_size, data::prefixes {
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
  repartition & get_partition(field_id_t) {
    return *this;
  }

private:
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
struct ragged_partitioned : data::region {
  template<class Topo, typename Topo::index_space S>
  ragged_partitioned(Color r, util::key_type<S, Topo> kt)
    : region({r, data::logical_size}, kt) {
    for(const auto & fi :
      run::context::instance().get_field_info_store<Topo, S>())
      part.try_emplace(fi->fid, *this);
  }
  repartition & operator[](field_id_t i) {
    return part.at(i);
  }
  const repartition & operator[](field_id_t i) const {
    return part.at(i);
  }

private:
  std::map<field_id_t, repartition> part;
};

// Element storage (i.e., the concatenated rows) for ragged fields.
struct ragged_base {
  using coloring = Color;
};
template<class P>
struct ragged_category : ragged_base {
  using index_spaces = typename P::index_spaces;
  using index_space = typename P::index_space;

  ragged_category(coloring c) : ragged_category(c, index_spaces()) {}

  Color colors() const {
    return part.front().size().first;
  }

  template<index_space S>
  data::region & get_region() {
    return part.template get<S>();
  }

  template<index_space S>
  repartition & get_partition(field_id_t i) {
    return part.template get<S>()[i];
  }

  // Ragged ghost copies must be handled at the level of the host topology.
  template<class R>
  void ghost_copy(const R &) {}

private:
  template<auto... VV>
  ragged_category(Color n,
    util::constants<VV...> /* index_spaces, to deduce a pack */
    )
    : part{{ragged_partitioned(n, util::key_type<VV, P>())...}} {}

  util::key_array<ragged_partitioned, index_spaces> part;
};
template<class T>
struct ragged : specialization<ragged_category, ragged<T>> {
  using index_space = typename T::index_space;
  using index_spaces = typename T::index_spaces;

  template<index_space S>
  static constexpr PrivilegeCount privilege_count =
    T::template privilege_count<S>;
};

// shared base, needed for metaprogramming detection of ragged fields
struct with_ragged_base {};

// Standardized interface for use by fields and accessors:
template<class P>
struct with_ragged : with_ragged_base {
  with_ragged(Color n) : ragged(n) {}

  typename topo::ragged<P>::core ragged;
};

template<>
struct detail::base<ragged_category> {
  using type = ragged_base;
};

// The user-facing variant of the color category supports ragged fields.
struct index_base : column_base {};

template<class P>
struct index_category : index_base, column<P>, with_ragged<P> {
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

/// \defgroup user User
/// "Simplest possible" topology for distributed data.

// The simplest topology that behaves as expected by application code.
struct user_base : array_base {};

/// A bare-bones topology supporting a single index space.  The User
/// topology implements arguably the simplest topology that provides
/// useful functionality for parallel data accesses.  It represents a
/// single index space whose size can vary by color.  Ghost copies are
/// not supported.
///
/// \tparam P the specialization
///
/// The User topology's `coloring` type is a vector (specifically, a
/// ``std::vector<std::size_t>``) indicating the number of indices per
/// color in the single index space.  The minimum requirement for a
/// `user` specialization is a `color` method that takes zero or more
/// arguments and returns a `coloring`:
///
/// \code{.cpp}
/// struct my_spec : flecsi::topo::specialization<flecsi::topo::user, my_spec> {
///   static coloring color() {
///      ⋮
///   }
/// }
/// \endcode
///
/// Because the User topology is hard-wired for a single index space,
/// specializations defining their own `index_space` and
/// `index_spaces` are not supported.
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
  borrow_ragged(typename Q::core &, claims::core &) {}
};
template<class Q, bool = std::is_base_of_v<with_meta<Q>, typename Q::core>>
struct borrow_meta {
  borrow_meta(typename Q::core &, claims::core &) {}
};
} // namespace detail
template<class T> // core topology
struct borrow_extra {
  borrow_extra(T &, claims::core &) {}
};
template<class>
struct borrow_category;
template<class>
struct borrow;

struct borrow_base {
  struct coloring {
    void * topo;
    claims::core * clm;
  };

  template<template<class> class C, class T>
  static auto & derived(borrow_extra<C<T>> & e) {
    return static_cast<typename topo::borrow<T>::core &>(e);
  }

protected:
  struct borrow {
  private:
    // An emulation of topo::repartition with borrowed rows.
    struct repartition : data::borrow {
      // A borrowed row can be empty either because the claim was
      // claims::row({}) or because the claimed row has 0 size.
      repartition(data::region & r, topo::repartition & p, claims::core & c)
        : borrow(r, data::pointers(p, c), data::pointers::field.fid),
          sz(p.sz, c, claims::field.fid, data::completeness()),
          growth(&p.grow()) {}

      const topo::resize::policy & grow() const {
        return *growth;
      }
      auto sizes() {
        return topo::resize::field(sz);
      }
      // TODO: this shouldn't happen depth() times; note that we'd need to
      // recreate the data::borrow base if we did resize here.  There's also
      // the issue of updates to the underlying partition not propagating into
      // ours.
      void resize() {}

      indirect<topo::resize>::core sz;

    private:
      const topo::resize::policy * growth;
    };

  public:
    template<class T, typename policy_t<T>::index_space S>
    borrow(T & t, util::constant<S>, claims::core & c)
      : reg(&t.template get_region<S>()) {
      std::map<data::partition *, typename decltype(part)::size_type> seen;
      for(const auto & fi :
        run::context::instance().get_field_info_store<policy_t<T>, S>()) {
        // TODO: change all topologies(!) to provide a more precise
        // get_partition return type
        topo::repartition & p = t.template get_partition<S>(fi->fid);
        auto [it, nu] = seen.try_emplace(&p, part.size());
        if(nu)
          part.emplace_back(*reg, p, c);
        fields.emplace(fi->fid, it->second);
      }
    }
    borrow(borrow &&) = default; // std::vector doesn't propagate copyability

    repartition & operator[](field_id_t f) {
      return part[fields.at(f)];
    }
    repartition & single() {
      flog_assert(part.size() == 1, "ambiguous repartition");
      return part.front();
    }
    data::region & get_region() {
      return *reg;
    }

  private:
    data::region * reg;
    std::vector<repartition> part; // one per unique partition
    std::map<field_id_t, typename decltype(part)::size_type> fields;
  };
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

  // The underlying topology's accessor is reused, wrapped in a multiplexer
  // that corresponds to more than one instance of this class.

  explicit borrow_category(const coloring & c)
    : borrow_category(*static_cast<Base *>(c.topo), *c.clm) {}
  borrow_category(Base & t, claims::core & c)
    : borrow_category(t, c, index_spaces()) {}

  Color colors() const {
    return clm->colors();
  }

  auto get_claims() {
    return claims::field(*clm);
  }

  template<index_space S>
  data::region & get_region() {
    return get<S>().get_region();
  }
  template<index_space S>
  auto & get_partition(field_id_t f) {
    return get<S>()[f];
  }

  template<class F>
  void ghost_copy(const F & f) {
    base->ghost_copy(f);
  }

private:
  // TIP: std::tuple<TT...> can be initialized from const TT&... (which
  // requires copying) or from UU&&... (which cannot use list-initialization).
  template<auto... SS>
  borrow_category(Base & t, claims::core & c, util::constants<SS...>)
    : borrow_category::borrow_ragged(t, c), borrow_category::borrow_meta(t, c),
      borrow_category::borrow_extra(t, c),
      base(&t), spc{{borrow(t, util::constant<SS>(), c)...}}, clm(&c) {}

  template<index_space S>
  borrow & get() {
    return spc.template get<S>();
  }

  friend typename borrow_category::borrow_extra;

  Base * base;
  util::key_array<borrow, index_spaces> spc;
  claims::core * clm;
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
  borrow_ragged(typename Q::core & t, claims::core & c) : ragged(t.ragged, c) {}
  typename borrow<topo::ragged<Q>>::core ragged;
};
template<class Q>
struct borrow_meta<Q, true> {
  borrow_meta(typename Q::core & t, claims::core & c) : meta(t.meta, c) {}
  typename borrow<topo::meta<Q>>::core meta;
};
} // namespace detail

/// \}
} // namespace topo
} // namespace flecsi
