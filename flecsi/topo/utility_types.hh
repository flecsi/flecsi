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

/*! @file */

#if !defined(__FLECSI_PRIVATE__)
#error Do not include this file directly!
#endif

#include "flecsi/data/field.hh"
#include "flecsi/topo/index.hh" // meta_topo
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/constant.hh"

#include <type_traits>

namespace flecsi {
namespace topo {
using connect_field = field<util::id, data::ragged>;

namespace detail {
template<class, class>
struct connect;
template<class P, class... VT>
struct connect<P, util::types<VT...>> {
  using type = util::key_tuple<util::key_type<VT::value,
    util::key_array<connect_field::definition<P, VT::value>,
      typename VT::type>>...>;
};

template<class, class>
struct lists;
template<class T, class... VT>
struct lists<T, util::types<VT...>> {
  using type = util::key_tuple<
    util::key_type<VT::value, util::key_array<T, typename VT::type>>...>;
};

template<class, std::size_t>
struct key_access;
template<class... VT, std::size_t Priv>
struct key_access<util::key_tuple<VT...>, Priv> {
  using type = util::key_tuple<util::key_type<VT::value,
    util::key_array<
      typename VT::type::value_type::Field::template accessor1<Priv>,
      typename VT::type::keys>>...>;
};
} // namespace detail

// Construct a "sparse matrix" of field definitions; the row is the source
// index space (which is enough to determine the field type) and the column is
// the destination.
template<class P>
using connect_t = typename detail::connect<P, typename P::connectivities>::type;

namespace detail {
template<class C, std::size_t Priv>
using key_access_t = typename key_access<C, Priv>::type;

// A parallel sparse matrix of accessors.
template<class C, std::size_t Priv>
struct connect_access : key_access_t<C, Priv> {
  // Prior to C++20, accessor_member can't refer to the subobjects of a
  // connect_t, so the accessors must be initialized externally.
  template<class... VT>
  connect_access(const util::key_tuple<VT...> & c)
    : key_access_t<C, Priv>(
        make_from<std::decay_t<decltype(this->template get<VT::value>())>>(
          c.template get<VT::value>())...) {}

private:
  // The .get<>s here and above just access the elements in order, of course.
  template<class T, class U, auto... VV>
  static T make_from(const util::key_array<U, util::constants<VV...>> & m) {
    return {{typename T::value_type(m.template get<VV>().fid)...}};
  }
};

struct identity {
  template<class T>
  T && operator()(T && x) {
    return std::forward<T>(x);
  }
};
} // namespace detail

// Accessors for the connectivity requested by a topology.
template<class P, std::size_t Priv>
using connect_access = detail::connect_access<connect_t<P>, Priv>;

template<class T, class P>
using lists_t = typename detail::lists<T, typename P::entity_lists>::type;

// Subtopologies for the distinguished entities requested by a topology.
template<class P>
struct lists : lists_t<typename array<P>::core, P> {
  using Base = typename lists::key_tuple;

  // Initializes each subtopology to zero size on every color.
  explicit lists(std::size_t nc)
    : Base(make_base(nc, typename P::entity_lists())) {}

  // TODO: std::vector<std::vector<std::vector<std::size_t>>> for direct
  // coloring-based allocation?

private:
  template<class... VT>
  Base make_base(std::size_t nc, util::types<VT...> /* to decue a pack */) {
    return {make_base1(nc, typename VT::type())...};
  }
  template<auto... VV>
  util::key_array<typename array<P>::core, util::constants<VV...>>
  make_base1(std::size_t nc, util::constants<VV...> /* to deduce a pack */) {
    return {{(
      (void)VV, typename array<P>::core(typename array<P>::coloring(nc)))...}};
  }
};

// Accessors for the distinguished entities requested by a topology.
template<class P, std::size_t Priv>
using list_access = detail::connect_access<lists<P>, Priv>;

template<class F, class... VT, class C, class S = detail::identity>
void
connect_send(F && f,
  util::key_tuple<VT...> & ca,
  C & cf,
  S && s = {}) { // s: topology -> subtopology
  (
    [&] {
      std::size_t i = 0;
      for(auto & a : ca.template get<VT::value>())
        f(a, [&](auto & t) {
          return cf.template get<VT::value>()[i++](std::invoke(s, t.get()));
        });
    }(),
    ...);
}

template<class F, class... VT, class L, class S>
void
lists_send(F && f,
  util::key_tuple<VT...> & la,
  L & lf,
  S && s) // s: topology -> lists
{
  (
    [&] {
      std::size_t i = 0;
      for(auto & a : la.template get<VT::value>()) {
        f(a, [&](auto & t) {
          return lf(std::invoke(s, t.get()).template get<VT::value>()[i++]);
        });
      }
    }(),
    ...);
}

// A "strong typedef" for T that supports overload resolution, template
// argument deduction, and limited arithmetic.
// The first parameter differentiates topologies/index spaces.
template<auto, class T = util::id>
struct id {
  using difference_type = std::make_signed_t<T>;

  id() = default; // allow trivial default initialization
  FLECSI_INLINE_TARGET
  explicit id(T t) : t(t) {}

  FLECSI_INLINE_TARGET
  T operator+() const {
    return t;
  }

  FLECSI_INLINE_TARGET
  operator T() const {
    return t;
  }

  // Prevent assigning to transform_view results:
  id & operator=(const id &) & = default;

  FLECSI_INLINE_TARGET
  id & operator++() & {
    ++t;
    return *this;
  }
  FLECSI_INLINE_TARGET
  id operator++(int) & {
    id ret = *this;
    ++*this;
    return ret;
  }
  FLECSI_INLINE_TARGET
  id & operator--() & {
    --t;
    return *this;
  }
  FLECSI_INLINE_TARGET
  id operator--(int) & {
    id ret = *this;
    --*this;
    return ret;
  }

  FLECSI_INLINE_TARGET
  id & operator+=(difference_type d) & {
    t += d;
    return *this;
  }
  FLECSI_INLINE_TARGET
  id operator+(difference_type d) const {
    return d + *this;
  }
  FLECSI_INLINE_TARGET
  void operator+(id) const = delete;
  friend id operator+(difference_type d, id i) {
    return i += d;
  }
  FLECSI_INLINE_TARGET
  id & operator-=(difference_type d) & {
    t -= d;
    return *this;
  }
  FLECSI_INLINE_TARGET
  id operator-(difference_type d) const {
    return d - *this;
  }
  FLECSI_INLINE_TARGET
  difference_type operator-(id i) const { // also avoids ambiguity
    return difference_type(t) - difference_type(i.t);
  }

private:
  T t;
};

template<auto S, class C>
FLECSI_INLINE_TARGET auto
make_ids(C && c) { // NB: return value may be lifetime-bound to c
  return util::transform_view(
    std::forward<C>(c), [](const auto & x) { return id<S>(x); });
}

} // namespace topo
} // namespace flecsi
