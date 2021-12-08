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

#include "flecsi/data/field.hh"
#include "flecsi/flog.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/constant.hh"

#include <type_traits>

/// \cond core
namespace flecsi {
namespace topo {
/// \addtogroup topology
/// \{
using connect_field = field<util::id, data::ragged>;

namespace detail {

template<class, class>
struct connect;

/*!
  Connectivity information for the given specialization policy \emph{P} for the
  given key_types in \emph{VT}. This data structure adds ragged fields to the
  specialized user type to store connectivity informaiton for each
  user-specified connectivity.

  @tparam P  A core topology specialization policy.
  @tparam VT A parameter pack of types defining a key value and a type.
 */

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

template<class, Privileges>
struct key_access;
template<class... VT, Privileges Priv>
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
template<class C, Privileges Priv>
using key_access_t = typename key_access<C, Priv>::type;

// A parallel sparse matrix of accessors.
template<class C, Privileges Priv>
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
} // namespace detail

// Accessors for the connectivity requested by a topology.
template<class P, Privileges Priv>
using connect_access = detail::connect_access<connect_t<P>, Priv>;

template<class T, class P>
using lists_t = typename detail::lists<T, typename P::entity_lists>::type;

// Subtopologies for the distinguished entities requested by a topology.
template<class P>
struct lists : lists_t<typename array<P>::core, P> {
  using Base = typename lists::key_tuple;

  // Initializes each subtopology to zero size on every color.
  explicit lists(Color nc) : lists(nc, typename P::entity_lists()) {}

  // TODO: std::vector<std::vector<std::vector<std::size_t>>> for direct
  // coloring-based allocation?

private:
  template<class... VT>
  lists(Color nc, util::types<VT...> /* deduce pack */)
    : Base{make_base1(nc, typename VT::type())...} {}
  template<auto... VV>
  util::key_array<typename array<P>::core, util::constants<VV...>>
  make_base1(Color nc, util::constants<VV...> /* to deduce a pack */) {
    return {{(
      (void)VV, typename array<P>::core(typename array<P>::coloring(nc)))...}};
  }
};

// Accessors for the distinguished entities requested by a topology.
template<class P, Privileges Priv>
using list_access = detail::connect_access<lists<P>, Priv>;

// Subroutines for topology accessors:
template<class F, class... VT, class C, class S = util::identity>
void connect_send(F && f,
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

  id(const id &) = default;

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
    return id(difference_type(*this) - d);
  }
  FLECSI_INLINE_TARGET
  difference_type operator-(id i) const { // also avoids ambiguity
    return difference_type(t) - difference_type(i.t);
  }

private:
  T t;
};

/// Specify an iteration over \c id objects.
/// \tparam S index space
/// \param c range of integers
/// \return a range of \c id<S> objects, perhaps lifetime-bound to \a c
template<auto S, class C>
FLECSI_INLINE_TARGET auto
make_ids(C && c) {
  return util::transform_view(
    std::forward<C>(c), [](const auto & x) { return id<S>(x); });
}

/// \}
} // namespace topo
} // namespace flecsi
/// \endcond
