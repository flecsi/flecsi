// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_TOPO_UTILITY_TYPES_HH
#define FLECSI_TOPO_UTILITY_TYPES_HH

#include "flecsi/data/field.hh"
#include "flecsi/flog.hh"
#include "flecsi/topo/index.hh"
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/mpi.hh"

#include <type_traits>

namespace flecsi {
namespace topo {
/// \addtogroup topology
/// \{
/// \cond core
using connect_field = field<util::id, data::ragged>;

// Specializations request storage for connectivity information and lists of
// special entities in a similar fashion, by providing a pack VT of
// util::key_type<...,util::constants<...>> types.

namespace detail {

template<class, class>
struct connect;
template<class P, class... VT>
struct connect<P, util::types<VT...>> {
  using type = util::key_tuple<util::key_type<VT::value,
    util::key_array<connect_field::definition<P, VT::value>,
      typename VT::type>>...>;
};

// Lists do not need to register fields on varied index spaces, so they can
// use a single type T here (which can be an accessor_member).
template<class, class>
struct lists;
template<class T, class... VT>
struct lists<T, util::types<VT...>> {
  using type = util::key_tuple<
    util::key_type<VT::value, util::key_array<T, typename VT::type>>...>;
};

template<class, Privileges>
struct key_access;
// Here VT is as constructed by 'connect' above.
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
// A parallel sparse matrix of accessors.
template<class C, Privileges Priv>
struct connect_access : key_access<C, Priv>::type {
  // Prior to C++20, accessor_member can't refer to the subobjects of a
  // connect_t, so the accessors must be initialized externally.
  template<class... VT>
  connect_access(const util::key_tuple<VT...> & c)
    : connect_access::key_tuple(
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

// Subroutines for topology accessors:
template<class F, class... VT, class C>
void
connect_send(F && f, util::key_tuple<VT...> & accs, C & flds) {
  (
    [&] {
      std::size_t i = 0;
      for(auto & a : accs.template get<VT::value>())
        f(a, [&](auto & t) {
          return flds.template get<VT::value>()[i++](t.get());
        });
    }(),
    ...);
}
template<class F, class... VT, class L, class S>
void
lists_send(F && f,
  util::key_tuple<VT...> & accs,
  L & fld,
  S && sub) // topology -> lists subtopologies structure
{
  (
    [&] {
      std::size_t i = 0;
      for(auto & a : accs.template get<VT::value>()) {
        f(a, [&](auto & t) {
          return fld(std::invoke(sub, t.get()).template get<VT::value>()[i++]);
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
  [[nodiscard]] FLECSI_INLINE_TARGET id operator++(int) & {
    id ret = *this;
    ++*this;
    return ret;
  }
  FLECSI_INLINE_TARGET
  id & operator--() & {
    --t;
    return *this;
  }
  [[nodiscard]] FLECSI_INLINE_TARGET id operator--(int) & {
    id ret = *this;
    --*this;
    return ret;
  }

  template<typename D>
  FLECSI_INLINE_TARGET id & operator+=(D d) & {
    static_assert(
      std::is_integral_v<D>, "Invalid addend type for flecsi::topo::id");
    t += d;
    return *this;
  }
  template<typename D>
  FLECSI_INLINE_TARGET id operator+(D d) const {
    id c = *this;
    return c += d;
  }
  template<typename D>
  FLECSI_INLINE_TARGET friend id operator+(D d, id i) {
    return i += d;
  }
  template<typename D>
  FLECSI_INLINE_TARGET id & operator-=(D d) & {
    static_assert(
      std::is_integral_v<D>, "Invalid subtrahend type for flecsi::topo::id");
    t -= d;
    return *this;
  }
  template<typename D>
  FLECSI_INLINE_TARGET id operator-(D d) const {
    id c = *this;
    return c -= d;
  }
  FLECSI_INLINE_TARGET
  difference_type operator-(id i) const {
    return difference_type(t) - difference_type(i.t);
  }
  template<typename D>
  friend id operator-(D d, id i) = delete;

private:
  T t;
};
/// \endcond

/// Specify an iteration over \c id objects.
/// \gpu{function}
/// \tparam S index space
/// \param c range of integers
/// \return a range of \c id\<S\> objects
template<auto S, class C>
FLECSI_INLINE_TARGET auto
make_ids(C && c) {
  return util::transform_view(
    std::forward<C>(c), [](const auto & x) { return id<S>(x); });
}

template<class T>
void
concatenate(std::vector<T> & v, Color total, MPI_Comm comm) {
  auto g = util::mpi::all_gatherv(v, comm);
  v.clear();
  v.reserve(total);
  for(auto & g1 : g)
    for(auto & t : g1)
      v.push_back(std::move(t));
}

// Describe a sorted range of integers as a union of intervals
// (a sort of run-length encoding), as to make a data::copy_plan.
template<class R>
std::vector<data::subrow>
rle(const R & r) {
  std::vector<data::subrow> ret;
  std::size_t start = 0, last = 0;
  const auto out = [&] {
    if(start != last)
      ret.emplace_back(start, last);
  };
  for(const auto i : r)
    if(i == last)
      ++last;
    else {
      out();
      start = i;
      last = i + 1;
    }
  out();
  return ret;
}

/// \}
} // namespace topo
} // namespace flecsi

#endif
