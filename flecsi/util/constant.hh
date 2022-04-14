// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_CONSTANT_HH
#define FLECSI_UTIL_CONSTANT_HH

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

/// \cond core
namespace flecsi {
namespace util {
/// \addtogroup utils
/// \{

template<class...>
struct types {};

template<auto Value>
struct constant { // like std::integral_constant, but better
  using type = decltype(Value);
  static constexpr const auto & value = Value;
};

namespace detail {
template<auto...>
struct constant1 {};
template<auto V>
struct constant1<V> : constant<V> {};
template<auto...>
extern void * const first_constant; // undefined
template<auto V, auto... VV>
constexpr const auto & first_constant<V, VV...> = V;
} // namespace detail

// Non-type template parameters must be fixed-size through C++20, so we must
// use a type to hold an arbitrary amount of information, but there's no need
// to convert each to a type separately.
template<auto... VV>
struct constants : detail::constant1<VV...> {
  static constexpr std::size_t size = sizeof...(VV);
  static constexpr decltype(detail::first_constant<VV...>) first =
    detail::first_constant<VV...>;

private:
  template<auto V, std::size_t... II>
  static constexpr std::size_t find(std::index_sequence<II...>) {
    std::size_t ret = size;
    ((V == VV ? void(ret = II) : void()), ...);
    if(ret < size)
      return ret;
    throw; // rejected by the constexpr initialization of index
  }

public:
  // Find the index of (the last appearance of) V in the sequence VV.
  // V must be comparable to each of VV.
  template<auto V>
  static constexpr std::size_t index = find<V>(
    std::make_index_sequence<size>());
};

template<class, class>
struct key_array;

namespace detail {
template<class A, class F, auto... VV>
constexpr auto
map(A && a, F & f, util::constants<VV...> /* A::keys */) {
  return key_array<std::decay_t<decltype(f(std::forward<A>(a).front()))>,
    util::constants<VV...>>{{f(std::forward<A>(a).template get<VV>())...}};
}
} // namespace detail

// A std::array<T> indexed by the values in a constants<...>.
template<class T, class C>
struct key_array : std::array<T, C::size> {
  using keys = C;

  template<template<class> class F>
  using map_type = key_array<F<T>, C>;

  template<auto V>
  constexpr T & get() {
    return (*this)[C::template index<V>];
  }
  template<auto V>
  constexpr const T & get() const {
    return (*this)[C::template index<V>];
  }

  // Apply a function to each element.
  template<class F>
  constexpr auto map(F && f) {
    return detail::map(*this, f, keys());
  }
  template<class F>
  constexpr auto map(F && f) const {
    return detail::map(*this, f, keys());
  }
};

template<auto V, class T>
struct key_type {
  static constexpr const auto & value = V;
  using type = T;

  template<template<class> class F>
  using map = key_type<V, F<T>>;
};

/*!
  A std::tuple containing the given types and indexed by the given keys.

  \tparam VT A parameter pack of types defining a key value and a type. The
             underlying storage will use a std::tuple of the given types. These
             can be referenced using the given keys.
 */

template<class... VT>
struct key_tuple : std::tuple<typename VT::type...> {
  using keys = constants<VT::value...>;

  using Base = typename key_tuple::tuple;
  using Base::Base;

  template<template<class> class F>
  using map_type = key_tuple<typename VT::template map<F>...>;

  // std::apply doesn't natively support classes derived from std::tuple.
  // We could alternatively specialize std::tuple_size for key_tuple.
  template<class F>
  decltype(auto) apply(F && f) & {
    return std::apply(std::forward<F>(f), static_cast<Base &>(*this));
  }
  template<class F>
  decltype(auto) apply(F && f) const & {
    return std::apply(std::forward<F>(f), static_cast<const Base &>(*this));
  }
  template<class F>
  decltype(auto) apply(F && f) && {
    return std::apply(std::forward<F>(f), static_cast<Base &&>(*this));
  }

  template<auto V>
  constexpr auto & get() {
    return std::get<keys::template index<V>>(*this);
  }
  template<auto V>
  constexpr const auto & get() const {
    return std::get<keys::template index<V>>(*this);
  }
}; // struct key_tuple

/// \}
} // namespace util
} // namespace flecsi
/// \endcond

#endif
