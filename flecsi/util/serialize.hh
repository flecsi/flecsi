// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_SERIALIZE_HH
#define FLECSI_UTIL_SERIALIZE_HH

#include <cstddef>
#include <cstring> // memcpy
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility> // declval
#include <vector>

#include "type_traits.hh"
#include <flecsi/flog.hh>

namespace flecsi {
namespace util {
namespace serial {
/// \defgroup serial Serialization
/// Serialization without default constructibility.
/// Supported types:
/// - any default-constructible, trivially-move-assignable, non-pointer type
/// - any type that supports the Legion return-value serialization interface
/// - \c std::string
/// - any type with an appropriate specialization of \c traits or \c convert
/// - any \c std::pair, \c std::tuple, \c std::array, \c std::vector,
///   \c std::set, \c std::map, \c std::unordered_map of a supported type
/// \ingroup utils
/// \{

// Similar to that in GNU libc.  NB: memcpy has no alignment requirements.
inline void
mempcpy(std::byte *& d, const void * s, std::size_t n) {
  std::memcpy(d, s, n);
  d += n;
}
// For size precalculation:
inline void
mempcpy(std::size_t & x, const void *, std::size_t n) {
  x += n;
}

/// Extension point for serialization.
/// The primary template is not really a complete type.
/// \tparam T object type
/// \tparam E unused SFINAE hook
template<class T, class E = void>
struct traits
#ifdef DOXYGEN
{
  /// Serialize an object.
  /// \tparam P see \c serial::put
  template<class P>
  static void put(P & p, const T &);
  /// Reconstruct an object.
  static T get(const std::byte *&);
}
#endif
;

/// Store objects and advance past their serialized form.
/// \tparam P \c std::size_t for calculating sizes, or `std::byte*` for actual
///   serialization
/// \param p pointer or size
/// \param tt objects
template<class... TT, class P>
void
put(P & p, const TT &... tt) {
  (traits<std::remove_const_t<TT>>::put(p, tt), ...);
}
/// Compute the serialized size of a fixed set of objects.
template<class... TT>
std::size_t
size(const TT &... tt) {
  std::size_t ret = 0;
  put(ret, tt...);
  return ret;
}
/// Reconstruct an object, advancing past its serialized form.
template<class T>
T
get(const std::byte *& p) {
  return traits<std::remove_const_t<T>>::get(p);
}

/// Serialize into a buffer.
/// \param f a function that accepts an argument for \c put
template<class F>
std::vector<std::byte>
buffer(F && f) {
  std::size_t sz = 0;
  f(sz);
  std::vector<std::byte> ret(sz);
  auto *const p0 = ret.data(), *p = p0;
  std::forward<F>(f)(p);
  flog_assert(p == p0 + sz, "Wrong serialization size");
  return ret;
}

/// Get a single object.
/// \param p may be an rvalue
template<class T>
T
get1(const std::byte * p) {
  return get<T>(p);
}

/// Aggregate helper that converts to any type via \c get.
struct cast {
  /// The pointer from which to \c get.
  const std::byte *&p,
    /// The pointer at which to end deserialization, if known.
    *e = nullptr;
  ~cast() {
    flog_assert(!e || p == e, "Wrong deserialization size");
  }
  template<class T>
  T get() const {
    return serial::get<T>(p);
  }
  template<class T>
  operator T() const {
    return get<T>();
  }
};

/// Serialize a fixed set of objects into a buffer.
/// Reconstruct with \c get_tuple.
template<class... TT>
auto
put_tuple(const TT &... tt) {
  return buffer([&](auto & p) { put(p, tt...); });
}
/// Construct a tuple of objects.
/// \param e pointer to end of representation, if known
template<class... TT>
auto
get_tuple(const std::byte * p, const std::byte * e = nullptr) {
  cast r{p, e};
  return std::tuple{r.get<TT>()...};
}

/// Construct a \c std::vector from a size and then elements.
/// \tparam S size type
/// \warning `get<std::vector<T>>` is not guaranteed to be equivalent.
template<class T, class S = typename std::vector<T>::size_type>
auto
get_vector(const std::byte *& p) {
  auto n = get<S>(p);
  std::vector<T> ret;
  ret.reserve(n);
  while(n--)
    ret.push_back(get<T>(p));
  return ret;
}

namespace detail {
template<class T>
struct container {
  template<class P>
  static void put(P & p, const T & c) {
    serial::put(p, c.size());
    for(auto & t : c)
      serial::put(p, t);
  }
  static T get(const std::byte *& p) {
    T ret;
    for(auto n = serial::get<typename T::size_type>(p); n--;)
      ret.insert(serial::get<typename T::value_type>(p));
    return ret;
  }
};
} // namespace detail
} // namespace serial

// Unfortunately, std::tuple<int> is not trivially copyable, so check more:
template<class T>
constexpr bool bit_assignable_v =
  std::is_trivially_copy_assignable_v<T> ||
  std::is_copy_assignable_v<T> && std::is_trivially_copy_constructible_v<T>;
template<class T>
constexpr bool bit_copyable_v =
  std::is_default_constructible_v<T> && bit_assignable_v<T>;

namespace serial {
template<class T>
struct traits<T, std::enable_if_t<bit_copyable_v<T>>> {
  static_assert(!std::is_pointer_v<T>, "Cannot serialize pointers");
  template<class P>
  static void put(P & p, const T & t) {
    mempcpy(p, &t, sizeof t);
  }
  static T get(const std::byte *& p) {
    T ret;
    // Suppress -Wclass-memaccess: the default constructor needn't be trivial!
    std::memcpy(static_cast<void *>(&ret), p, sizeof ret);
    p += sizeof ret;
    return ret;
  }
};
template<class T, class U>
struct traits<std::pair<T, U>,
  std::enable_if_t<!bit_copyable_v<std::pair<T, U>>>> {
  using type = std::pair<T, U>;
  template<class P>
  static void put(P & p, const type & v) {
    serial::put<T, U>(p, v.first, v.second); // explicit T/U rejects references
  }
  static type get(const std::byte *& p) {
    return {serial::get<T>(p), serial::get<U>(p)};
  }
};
template<class... TT>
struct traits<std::tuple<TT...>,
  std::enable_if_t<!bit_copyable_v<std::tuple<TT...>>>> {
  using type = std::tuple<TT...>;
  template<class P>
  static void put(P & p, const type & t) {
    std::apply([&p](const TT &... xx) { serial::put<TT...>(p, xx...); }, t);
  }
  static type get(const std::byte *& p) {
    return type{serial::get<TT>(p)...};
  }
};
template<class T, std::size_t N>
struct traits<std::array<T, N>,
  std::enable_if_t<!bit_copyable_v<std::array<T, N>>>> {
  using type = std::array<T, N>;
  template<class P>
  static void put(P & p, const type & a) {
    for(auto e : a) {
      serial::put(p, e);
    }
  }
  template<std::size_t... I>
  static type make_array(const std::byte *& p, std::index_sequence<I...>) {
    return {(void(I), serial::get<T>(p))...};
  }
  static type get(const std::byte *& p) {
    return make_array(p, std::make_index_sequence<N>());
  }
};
template<class T>
struct traits<std::vector<T>> {
  using type = std::vector<T>;
  template<class P>
  static void put(P & p, const type & v) {
    serial::put(p, v.size());
    for(auto & t : v)
      serial::put(p, t);
  }
  static type get(const std::byte *& p) {
    return get_vector<T>(p);
  }
};
template<class T>
struct traits<std::set<T>> : detail::container<std::set<T>> {};
template<class K, class V>
struct traits<std::map<K, V>> : detail::container<std::map<K, V>> {};
template<class K, class V>
struct traits<std::unordered_map<K, V>>
  : detail::container<std::unordered_map<K, V>> {};
template<>
struct traits<std::string> {
  template<class P>
  static void put(P & p, const std::string & s) {
    const auto n = s.size();
    serial::put(p, n);
    mempcpy(p, s.data(), n);
  }
  static std::string get(const std::byte *& p) {
    const auto n = serial::get<std::string::size_type>(p);
    const auto d = p;
    p += n;
    return {reinterpret_cast<const char *>(d), n};
  }
};

// Adapters for other protocols:

/// Convenience for stateless types.
/// Derive their specialization of \c traits from it.
/// \tparam T default-constructible type with no significant state
template<class T>
struct value {
  using type = T;
  /// Do nothing.
  template<class P>
  static void put(P &, const T &) {}
  /// Construct an object.
  /// \return `T()`
  static T get(const std::byte *&) {
    return T();
  }
};

// This works even without Legion:
template<class T>
struct traits<T,
  voided<decltype(&T::legion_buffer_size),
    std::enable_if_t<!bit_copyable_v<T>>>> {
  template<class P>
  static void put(P & p, const T & t) {
    if constexpr(std::is_pointer_v<P>)
      p += t.legion_serialize(p);
    else
      p += t.legion_buffer_size();
  }
  static T get(const std::byte *& p) {
    T ret;
    p += ret.legion_deserialize(p);
    return ret;
  }
};

/// Defines a serialization in terms of a representation type.
/// The primary template is not really a complete type.
/// \tparam T object type
template<class T>
struct convert
#ifdef DOXYGEN
{
  /// Convert an object.
  /// \return a serializable type
  static R put(const T &);
  /// Get the serialized size of an object.
  /// This interface is optional; \c put will be called twice in its absence.
  static std::size_t size(const T &);
  /// Reconstruct an object.
  /// \param r reconstructed object returned from \c put
  static T get(R && r);
}
#endif
;
template<class T, class = void>
struct convert_traits : convert<T> {
  static std::size_t size(const T & t) {
    return serial::size(convert<T>::put(t));
  }
};
template<class T>
struct convert_traits<T, decltype(void(convert<T>::size))> : convert<T> {};
template<class T>
struct traits<T, decltype(void(convert<T>::put))> {
  using Convert = convert_traits<T>;
  template<class P>
  static void put(P & p, const T & t) {
    if constexpr(std::is_pointer_v<P>)
      serial::put(p, Convert::put(t));
    else
      p += Convert::size(t);
  }
  static T get(const std::byte *& p) {
    return Convert::get(
      serial::get<std::decay_t<decltype(Convert::put(std::declval<T>()))>>(p));
  }
};

/// \}
} // namespace serial
} // namespace util
} // namespace flecsi

#endif
