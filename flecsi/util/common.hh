// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_COMMON_HH
#define FLECSI_UTIL_COMMON_HH

#include "flecsi/config.hh"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <ios>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace flecsi {
/// Type for spatial dimensions and counts thereof.
/// \ingroup utils
using Dimension = unsigned short;

namespace util {
/// \addtogroup utils
/// \{

/// Local (color-specific) topology entity ID type.
/// Often provided in a index-space-specific convertible wrapper.
using id = FLECSI_ID_TYPE;

/// Global topology entity ID type.
using gid = FLECSI_GID_TYPE;

/// Interpret a type as itself in functional contexts.
struct identity {
  template<class T>
  T && operator()(T && x) {
    return std::forward<T>(x);
  }
};

/// \cond core

template<class T>
constexpr std::enable_if_t<std::is_unsigned_v<T>, T>
ceil_div(T a, T b) {
  return a / b + !!(a % b); // avoids overflow in (a+(b-1))/b
}

//! P.O.D.
template<typename T>
inline T
square(const T & a) {
  return a * a;
}

/// A counter with a maximum.
template<auto M>
struct counter {
  using type = decltype(M);

  constexpr explicit counter(type l) : last(l) {}

  const type & operator()() {
    assert(last < M && "counter overflow");
    return ++last;
  }

private:
  type last;
};

/// Sort a std::vector and remove duplicates.
template<typename T>
void
force_unique(std::vector<T> & v) {
  std::sort(v.begin(), v.end());
  auto first = v.begin();
  auto last = std::unique(first, v.end());
  v.erase(last, v.end());
}

/// Apply force_unique to each element of a std::map. Note that force_unique
/// is currently only implemented for std::vector.
template<typename K, typename T>
void
unique_each(std::map<K, T> & m) {
  for(auto & v : m)
    force_unique(v.second);
}

/// Apply force_unique to each element of a std::vector. Note that force_unique
/// is currently only implemented for std::vector.
template<typename T>
void
unique_each(std::vector<T> & vv) {
  for(auto & v : vv)
    force_unique(v);
}

struct FILE {
  FILE(const char * n, const char * m) : f(std::fopen(n, m)) {
    if(!f)
      throw std::ios::failure(
        "cannot open file", {errno, std::system_category()});
  }
  FILE(FILE && f) noexcept : f(std::exchange(f.f, {})) {}
  ~FILE() {
    if(f)
      std::fclose(f); // NB: error lost
  }
  FILE & operator=(FILE src) & noexcept {
    std::swap(f, src.f);
    return *this;
  }

  operator ::FILE *() const noexcept {
    return f;
  }

private:
  ::FILE * f;
};

/// \endcond
/// \}
} // namespace util
} // namespace flecsi

#endif
