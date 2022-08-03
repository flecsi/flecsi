// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_COMMON_HH
#define FLECSI_UTIL_COMMON_HH

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <type_traits>
#include <vector>

namespace flecsi {
/// Type for spatial dimensions and counts thereof.
/// \ingroup utils
using Dimension = unsigned short;

namespace util {
/// \defgroup utils Utilities
/// Non-domain-specific infrastructure.
/// \{

//----------------------------------------------------------------------------//
// Entity id type. This type should be used for id types for entities
// in topologies.
//----------------------------------------------------------------------------//

#ifndef FLECSI_ID_TYPE
#define FLECSI_ID_TYPE std::uint32_t
#endif

using id = FLECSI_ID_TYPE;

//----------------------------------------------------------------------------//
// Index type
//----------------------------------------------------------------------------//

#ifndef FLECSI_COUNTER_TYPE
#define FLECSI_COUNTER_TYPE int32_t
#endif

using counter_t = FLECSI_COUNTER_TYPE;

/// \cond core

template<class T>
constexpr std::enable_if_t<std::is_unsigned_v<T>, T>
ceil_div(T a, T b) {
  return a / b + !!(a % b); // avoids overflow in (a+(b-1))/b
}

//----------------------------------------------------------------------------//
// Square
//----------------------------------------------------------------------------//

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

template<typename T>
void
force_unique(std::vector<T> & v) {
  std::sort(v.begin(), v.end());
  auto first = v.begin();
  auto last = std::unique(first, v.end());
  v.erase(last, v.end());
}

template<typename K, typename T>
void
unique_each(std::map<K, T> & m) {
  for(auto & v : m)
    force_unique(v.second);
}

template<typename T>
void
unique_each(std::vector<T> & vv) {
  for(auto & v : vv)
    force_unique(v);
}

struct identity {
  template<class T>
  T && operator()(T && x) {
    return std::forward<T>(x);
  }
};

/// \endcond
/// \}
} // namespace util
} // namespace flecsi

#endif
