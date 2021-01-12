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

#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <typeinfo>
#include <vector>

namespace flecsi {
namespace util {

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

//----------------------------------------------------------------------------//
// Square
//----------------------------------------------------------------------------//

//! P.O.D.
template<typename T>
inline T
square(const T & a) {
  return a * a;
}

//----------------------------------------------------------------------------//
// Unique Identifier Utilities
//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
// This value is used by the Legion runtime backend to automatically
// assign task and field ids. The current maximum value that is allowed
// in legion_config.h is 1<<20.
//
// We are reserving 4096 places for internal use.
//----------------------------------------------------------------------------//

#if !defined(FLECSI_GENERATED_ID_MAX)
// 1044480 = (1<<20) - 4096
#define FLECSI_GENERATED_ID_MAX 1044480
#endif

/*!
  The unique_id type provides a utility to generate a series of unique ids.

  @tparam UNIQUENESS_TYPE A dummy type to differentiate instances.
  @tparam COUNTER_TYPE    The underlying counter type.
  @tparam MAXIMUM         The maximum legal id.
 */

template<typename UNIQUENESS_TYPE,
  typename COUNTER_TYPE = size_t,
  COUNTER_TYPE MAXIMUM = (std::numeric_limits<COUNTER_TYPE>::max)()>
struct unique_id {

  static_assert(std::is_integral<COUNTER_TYPE>::value,
    "COUNTER_TYPE must be an integral type");

  static unique_id & instance() {
    static unique_id u;
    return u;
  } // instance

  auto next() {
    assert(id_ + 1 <= MAXIMUM && "id exceeds maximum value");
    return ++id_;
  } // next

private:
  unique_id() : id_(0) {}
  unique_id(const unique_id &) {}
  ~unique_id() {}

  COUNTER_TYPE id_;
}; // unique_id

//! Create a unique name from the type, address, and unique id
template<typename T>
std::string
unique_name(const T * const t) {
  const void * const address = static_cast<const void *>(t);
  const std::size_t id = unique_id<T>::instance().next();
  std::stringstream ss;
  ss << typeid(T).name() << "-" << address << "-" << id;
  return ss.str();
} // unique_name

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
force_unique(std::map<K, std::vector<T>> & m) {
  for(auto & v : m)
    force_unique(v.second);
}

template<typename T>
void
force_unique(std::vector<std::vector<T>> & vv) {
  for(auto & v : vv)
    force_unique(v);
}

} // namespace util
} // namespace flecsi
