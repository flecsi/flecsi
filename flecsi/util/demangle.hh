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

#include <string>
#include <typeinfo> // typeid()

namespace flecsi {
namespace util {

/*!
  Return the demangled name.

  @param name The string to demangle.

  @ingroup utils
 */

std::string demangle(const char * const name);

/*!
  Return the demangled name of the type T.

  @tparam T the type (references and cv-qualification ignored)

  @ingroup utils
 */

template<class T>
inline std::string
type() {
  return demangle(typeid(T).name());
} // type

/*!
  Return the demangled name of the type identified by type_info.

  @ingroup utils
 */

inline std::string
type(const std::type_info & type_info) {
  return demangle(type_info.name());
} // type

/// Dummy class template.
/// \tparam S a reference to a function or variable
template<auto & S>
struct Symbol {};
/// Return the name of the template argument.
/// \tparam a reference to a function or variable
/// \return demangled name
template<auto & S>
std::string
symbol() {
  constexpr int PFX = sizeof("flecsi::util::Symbol<") - 1;
  const auto s = type<Symbol<S>>();
  return s.substr(PFX, s.size() - 1 - PFX);
}

} // namespace util
} // namespace flecsi
