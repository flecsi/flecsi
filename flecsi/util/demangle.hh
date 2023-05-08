// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_DEMANGLE_HH
#define FLECSI_UTIL_DEMANGLE_HH

#include <string>
#include <typeinfo> // typeid()

namespace flecsi {
namespace util {
/// \addtogroup utils
/// \{

/*!
  Return the demangled name.

  @param name The string to demangle.
 */

std::string demangle(const char * const name);

/*!
  Return signature without parameter list

  @param sig The function signature
 */

std::string strip_parameter_list(const std::string & sig);

/*!
  Return signature without return type

  While this works for many cases, if the heuristic fails to detect the return
  type, it will return the full signature.

  Example of a not supported signature:
  \code
  void f<X<void, K<(0)> e<0, 0>()>()
  \endcode

  @param sig The function signature
 */

std::string strip_return_type(const std::string & sig);

/*!
  Return the demangled name of the type T.

  @tparam T the type (references and cv-qualification ignored)
 */

template<class T>
inline std::string
type() {
  return demangle(typeid(T).name());
} // type

/*!
  Return the demangled name of the type identified by type_info.
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

/// \}
} // namespace util
} // namespace flecsi

#endif
