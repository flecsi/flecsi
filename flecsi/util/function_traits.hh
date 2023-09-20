// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_FUNCTION_TRAITS_HH
#define FLECSI_UTIL_FUNCTION_TRAITS_HH

#include <functional>
#include <tuple>

namespace flecsi {
namespace util {

template<typename T>
struct function_traits {};

template<typename R, typename... As>
struct function_traits<R(As...)> {
  static constexpr bool nonthrowing = false;
  using return_type = R;
  using arguments_type = std::tuple<As...>;
};

template<typename R, typename... As>
struct function_traits<R(As...) noexcept> : function_traits<R(As...)> {
  static constexpr bool nonthrowing = true;
};

template<auto & F>
using function_t = function_traits<std::remove_reference_t<decltype(F)>>;

} // namespace util
} // namespace flecsi
#endif
