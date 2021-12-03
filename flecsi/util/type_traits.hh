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

#include <type_traits>

/// \cond core
namespace flecsi {
namespace util {
/// \addtogroup utils
/// \{

namespace detail {

template<typename... Ts>
struct hold {};

} // namespace detail

// Workaround for Clang's eager reduction of void_t (see also CWG1980)
template<class... TT>
using voided = std::conditional_t<false, detail::hold<TT...>, void>;

/// \}
} // namespace util
} // namespace flecsi
/// \endcond
