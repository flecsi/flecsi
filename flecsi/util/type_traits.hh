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

#include <type_traits>

namespace flecsi {
namespace util {

namespace detail {

template<typename... Ts>
struct hold {};

} // namespace detail

// Workaround for Clang's eager reduction of void_t (see also CWG1980)
template<class... TT>
using voided = std::conditional_t<false, detail::hold<TT...>, void>;

} // namespace util
} // namespace flecsi
