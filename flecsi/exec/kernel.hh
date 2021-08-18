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

#include <numeric>

#include "flecsi/exec/fold.hh"

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>
#define FLECSI_LAMBDA KOKKOS_LAMBDA
#else
#define FLECSI_LAMBDA [=]
#endif

namespace flecsi {
namespace exec {
#if defined(FLECSI_ENABLE_KOKKOS)
namespace kok {

template<class R, class T, class = void>
struct wrap {
  using reducer = wrap;
  using value_type = T;
  using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace>;

  FLECSI_INLINE_TARGET
  void join(T & a, const T & b) const {
    a = R::combine(a, b);
  }

  FLECSI_INLINE_TARGET
  void join(volatile T & a, const volatile T & b) const {
    a = R::combine(a, b);
  }

  FLECSI_INLINE_TARGET
  void init(T & v) const {
    v = detail::identity_traits<R>::template value<T>;
  }

  // Also useful to read the value!
  FLECSI_INLINE_TARGET
  T & reference() const {
    return t;
  }

  FLECSI_INLINE_TARGET
  result_view_type view() const {
    return &t;
  }

  wrap & kokkos() {
    return *this;
  }

private:
  mutable T t;
};

// Kokkos's built-in reducers are just as effective as ours for generic
// types, although we can't provide Kokkos::reduction_identity in terms of
// our interface in C++17 because it has no extra template parameter via
// which to apply SFINAE.
template<class>
struct reducer; // undefined
template<>
struct reducer<fold::min> {
  template<class T>
  using type = Kokkos::Min<T>;
};
template<>
struct reducer<fold::max> {
  template<class T>
  using type = Kokkos::Max<T>;
};
template<>
struct reducer<fold::sum> {
  template<class T>
  using type = Kokkos::Sum<T>;
};
template<>
struct reducer<fold::product> {
  template<class T>
  using type = Kokkos::Prod<T>;
};

template<class R, class T>
struct wrap<R, T, decltype(void(reducer<R>()))> {
private:
  T t;
  typename reducer<R>::template type<T> native{t};

public:
  const T & reference() const {
    return t;
  }
  auto & kokkos() {
    return native;
  }
};
} // namespace kok
#endif
/*!
  This function is a wrapper for Kokkos::parallel_for that has been adapted to
  work with FleCSI's topology iterator types. In particular, this function
  invokes a map from the normal kernel index space to the FleCSI index space,
  which may require indirection.
 */

template<typename Range, typename Lambda>
void
parallel_for(Range && range, Lambda && lambda, const std::string & name = "") {
#if defined(FLECSI_ENABLE_KOKKOS)
  const auto n = range.size(); // before moving
  Kokkos::parallel_for(name,
    n,
    [it = std::forward<Range>(range),
      f = std::forward<Lambda>(lambda)] FLECSI_TARGET(int i) {
      return f(it[i]);
    });
#else
  (void)name;
  std::for_each(range.begin(),
    range.end(),
    [f = std::forward<Lambda>(lambda)] FLECSI_TARGET(
      auto && i) { return f(i); });
#endif
} // parallel_for

/*!
  The forall type provides a pretty interface for invoking data-parallel
  execution.
 */

template<typename Range>
struct forall_t {
  template<typename Callable>
  void operator->*(Callable l) && {
    parallel_for(std::move(range_), std::move(l), name_);
  }

  Range range_;
  std::string name_;
}; // struct forall_t
template<class I>
forall_t(I, std::string)->forall_t<I>; // automatic in C++20

#define forall(it, range, name)                                                \
  ::flecsi::exec::forall_t{range, name}->*FLECSI_LAMBDA(auto && it)

/*!
  This function is a wrapper for Kokkos::parallel_reduce that has been adapted
  to work with FleCSI's topology iterator types.
 */
template<class R, class T, typename Range, typename Lambda>
T
parallel_reduce(Range && range,
  Lambda && lambda,
  const std::string & name = "") {
#if defined(FLECSI_ENABLE_KOKKOS)
  const auto n = range.size(); // before moving
  kok::wrap<R, T> result;
  Kokkos::parallel_reduce(
    name,
    n,
    [it = std::forward<Range>(range),
      f = std::forward<Lambda>(lambda)] FLECSI_TARGET(int i, T & tmp) {
      return f(it[i], tmp);
    },
    result.kokkos());
  return result.reference();
#else
  (void)name;
  T res = detail::identity_traits<R>::template value<T>;
  std::for_each(range.begin(),
    range.end(),
    [f = std::forward<Lambda>(lambda), &res] FLECSI_TARGET(
      auto && i) { return f(i, res); });
  return res;
#endif

} // parallel_reduce

/*!
  The reduce_all type provides a pretty interface for invoking data-parallel
  reductions.
 */
template<class Range, class R, class T>
struct reduceall_t {
  template<typename Lambda>
  T operator->*(Lambda lambda) && {
    return parallel_reduce<R, T>(std::move(range_), std::move(lambda), name_);
  }

  Range range_;
  std::string name_;
};

template<class R, class T, class I>
reduceall_t<I, R, T>
make_reduce(I i, std::string n) {
  return {std::move(i), n};
}

#define reduceall(it, tmp, range, R, T, name)                                  \
  ::flecsi::exec::make_reduce<R, T>(range, name)                               \
      ->*FLECSI_LAMBDA(auto && it, T & tmp)

//----------------------------------------------------------------------------//
//! Abstraction function for fine-grained, data-parallel interface.
//!
//! @tparam R range type
//! @tparam FUNCTION    The calleable object type.
//!
//! @param r range over which to execute \a function
//! @param function     The calleable object instance.
//!
//! @ingroup execution
//----------------------------------------------------------------------------//

template<class R, typename Function>
inline void
for_each(R && r, Function && function) {
  std::for_each(r.begin(), r.end(), std::forward<Function>(function));
} // for_each_u

//----------------------------------------------------------------------------//
//! Abstraction function for fine-grained, data-parallel interface.
//!
//! @tparam R range type
//! @tparam Function    The calleable object type.
//! @tparam Reduction   The reduction variabel type.
//!
//! @param r range over which to execute \a function
//! @param function     The calleable object instance.
//!
//! @ingroup execution
//----------------------------------------------------------------------------//

template<class R, typename Function, typename Reduction>
inline void
reduce_each(R && r, Reduction & reduction, Function && function) {
  for(const auto & e : r)
    function(e, reduction);
} // reduce_each_u

} // namespace exec
} // namespace flecsi
