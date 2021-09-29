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
#define FLECSI_LAMBDA [=] FLECSI_TARGET
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
  Kokkos provides different execution policies that controls how the parallel
  execution is done in Kokkos::parallel_for and Kokkos::parallel_reduce. FleCSI
  currently provide support for the following execution policies; 1) range
  policy, 2) range policy with lower and upper bounds defined.
 */

struct policy_tag {};

template<typename Range>
struct range_policy : policy_tag {
  range_policy(Range r) : range(r) {}
#if defined(FLECSI_ENABLE_KOKKOS)
  auto get_policy() {
    return Kokkos::RangePolicy<>(0, range.size());
  }
#endif
  Range range;
};
template<class R>
range_policy(R)->range_policy<R>;

struct range_bound_base {
#if defined(FLECSI_ENABLE_KOKKOS)
  typedef Kokkos::RangePolicy<>::member_type index;
#else
  typedef util::counter_t index;
#endif
};

template<typename Range>
struct range_bound : range_bound_base, policy_tag {
  range_bound(Range r, index l, index u) : range(r), lb(l), ub(u) {}
#if defined(FLECSI_ENABLE_KOKKOS)
  auto get_policy() {
    return Kokkos::RangePolicy<>(lb, ub);
  }
#endif
  Range range;
  index lb;
  index ub;
};

template<class R>
range_bound(R, range_bound_base::index, range_bound_base::index)
  ->range_bound<R>;

/*!
  This function is a wrapper for Kokkos::parallel_for that has been adapted to
  work with random access ranges common in FleCSI topologies. The parallel_for
  function takes in policy objects or the range. In particular, this function
  invokes a map from the normal kernel index space to the FleCSI index space,
  which may require indirection.
 */

template<typename Policy, typename Lambda>
void
parallel_for(Policy && p, Lambda && lambda, const std::string & name = "") {
  if constexpr(std::is_base_of_v<policy_tag, std::remove_reference_t<Policy>>) {
#if defined(FLECSI_ENABLE_KOKKOS)
    Kokkos::parallel_for(name,
      std::move(p.get_policy()),
      [it = std::forward<Policy>(p).range,
        f = std::forward<Lambda>(lambda)] FLECSI_TARGET(int i) {
        return f(it[i]);
      });
#else
    (void)name;
    std::for_each(p.range.begin(), p.range.end(), lambda);
#endif
  }
  else {
    parallel_for(range_policy(std::forward<Policy>(p)),
      std::forward<Lambda>(lambda),
      name);
  }
} // parallel_for

template<typename P>
struct forall_t {
  template<typename Callable>
  void operator->*(Callable l) && {
    parallel_for(std::move(policy_), std::move(l), name_);
  }
  P policy_;
  std::string name_;
}; // struct forall_t
template<class P>
forall_t(P, std::string)->forall_t<P>; // automatic in C++20

/*!
  The forall type provides a pretty interface for invoking data-parallel
  execution.
 */

#define forall(it, P, name)                                                    \
  ::flecsi::exec::forall_t{P, name}->*FLECSI_LAMBDA(auto && it)

/*!
  This function is a wrapper for Kokkos::parallel_reduce that has been adapted
  to work with random access ranges common in FleCSI topologies. The 
  parallel_reduce function takes in policy objects or the range.
 */
template<class R, class T, typename Policy, typename Lambda>
T
parallel_reduce(Policy && p, Lambda && lambda, const std::string & name = "") {
  if constexpr(std::is_base_of_v<policy_tag, std::remove_reference_t<Policy>>) {
#if defined(FLECSI_ENABLE_KOKKOS)
    kok::wrap<R, T> result;
    Kokkos::parallel_reduce(
      name,
      std::move(p.get_policy()),
      [it = std::forward<Policy>(p).range,
        f = std::forward<Lambda>(lambda)] FLECSI_TARGET(int i, T & tmp) {
        return f(it[i], tmp);
      },
      result.kokkos());
    return result.reference();
#else
    (void)name;
    T res = detail::identity_traits<R>::template value<T>;
    std::for_each(p.range.begin(),
      p.range.end(),
      [f = std::forward<Lambda>(lambda), &res](
        auto && i) { return f(i, res); });
    return res;
#endif
  }
  else {
    return parallel_reduce<R, T>(range_policy(std::forward<Policy>(p)),
      std::forward<Lambda>(lambda),
      name);
  }
} // parallel_reduce

/*!
  The reduce_all type provides a pretty interface for invoking data-parallel
  reductions.
 */
template<class Policy, class R, class T>
struct reduceall_t {
  template<typename Lambda>
  T operator->*(Lambda lambda) && {
    return parallel_reduce<R, T>(std::move(policy_), std::move(lambda), name_);
  }

  Policy policy_;
  std::string name_;
};

template<class R, class T, class P>
reduceall_t<P, R, T>
make_reduce(P p, std::string n) {
  return {std::move(p), n};
}

#define reduceall(it, tmp, P, R, T, name)                                      \
  ::flecsi::exec::make_reduce<R, T>(P, name)->*FLECSI_LAMBDA(                  \
                                                 auto && it, T & tmp)

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
