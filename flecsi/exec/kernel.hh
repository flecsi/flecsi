// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_KERNEL_HH
#define FLECSI_EXEC_KERNEL_HH

#include <numeric>

#include "flecsi/exec/fold.hh"

#if defined(FLECSI_ENABLE_KOKKOS)
#include <Kokkos_Core.hpp>
#define FLECSI_LAMBDA KOKKOS_LAMBDA
#else
#define FLECSI_LAMBDA [=] FLECSI_TARGET
#endif

#if defined(__HIPCC__)

#ifdef REALM_USE_HIP
#include "realm/hip/hiphijack_api.h"
#endif
#endif

namespace flecsi {
namespace exec {
/// \defgroup kernel Kernels
/// Local concurrent operations.
/// If Kokkos is not available, they simply execute serially.
/// To avoid unnecessary copies, one needs to pass a view since the ranges
/// provided by the user are copied.
/// \ingroup execution
/// \{
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

struct policy_tag {};

struct range_base {
#if defined(FLECSI_ENABLE_KOKKOS)
  using Policy = Kokkos::RangePolicy<>;
  using index = Policy::member_type;
#else
  typedef util::counter_t index;
  using Policy = util::iota_view<index>;
#endif
};

template<typename Range>
struct range_policy : range_base, policy_tag {
  range_policy(Range r) : range(std::move(r)) {}

  auto get_policy() {
    return Policy(0, range.size());
  }
  Range range;
};

/// This function is a wrapper for Kokkos::parallel_for that has been adapted to
/// work with random access ranges common in FleCSI topologies. In particular,
/// this function invokes a map from the normal kernel index space to the FleCSI
/// index space, which may require indirection.
/// \param p sized random-access range
template<typename Policy, typename Lambda>
void
parallel_for(Policy && p, Lambda && lambda, const std::string & name = "") {
  if constexpr(std::is_base_of_v<policy_tag, std::remove_reference_t<Policy>>) {
    auto policy_type = p.get_policy(); // before moving
#if defined(FLECSI_ENABLE_KOKKOS)
    Kokkos::parallel_for(name,
      policy_type,
      [it = std::forward<Policy>(p).range,
        f = std::forward<Lambda>(lambda)] FLECSI_TARGET(int i) {
        f(it.begin()[i]);
      });
#else
    (void)name;
    for(auto i : policy_type)
      lambda(p.range.begin()[i]);
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
forall_t(P, std::string) -> forall_t<P>; // automatic in C++20

/// A parallel range-for loop.  Follow with a compound statement and `;`.
/// Often the elements of \a range (and thus the values of \p it) are indices
/// for other ranges.
/// \param it variable name to introduce
/// \param P sized random-access range
/// \param name debugging name, convertible to \c std::string
#define forall(it, P, name)                                                    \
  ::flecsi::exec::forall_t{P, name}->*FLECSI_LAMBDA(auto && it)

namespace detail {
template<class R, class T>
struct reduce_ref {
  FLECSI_INLINE_TARGET void operator()(const T & v) const {
    t = R::combine(t, v);
  }
  T & t;
};
} // namespace detail

/// This function is a wrapper for Kokkos::parallel_reduce that has been adapted
/// to work with random access ranges common in FleCSI topologies.
/// \param p sized random-access range

template<class R, class T, typename Policy, typename Lambda>
T
parallel_reduce(Policy && p, Lambda && lambda, const std::string & name = "") {
  if constexpr(std::is_base_of_v<policy_tag, std::remove_reference_t<Policy>>) {
    auto policy_type = p.get_policy(); // before moving
    using ref = detail::reduce_ref<R, T>;
#if defined(FLECSI_ENABLE_KOKKOS)
    kok::wrap<R, T> result;
    Kokkos::parallel_reduce(
      name,
      policy_type,
      [it = std::forward<Policy>(p).range,
        f = std::forward<Lambda>(lambda)] FLECSI_TARGET(int i, T & tmp) {
        f(it.begin()[i], ref{tmp});
      },
      result.kokkos());
    return result.reference();
#else
    (void)name;
    T res = detail::identity_traits<R>::template value<T>;
    ref r{res};
    for(auto i : policy_type)
      lambda(p.range.begin()[i], r);
    return res;
#endif
  }
  else {
    return parallel_reduce<R, T>(range_policy(std::forward<Policy>(p)),
      std::forward<Lambda>(lambda),
      name);
  }
} // parallel_reduce

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
make_reduce(P policy, std::string n) {
  return {std::move(policy), n};
}

/// A parallel reduction loop.  Follow with a compound statement and `;`.
/// Often the elements of \a range (and thus the values of \p it) are indices
/// for other ranges.
/// \param it variable name to introduce for elements
/// \param ref variable name to introduce for storing results: call it to add
///   a value to the reduction
/// \param p sized random-access range
/// \param R reduction operation type
/// \param T data type
/// \param name debugging name, convertible to \c std::string
/// \return the reduced result
#define reduceall(it, ref, p, R, T, name)                                      \
  ::flecsi::exec::make_reduce<R, T>(p, name)->*FLECSI_LAMBDA(                  \
                                                 auto && it, auto ref)

/// \}
} // namespace exec
} // namespace flecsi

#endif
