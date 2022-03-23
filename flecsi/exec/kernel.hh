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

namespace flecsi {
namespace exec {
/// \defgroup kernel Kernels
/// Local concurrent operations.
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
/*!
  This function is a wrapper for Kokkos::parallel_for that has been adapted to
  work with random access ranges common in FleCSI topologies.
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
  std::for_each(range.begin(), range.end(), lambda);
#endif
} // parallel_for

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
forall_t(I, std::string) -> forall_t<I>; // automatic in C++20

/// A parallel range-for loop.  Follow with a compound statement and `;`.
/// Often the elements of \a range (and thus the values of \p it) are indices
/// for other ranges.
/// \param it variable name to introduce
/// \param range sized random-access range
/// \param name debugging name, convertible to \c std::string
#define forall(it, range, name)                                                \
  ::flecsi::exec::forall_t{range, name}->*FLECSI_LAMBDA(auto && it)

namespace detail {
template<class R, class T>
struct reduce_ref {
  void operator()(const T & v) const {
    t = R::combine(t, v);
  }
  T & t;
};
} // namespace detail

/*!
  This function is a wrapper for Kokkos::parallel_reduce that has been adapted
  to work with random access ranges common in FleCSI topologies.
 */
template<class R, class T, typename Range, typename Lambda>
T
parallel_reduce(Range && range,
  Lambda && lambda,
  const std::string & name = "") {
  using ref = detail::reduce_ref<R, T>;
#if defined(FLECSI_ENABLE_KOKKOS)
  const auto n = range.size(); // before moving
  kok::wrap<R, T> result;
  Kokkos::parallel_reduce(
    name,
    n,
    [it = std::forward<Range>(range),
      f = std::forward<Lambda>(lambda)] FLECSI_TARGET(int i, T & tmp) {
      return f(it[i], ref{tmp});
    },
    result.kokkos());
  return result.reference();
#else
  (void)name;
  T res = detail::identity_traits<R>::template value<T>;
  std::for_each(range.begin(),
    range.end(),
    [f = std::forward<Lambda>(lambda), res = ref{res}](
      auto && i) { return f(i, res); });
  return res;
#endif

} // parallel_reduce

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

/// A parallel reduction loop.  Follow with a compound statement and `;`.
/// Often the elements of \a range (and thus the values of \p it) are indices
/// for other ranges.
/// \param it variable name to introduce for elements
/// \param ref variable name to introduce for storing results: call it to add
///   a value to the reduction
/// \param range sized random-access range
/// \param R reduction operation type
/// \param T data type
/// \param name debugging name, convertible to \c std::string
/// \return the reduced result
#define reduceall(it, ref, range, R, T, name)                                  \
  ::flecsi::exec::make_reduce<R, T>(range, name)                               \
      ->*FLECSI_LAMBDA(auto && it, auto ref)

/// \}
} // namespace exec
} // namespace flecsi

#endif
