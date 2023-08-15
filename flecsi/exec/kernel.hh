// Copyright (C) 2016, Triad National Security, LLC
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

template<class R, class T>
struct wrap {
  using reducer = wrap;
  using value_type = T;
  using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace>;

  wrap(T & t) : v(&t) {} // like the built-in reducers

  FLECSI_INLINE_TARGET
  static void join(T & a, const T & b) {
    a = R::combine(a, b);
  }

  FLECSI_INLINE_TARGET
  static void init(T & v) {
    new(&v) T(detail::identity_traits<R>::template value<T>);
  }

  // Kokkos doesn't actually use 'reference' from ReducerConcept.
  FLECSI_INLINE_TARGET
  result_view_type view() const {
    return v;
  }

private:
  result_view_type v;
};

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

template<class R, class T, class = void>
struct reducer_trait {
  using type = wrap<R, T>;
};
template<class R, class T>
struct reducer_trait<R,
  T,
  decltype(Kokkos::reduction_identity<T>(), void(reducer<R>()))> {
  using type = typename reducer<R>::template type<T>;
};

template<class R, class T>
using reducer_t = typename reducer_trait<R, T>::type;
} // namespace kok
#endif

struct policy_tag {};

struct range_base {
#if defined(FLECSI_ENABLE_KOKKOS)
  using Policy = Kokkos::RangePolicy<Kokkos::IndexType<util::id>>;
  using index = Policy::member_type;
#else
  typedef util::id index;
  using Policy = util::iota_view<index>;
#endif
};
/// The integer type used for multi-dimensional indices.
using range_index = range_base::index;

template<typename Range>
struct range_policy : range_base, policy_tag {
  range_policy(Range r) : range(std::move(r)) {}

  FLECSI_INLINE_TARGET auto get_policy() {
    return Policy(0, range.size());
  }
  Range range;
};

/// This class computes subinterval of a range based on the starting and ending
/// indices provided.
struct sub_range {
  /// starting index which is inclusive
  range_index beg;
  /// ending index which is exclusive
  range_index end;
  FLECSI_INLINE_TARGET auto size() const {
    return end - beg;
  }

  FLECSI_INLINE_TARGET auto start() const {
    return beg;
  }

  FLECSI_INLINE_TARGET auto get(range_index) const {
    return *this;
  }
};
/// This class computes the range based on the prefix specified
struct prefix_range {
  /// size of the range
  range_index size_len;
  FLECSI_INLINE_TARGET auto size() const {
    return size_len;
  }
  FLECSI_INLINE_TARGET auto start() const {
    return 0;
  }
  FLECSI_INLINE_TARGET auto get(range_index) const {
    return *this;
  }
};
/// This class computes full range size if prefix or subinterval of the range is
/// not specified
struct full_range {
  auto get(range_index n) const {
    return prefix_range{n};
  }
};

template<std::size_t N, std::size_t II, class RR, class AR>
class nested_f
{
  AR & ret;
  const RR & rr;

public:
  __attribute__((host)) __attribute__((device))
  nested_f(const RR & Rr, AR & Ret)
    : rr(Rr), ret(Ret) {}
  __attribute__((host)) __attribute__((device)) auto operator()(
    range_index & i) {
    if constexpr(II < N - 1) {
      const auto n = rr.size(), ret = i % n;
      i /= n;
      return ret;
    }
    else {
      return i;
    }
  }
};

template<class IdxSeq, class... RR>
class mv_f;

template<std::size_t... II, class... RR>
class mv_f<std::index_sequence<II...>, RR...>
{
  std::tuple<RR...> my_tuple;
  std::size_t N = sizeof...(RR);
  std::index_sequence<II...> idx_seq;

public:
  __attribute__((host)) __attribute__((device))
  mv_f(std::index_sequence<II...>, const RR &... tt)
    : my_tuple(tt...) {}

  __attribute__((host)) __attribute__((device)) auto operator()(
    range_index i) const {
    static constexpr std::size_t N = sizeof...(RR);
    std::array<range_index, N> ret;
    auto p = ret.end();
    ((*--p =
         nested_f<N, II, RR, std::array<range_index, N>>{
           std::get<II>(my_tuple), ret}(i) +
         std::get<II>(my_tuple).start()),
      ...);
    return ret;
  }
};

template<std::size_t... II, class... RR>
FLECSI_INLINE_TARGET auto
mdiota_view(std::index_sequence<II...> ii, const RR &... rr) {
  static constexpr std::size_t N = sizeof...(RR);
  return util::transform_view(
    util::iota_view<range_index>(0, (1 * ... * rr.size())),
    mv_f<std::index_sequence<II...>, RR...>{ii, rr...});
}
// An extra helper is needed to use II to convert full_range objects.
template<class M, std::size_t... II, class R>
FLECSI_INLINE_TARGET auto
mdiota_view(const M & m, std::index_sequence<II...> ii, const R & rt) {
  return mdiota_view(ii, [m, rt] { // pass ranges least-significant first
    constexpr auto J = sizeof...(II) - 1 - II;
    return std::get<J>(rt).get(m.length(J));
  }()...);
}

/// Compute the Cartesian product of several intervals of integers.
/// @param m mdspan or mdcolex object
/// \param rr \c full_range, \c prefix_range, or \c sub_range objects for each
///   dimension
/// \return sized random-access range of \c std::array objects, each with one
/// index of type \c range_index for each argument in \a rr
template<class M, class... RR>
auto
mdiota_view(const M & m, RR... rr) {
  return mdiota_view(
    m, std::index_sequence_for<RR...>(), std::make_tuple(rr...));
}

// This could be a lambda in parallel_for, but NVCC doesn't like that.
template<class R, class F>
struct index_wrapper {
  R r;
  F f;
  FLECSI_TARGET void operator()(int i) {
    f(r.begin()[i]);
  }
  FLECSI_TARGET void operator()(int i) const {
    f(r.begin()[i]);
  }
};
template<class R, class F>
index_wrapper(R, F) -> index_wrapper<R, F>; // automatic in C++20

/// This function is a wrapper for Kokkos::parallel_for that has been adapted to
/// work with random access ranges common in FleCSI topologies. In particular,
/// this function invokes a map from the normal kernel index space to the FleCSI
/// index space, which may require indirection.
/// \param p sized random-access range
/// \param name operation name, for debugging
template<typename Policy, typename Lambda>
void
parallel_for(Policy && p, Lambda && lambda, const std::string & name = "") {
  if constexpr(std::is_base_of_v<policy_tag, std::remove_reference_t<Policy>>) {
    auto policy_type = p.get_policy(); // before moving
#if defined(FLECSI_ENABLE_KOKKOS)
    // nvcc does not support init capture for extended host device lambdas
    // so use temporary variables
    const auto it = std::forward<Policy>(p).range;
    auto f = std::forward<Lambda>(lambda);
    Kokkos::parallel_for(name,
      policy_type,
      // [it, f] FLECSI_TARGET(int i) {
      //   f(it.begin()[i]);
      // });
      index_wrapper{it, f});
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
#ifdef __CUDACC__ // NV
  __host__ __device__
#endif
    FLECSI_INLINE_TARGET void
    operator()(const T & v) const {
    t = R::combine(t, v);
  }
  T & t;
};
} // namespace detail

// This could be a lambda in parallel_for, but NVCC doesn't like that.
template<class I, class F, class R, class T>
struct index_reduce_wrapper {
  using ref = detail::reduce_ref<R, T>;
  I it;
  F f;
  FLECSI_TARGET void operator()(int i, T & tmp) {
    f(it.begin()[i], ref{tmp});
  }
  FLECSI_TARGET void operator()(int i, T & tmp) const {
    f(it.begin()[i], ref{tmp});
  }
};
// Partial CTAD isn't a thing, apparently
// template<class I, class F, class R, class T>
// index_reduce_wrapper<R, T>(I, F) -> index_reduce_wrapper<I, F, R, T>;

/// This function is a wrapper for Kokkos::parallel_reduce that has been adapted
/// to work with random access ranges common in FleCSI topologies.
/// \tparam R reduction operation type
/// \tparam T data type
/// \tparam Lambda function of an element of \a p and a function object that
///   calls the latter with each value participating in the reduction
/// \param p sized random-access range
/// \param name operation name, for debugging
template<class R, class T, typename Policy, typename Lambda>
T
parallel_reduce(Policy && p, Lambda && lambda, const std::string & name = "") {
  if constexpr(std::is_base_of_v<policy_tag, std::remove_reference_t<Policy>>) {
    auto policy_type = p.get_policy(); // before moving
#if defined(FLECSI_ENABLE_KOKKOS)
    kok::wrap<R, T> result;
    auto it = std::forward<Policy>(p).range;
    auto f = std::forward<Lambda>(lambda);
    Kokkos::parallel_reduce(name,
      policy_type,
      // [it, f] FLECSI_TARGET(int i, T& tmp) {
      //   f(it.begin()[i], ref{tmp});
      // },
      index_reduce_wrapper<decltype(it), decltype(f), R, T>{it, f},
      result.kokkos());
    return result.reference();
#else
    (void)name;
    T res = detail::identity_traits<R>::template value<T>;
    ref r{res};
    for(auto i : policy_type)
      lambda(p.range.begin()[i], r);
#endif
    return res;
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

/// A parallel reduction loop.
/// Follow with a compound statement to form an expression.
/// Often the elements of \a range (and thus the values of \p it) are indices
/// for other ranges.
/// \param it variable name to introduce for elements
/// \param ref variable name to introduce for storing results; call it with
///   each value participating in the reduction
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
