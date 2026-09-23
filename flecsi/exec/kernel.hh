// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_KERNEL_HH
#define FLECSI_EXEC_KERNEL_HH

#include <numeric>

#include "flecsi/config.hh"
#include "flecsi/exec/fold.hh"
#include "flecsi/util/array_ref.hh"

#include <Kokkos_Core.hpp>
#define FLECSI_LAMBDA KOKKOS_LAMBDA

#if defined(__HIPCC__)

#ifdef REALM_USE_HIP
#include "realm/hip/hiphijack_api.h"
#endif
#endif

namespace flecsi {
namespace exec {
/// \defgroup kernel Kernels
/// Local concurrent operations.
/// They use the default Kokkos execution space.
/// To avoid unnecessary copies, one needs to pass a view since the ranges
/// provided by the user are copied.
/// \ingroup execution
/// \{
namespace kok {

template<class R, class T>
struct wrap {
  using reducer = wrap;
  using value_type = T;
  using result_view_type = Kokkos::View<value_type, Kokkos::HostSpace>;

  wrap(T & t) : v(&t) {} // like the built-in reducers

  KOKKOS_INLINE_FUNCTION static void join(T & a, const T & b) {
    a = R::combine(a, b);
  }

  KOKKOS_INLINE_FUNCTION static void init(T & v) {
    new(&v) T(detail::identity_traits<R>::template value<T>);
  }

  // Kokkos doesn't actually use 'reference' from ReducerConcept.
  KOKKOS_INLINE_FUNCTION result_view_type view() const {
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
struct reduce {
  KOKKOS_INLINE_FUNCTION void operator()(const T & v) const {
    t = R::combine(t, v);
  }
  auto kokkos() const {
    return typename reducer_trait<R, T>::type(t);
  }
  T & t;
};

template<class P, class C, class F>
void
parallel_for(const std::string & n, const P & p, C && c, F && f) {
  Kokkos::parallel_for(n,
    p,
    [c = std::forward<C>(c), f = std::forward<F>(f)]
    KOKKOS_FUNCTION(util::id i) { f(c.begin()[i]); });
}
template<class R, class T, class P, class C, class F>
[[nodiscard]] T
parallel_reduce(const std::string & n, const P & p, C && c, F && f) {
  using ref = reduce<R, T>;
  T ret;
  Kokkos::parallel_reduce(
    n,
    p,
    [c = std::forward<C>(c), f = std::forward<F>(f)]
    KOKKOS_FUNCTION(util::id i, T & t) { f(c.begin()[i], ref{t}); },
    ref{ret}.kokkos());
  return ret;
}

} // namespace kok

struct policy_tag {};

template<class... PP>
using policy_type = Kokkos::RangePolicy<Kokkos::IndexType<util::id>, PP...>;

template<typename Range>
struct range_policy : policy_tag {
  range_policy(Range r) : range(std::move(r)) {}
  using Policy = policy_type<>;
  using index = typename Policy::member_type;
  auto get_policy() {
    return Policy(0, range.size());
  }
  Range range;
};

using range_index = range_policy<int>::index;

/// This class computes subinterval of a range based on the starting and ending
/// indices provided.
struct sub_range {
  /// starting index which is inclusive
  range_index beg;
  /// ending index which is exclusive
  range_index end;
  KOKKOS_INLINE_FUNCTION auto size() const {
    return end - beg;
  }

  KOKKOS_INLINE_FUNCTION auto start() const {
    return beg;
  }

  auto get(range_index) const {
    return *this;
  }
};
/// This class computes the range based on the prefix specified
struct prefix_range {
  /// size of the range
  range_index size_len;
  KOKKOS_INLINE_FUNCTION auto size() const {
    return size_len;
  }
  KOKKOS_INLINE_FUNCTION auto start() const {
    return 0;
  }
  auto get(range_index) const {
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

template<std::size_t... II, class... RR>
KOKKOS_INLINE_FUNCTION auto
mdiota_view(std::index_sequence<II...>, const RR &... rr) {
  static constexpr std::size_t N = sizeof...(RR);
  return util::transform_view(
    util::iota_view<range_index>(0, (1 * ... * rr.size())),
    [rr...](range_index i) {
      std::array<range_index, N> ret;
      auto p = ret.end();
      ((*--p =
           [&i, &rr = rr] {
             // capture workaround instead of [&] due to GCC bug
             // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103876
             if constexpr(II < N - 1) {
               const auto n = rr.size(), ret = i % n;
               i /= n;
               return ret;
             }
             else {
               // avoid unused-lambda-capture error due to workaround
               (void)rr;
               return i;
             }
           }() +
           rr.start()),
        ...);
      return ret;
    });
}
// An extra helper is needed to use II to convert full_range objects.
template<class M, std::size_t... II, class R>
KOKKOS_INLINE_FUNCTION auto
mdiota_view(const M & m, std::index_sequence<II...> ii, const R & rt) {
  return mdiota_view(ii, [m, rt] { // pass ranges least-significant first
    constexpr auto J = sizeof...(II) - 1 - II;
    return std::get<J>(rt).get(m.length(J));
  }()...);
}

/// Compute the Cartesian product of several intervals of integers.
/// @param m mdspan or mdcolex object
/// \param rr \c full_range, \c prefix_range, or \c sub_range objects for each
///   dimension, least-significant index first
/// \return sized random-access range of \c std::array objects, each with one
/// index of type \c range_index for each argument in \a rr
template<class M, class... RR>
auto
mdiota_view(const M & m, RR... rr) {
  return mdiota_view(
    m, std::index_sequence_for<RR...>(), std::make_tuple(rr...));
}

/// Call a function on each element of a range, potentially in parallel.
/// If GPU support is available, \a lambda is executed there.
/// \param p sized random-access range
/// \param name operation name, for debugging
/// \deprecated Use \c accelerator::for_each.
template<typename Policy, typename Lambda>
[[deprecated("use accelerator::for_each")]] void
parallel_for(Policy && p, Lambda && lambda, const std::string & name = "") {
  if constexpr(std::is_base_of_v<policy_tag, std::remove_reference_t<Policy>>) {
    kok::parallel_for(name,
      p.get_policy(),
      std::forward<Policy>(p).range,
      std::forward<Lambda>(lambda));
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

/// A parallel range-for loop.  Follow with a compound statement and `;`.
/// Often the elements of \a range (and thus the values of \p it) are indices
/// for other ranges.
///
/// Use as a member function; use in isolation is \b deprecated.
/// \param it variable name to introduce
/// \param P sized random-access range
/// \param name optional debugging name, convertible to \c std::string; not
///   available for the member function form
/// \relates executor_base
#define forall(it, ...)                                                        \
  flecsi_macro_forall(__VA_ARGS__)->*FLECSI_LAMBDA(auto && it)

/// Perform a reduction based on the elements of a range, potentially in
/// parallel.  If GPU support is available, \a lambda is executed there.
/// \tparam R reduction operation type
/// \tparam T data type
/// \tparam Lambda function of an element of \a p and a function object that
///   calls the latter with each value participating in the reduction
/// \param p sized random-access range
/// \param name operation name, for debugging
/// \deprecated Use \c accelerator::reduce.
template<class R, class T, typename Policy, typename Lambda>
[[deprecated("use accelerator::reduce")]] [[nodiscard]] T
parallel_reduce(Policy && p, Lambda && lambda, const std::string & name = "") {
  if constexpr(std::is_base_of_v<policy_tag, std::remove_reference_t<Policy>>) {
    return kok::parallel_reduce<R, T>(name,
      p.get_policy(),
      std::forward<Policy>(p).range,
      std::forward<Lambda>(lambda));
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
  [[nodiscard]] T operator->*(Lambda lambda) && {
    return parallel_reduce<R, T>(std::move(policy_), std::move(lambda), name_);
  }

  Policy policy_;
  std::string name_;
};

/// A parallel reduction loop.
/// Follow with a compound statement to form an expression.
/// Often the elements of \a range (and thus the values of \p it) are indices
/// for other ranges.
///
/// Use as a member function; use in isolation is \b deprecated.
/// \param it variable name to introduce for elements
/// \param ref variable name to introduce for storing results; call it with
///   each value participating in the reduction
/// \param p sized random-access range
/// \param R reduction operation type
/// \param T data type
/// \param name as for <code>\ref forall</code>
/// \return the reduced result
/// \relates executor_base
#define reduceall(it, ref, p, R, T, ...)                                       \
  flecsi_macro_reduceall(static_cast<std::add_pointer_t<R>>(nullptr),          \
    static_cast<std::add_pointer_t<T>>(nullptr),                               \
    p,                                                                         \
    {__VA_ARGS__})                                                             \
      ->*FLECSI_LAMBDA(auto && it, auto ref)

/// \}
} // namespace exec
} // namespace flecsi

// Ugly names to allow unqualified, member-function-compatible use in macros.
template<class P>
flecsi::exec::forall_t<P>
flecsi_macro_forall(P && p, std::string n = {}) {
  return {std::forward<P>(p), std::move(n)};
}
template<class R, class T, class P>
flecsi::exec::reduceall_t<P, R, T>
flecsi_macro_reduceall(R *, T *, P && p, std::optional<std::string> n) {
  return {std::forward<P>(p), std::move(n).value_or("")};
}

#endif
