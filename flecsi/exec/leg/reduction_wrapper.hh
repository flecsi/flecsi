// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_REDUCTION_WRAPPER_HH
#define FLECSI_EXEC_LEG_REDUCTION_WRAPPER_HH

#include "flecsi/exec/fold.hh"
#include "flecsi/run/backend.hh"
#include "flecsi/util/common.hh"
#include "flecsi/util/demangle.hh"

#include <legion.h>

#include <atomic>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>

namespace flecsi {

inline flog::devel_tag reduction_wrapper_tag("reduction_wrapper");

namespace exec {

namespace detail {

struct atomic_base {
  // Expecting that concurrent atomic operations are more likely on a single
  // type, we share one large lock array:
  static constexpr std::uintptr_t locks = 255;
  static inline std::mutex lock[locks];

  static auto lower(std::memory_order o) {
    switch(o) {
      case std::memory_order_relaxed:
        return __ATOMIC_RELAXED;
      case std::memory_order_consume:
        return __ATOMIC_CONSUME;
      case std::memory_order_acquire:
        return __ATOMIC_ACQUIRE;
      case std::memory_order_release:
        return __ATOMIC_RELEASE;
      case std::memory_order_acq_rel:
        return __ATOMIC_ACQ_REL;
      default:
        return __ATOMIC_SEQ_CST;
    }
  }
};
// A simple, abridged version of std::atomic_ref from C++20.
template<class T, class = void>
struct atomic_ref : private atomic_base {
  explicit atomic_ref(T & t) : p(&t) {}
  bool compare_exchange_strong(T & expected,
    T desired,
    std::memory_order = {}) const noexcept {
    const std::unique_lock guard{
      lock[reinterpret_cast<std::uintptr_t>(p) / alignof(T) % locks]};
    const bool fail = std::memcmp(p, &expected, sizeof(T));
    std::memcpy(fail ? &expected : p, fail ? p : &desired, sizeof(T));
    return !fail;
  }

private:
  T * p;
};
// The real implementation for certain built-in types:
template<class T>
struct atomic_ref<T,
  std::enable_if_t<std::is_pointer_v<T> || std::is_integral_v<T>>>
  : private atomic_base {
  explicit atomic_ref(T & t) : p(&t) {}

  bool compare_exchange_strong(T & expected,
    T desired,
    std::memory_order o = std::memory_order_seq_cst) const noexcept {
    return __atomic_compare_exchange_n(
      p, &expected, desired, false, lower(o), lower([o]() {
        switch(o) {
          case std::memory_order_acq_rel:
            return std::memory_order_acquire;
          case std::memory_order_release:
            return std::memory_order_relaxed;
          default:
            return o;
        }
      }()));
  }

private:
  T * p;
};
} // namespace detail

namespace fold {
/// \addtogroup legion-execution
/// \{

// Adapts our interface to Legion's.
template<class R, class T>
struct custom_wrap {

  typedef T LHS, RHS;

  template<bool E>
  static void apply(LHS & a, RHS b) {
    if constexpr(E)
      a = R::combine(a, b);
    else {
      LHS rd{};
      detail::atomic_ref<LHS> r(a);
      while(!r.compare_exchange_strong(
        rd, R::combine(rd, b), std::memory_order_relaxed))
        ;
    }
  }
  // Legion actually requires this additional interface:
  static constexpr const T & identity =
    detail::identity_traits<R>::template value<T>;
  template<bool E>
  static void fold(RHS & a, RHS b) {
    apply<E>(a, b);
  }

private:
  static void init() {
    Legion::Runtime::register_reduction_op<custom_wrap>(REDOP_ID);
  }

public:
  static inline const Legion::ReductionOpID REDOP_ID =
    (run::context::register_init(init),
      Legion::Runtime::generate_static_reduction_id());
};

namespace detail {
template<class T>
struct legion_reduction;

template<>
struct legion_reduction<sum> {
  template<class T>
  using type = Legion::SumReduction<T>;
};

template<>
struct legion_reduction<product> {
  template<class T>
  using type = Legion::ProdReduction<T>;
};

template<>
struct legion_reduction<max> {
  template<class T>
  using type = Legion::MaxReduction<T>;
};

template<>
struct legion_reduction<min> {
  template<class T>
  using type = Legion::MinReduction<T>;
};

template<class R, class T, class = void>
struct wrap {
  using type = custom_wrap<R, T>;
};

template<class R, class T>
struct wrap<R,
  T,
  decltype(void(legion_reduction<R>::template type<T>::REDOP_ID))> {
  using type = typename legion_reduction<R>::template type<T>;
};
} // namespace detail

template<class R, class T>
using wrap = typename detail::wrap<R, T>::type;

/// \}
} // namespace fold
} // namespace exec
} // namespace flecsi

#endif
