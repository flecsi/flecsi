// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_EXEC_LEG_FOLD_HH
#define FLECSI_EXEC_LEG_FOLD_HH

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

namespace flecsi::exec::fold {
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
      std::atomic_ref<LHS> r(a);
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
} // namespace flecsi::exec::fold

#endif
