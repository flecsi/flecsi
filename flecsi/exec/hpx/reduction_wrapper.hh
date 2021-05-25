// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_HPX_REDUCTION_WRAPPER_HH
#define FLECSI_DATA_HPX_REDUCTION_WRAPPER_HH

#include "flecsi/exec/fold.hh"
#include "flecsi/run/backend.hh"
#include <flecsi/flog.hh>

namespace flecsi {

inline log::devel_tag reduction_wrapper_tag("reduction_wrapper");

namespace exec {
namespace fold {
/// \addtogroup hpx-execution
/// \{

template<typename R>
struct wrap {
  template<typename T>
  auto operator()(T a, T b) const {
    return R::combine(a, b);
  }

  template<typename T>
  auto operator()(::hpx::serialization::serialize_buffer<T> a,
    ::hpx::serialization::serialize_buffer<T> const & b) const {
    flog_assert(a.size() == b.size(), "arguments must be of same size");

    for(std::size_t i = 0; i != a.size(); ++i) {
      a[i] = R::combine(a[i], b[i]);
    }
    return a;
  }
};

/// \}
} // namespace fold
} // namespace exec
} // namespace flecsi

#endif
