#ifndef POISSON_SPECIALIZATION_TYPES_HH
#define POISSON_SPECIALIZATION_TYPES_HH

#include <flecsi/data.hh>

namespace poisson {

inline constexpr flecsi::partition_privilege_t na = flecsi::na, ro = flecsi::ro,
                                               wo = flecsi::wo, rw = flecsi::rw;

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

} // namespace poisson

#endif
