#ifndef POISSON_SPECIALIZATION_TYPES_HH
#define POISSON_SPECIALIZATION_TYPES_HH

#include <flecsi/data.hh>

namespace poisson {

static constexpr flecsi::partition_privilege_t na = flecsi::na;
static constexpr flecsi::partition_privilege_t ro = flecsi::ro;
static constexpr flecsi::partition_privilege_t wo = flecsi::wo;
static constexpr flecsi::partition_privilege_t rw = flecsi::rw;

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

} // namespace poisson

#endif
