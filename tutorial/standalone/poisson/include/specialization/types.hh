#ifndef POISSON_SPECIALIZATION_TYPES_HH
#define POISSON_SPECIALIZATION_TYPES_HH

#include <flecsi/data.hh>

namespace poisson {

using flecsi::na, flecsi::ro, flecsi::wo, flecsi::rw;

template<typename T, flecsi::data::layout L = flecsi::data::layout::dense>
using field = flecsi::field<T, L>;

} // namespace poisson

#endif
