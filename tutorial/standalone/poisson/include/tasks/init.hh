#ifndef POISSON_TASKS_INIT_HH
#define POISSON_TASKS_INIT_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void eggcarton(flecsi::exec::accelerator,
  mesh::accessor<flecsi::ro> m,
  flecsi::field<double>::accessor<flecsi::wo, flecsi::na> ua,
  flecsi::field<double>::accessor<flecsi::wo, flecsi::na> fa,
  flecsi::field<double>::accessor<flecsi::wo, flecsi::na> sa,
  flecsi::field<double>::accessor<flecsi::wo, flecsi::na> Aua) noexcept;

} // namespace task
} // namespace poisson

#endif
