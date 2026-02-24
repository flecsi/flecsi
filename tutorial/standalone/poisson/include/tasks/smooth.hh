#ifndef POISSON_TASKS_SMOOTH_HH
#define POISSON_TASKS_SMOOTH_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void red(flecsi::exec::accelerator,
  mesh::accessor<flecsi::ro> m,
  flecsi::field<double>::accessor<flecsi::rw, flecsi::ro> ua,
  flecsi::field<double>::accessor<flecsi::ro, flecsi::na> fa) noexcept;
void black(flecsi::exec::accelerator,
  mesh::accessor<flecsi::ro> m,
  flecsi::field<double>::accessor<flecsi::rw, flecsi::ro> ua,
  flecsi::field<double>::accessor<flecsi::ro, flecsi::na> fa) noexcept;

} // namespace task
} // namespace poisson

#endif
