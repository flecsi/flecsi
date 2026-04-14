#ifndef POISSON_TASKS_NORM_HH
#define POISSON_TASKS_NORM_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

double diff(mesh::accessor<flecsi::ro> m,
  flecsi::field<double>::accessor<flecsi::ro, flecsi::na> aa,
  flecsi::field<double>::accessor<flecsi::ro, flecsi::na> ba) noexcept;

double scale(mesh::accessor<flecsi::ro> m, double sum) noexcept;

void discrete_operator(mesh::accessor<flecsi::ro> m,
  flecsi::field<double>::accessor<flecsi::ro, flecsi::ro> ua,
  flecsi::field<double>::accessor<flecsi::wo, flecsi::na> Aua) noexcept;

} // namespace task
} // namespace poisson

#endif
