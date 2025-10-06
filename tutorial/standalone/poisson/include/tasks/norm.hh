#ifndef POISSON_TASKS_NORM_HH
#define POISSON_TASKS_NORM_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

double diff(mesh::accessor<ro> m,
  field<double>::accessor<ro, na> aa,
  field<double>::accessor<ro, na> ba) noexcept;

double scale(mesh::accessor<ro> m, double sum) noexcept;

void discrete_operator(mesh::accessor<ro> m,
  field<double>::accessor<ro, ro> ua,
  field<double>::accessor<wo, na> Aua) noexcept;

} // namespace task
} // namespace poisson

#endif
