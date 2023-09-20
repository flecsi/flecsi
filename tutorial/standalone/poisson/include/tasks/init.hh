#ifndef POISSON_TASKS_INIT_HH
#define POISSON_TASKS_INIT_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void eggcarton(mesh::accessor<ro> m,
  field<double>::accessor<wo, na> ua,
  field<double>::accessor<wo, na> fa,
  field<double>::accessor<wo, na> sa,
  field<double>::accessor<wo, na> Aua);

void constant(mesh::accessor<ro> m,
  field<double>::accessor<wo, na> fa,
  double value);

void redblack(mesh::accessor<ro> m, field<double>::accessor<wo, na> fa);

} // namespace task
} // namespace poisson

#endif
