#ifndef POISSON_TASKS_SMOOTH_HH
#define POISSON_TASKS_SMOOTH_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void red(mesh::accessor<ro> m,
  field<double>::accessor<rw, ro> ua,
  field<double>::accessor<ro, ro> fa);
void black(mesh::accessor<ro> m,
  field<double>::accessor<rw, ro> ua,
  field<double>::accessor<ro, ro> fa);

} // namespace task
} // namespace poisson

#endif
