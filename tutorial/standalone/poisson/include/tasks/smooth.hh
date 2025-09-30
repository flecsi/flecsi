#ifndef POISSON_TASKS_SMOOTH_HH
#define POISSON_TASKS_SMOOTH_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void red(flecsi::exec::accelerator,
  mesh::accessor<ro> m,
  field<double>::accessor<rw, ro> ua,
  field<double>::accessor<ro, na> fa) noexcept;
void black(flecsi::exec::accelerator,
  mesh::accessor<ro> m,
  field<double>::accessor<rw, ro> ua,
  field<double>::accessor<ro, na> fa) noexcept;

} // namespace task
} // namespace poisson

#endif
