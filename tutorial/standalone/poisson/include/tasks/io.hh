#ifndef POISSON_TASKS_IO_HH
#define POISSON_TASKS_IO_HH

#include "specialization/mesh.hh"

#include <string>

namespace poisson {
namespace task {

void io(flecsi::exec::cpu,
  mesh::accessor<flecsi::ro> m,
  flecsi::field<double>::accessor<flecsi::ro, flecsi::na> ua,
  const std::string & filebase) noexcept;

} // namespace task
} // namespace poisson

#endif
