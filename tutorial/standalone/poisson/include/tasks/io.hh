#ifndef POISSON_TASKS_IO_HH
#define POISSON_TASKS_IO_HH

#include "specialization/mesh.hh"

#include <string>

namespace poisson {
namespace task {

void io(flecsi::exec::cpu,
  mesh::accessor<ro> m,
  field<double>::accessor<ro, na> ua,
  std::string filebase);

} // namespace task
} // namespace poisson

#endif
