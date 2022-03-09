/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#ifndef POISSON_TASKS_IO_HH
#define POISSON_TASKS_IO_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void io(mesh::accessor<ro> m, field<double>::accessor<ro, ro> ua);

} // namespace task
} // namespace poisson

#endif
