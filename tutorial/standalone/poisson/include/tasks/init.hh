/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#ifndef POISSON_TASKS_INIT_HH
#define POISSON_TASKS_INIT_HH

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void eggcarton(mesh::accessor<ro> m,
  field<double>::accessor<wo, ro> ua,
  field<double>::accessor<wo, ro> fa,
  field<double>::accessor<wo, ro> sa);

} // namespace task
} // namespace poisson

#endif
