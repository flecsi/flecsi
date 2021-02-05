/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void eggcarton(mesh::accessor<ro> m,
  field<double>::accessor<wo, ro> ua,
  field<double>::accessor<wo, ro> fa,
  field<double>::accessor<wo, ro> sa);

} // namespace task
} // namespace poisson
