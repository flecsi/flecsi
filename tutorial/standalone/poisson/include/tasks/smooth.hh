/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

void smooth(mesh::accessor<ro> m,
  field<double>::accessor<rw, ro> ua,
  field<double>::accessor<ro, ro> fa);

} // namespace task
} // namespace poisson
