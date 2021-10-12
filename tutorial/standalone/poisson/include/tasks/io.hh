/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include "specialization/mesh.hh"

#include <string>

namespace poisson {
namespace task {

void io(mesh::accessor<ro> m,
  field<double>::accessor<ro, ro> ua,
  std::string filebase);

void print(mesh::accessor<ro> m, field<double>::accessor<ro, ro> fa);

} // namespace task
} // namespace poisson
