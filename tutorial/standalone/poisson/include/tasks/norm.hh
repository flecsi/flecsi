/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include "specialization/mesh.hh"

namespace poisson {
namespace task {

double diff(mesh::accessor<ro> m,
  field<double>::accessor<ro, ro> aa,
  field<double>::accessor<ro, ro> ba);

double scale(mesh::accessor<ro> m, double sum);

void discrete_operator(mesh::accessor<ro> m,
  field<double>::accessor<ro, ro> ua,
  field<double>::accessor<rw, ro> Aua);

} // namespace task
} // namespace poisson
