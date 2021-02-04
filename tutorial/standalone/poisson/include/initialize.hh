/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include "specialization/control.hh"

namespace poisson {
namespace action {

int init_mesh();
inline control::action<init_mesh, cp::initialize> init_mesh_action;

} // namespace action
} // namespace poisson
