/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include "initialize.hh"
#include "specialization/control.hh"

namespace poisson {
namespace action {

int problem();
inline control::action<problem, cp::initialize> problem_action;
inline auto const problem_dep = problem_action.add(init_mesh_action);

} // namespace action
} // namespace poisson
