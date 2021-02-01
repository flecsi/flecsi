/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include "specialization/control.hh"

namespace poisson {
namespace action {

int solve();
inline control::action<solve, cp::solve> solve_action;

} // namespace action
} // namespace poisson
