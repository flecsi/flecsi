/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#ifndef POISSON_SOLVE_HH
#define POISSON_SOLVE_HH

#include "specialization/control.hh"

namespace poisson {
namespace action {

int solve(control_policy &);
inline control::action<solve, cp::solve> solve_action;

} // namespace action
} // namespace poisson

#endif
