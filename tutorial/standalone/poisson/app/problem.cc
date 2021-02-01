/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#include "problem.hh"
#include "state.hh"
#include "tasks/init.hh"

#include <flecsi/execution.hh>

using namespace flecsi;

int
poisson::action::problem() {
  execute<task::eggcarton>(m, ud(m), fd(m), sd(m));
  return 0;
} // problem
