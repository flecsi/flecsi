/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#include "problem.hh"
#include "poisson.hh"
#include "state.hh"
#include "tasks/init.hh"
#include "tasks/io.hh"

#include <flecsi/execution.hh>

using namespace flecsi;

int
poisson::action::problem() {
  annotation::rguard<problem_region> guard;
  execute<task::eggcarton>(m, ud(m), fd(m), sd(m), Aud(m));
  execute<task::io, flecsi::mpi>(m, ud(m), "init");
  execute<task::io, flecsi::mpi>(m, sd(m), "actual");
  flog::flush();
  return 0;
} // problem
