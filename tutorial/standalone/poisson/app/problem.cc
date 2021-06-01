/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#include "poisson.hh"
#include "problem.hh"
#include "state.hh"
#include "tasks/init.hh"

#include <flecsi/execution.hh>

using namespace flecsi;
struct problem_region : annotation::region<user_execution> {
  inline static const std::string name{"problem"};
};

int
poisson::action::problem() {
  annotation::rguard<problem_region> guard;
  execute<task::eggcarton>(m, ud(m), fd(m), sd(m));
  return 0;
} // problem
