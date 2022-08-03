#include "problem.hh"
#include "poisson.hh"
#include "state.hh"
#include "tasks/init.hh"

#include <flecsi/execution.hh>

using namespace flecsi;

int
poisson::action::problem() {
  annotation::rguard<problem_region> guard;
  execute<task::eggcarton>(m, ud(m), fd(m), sd(m));
  return 0;
} // problem
