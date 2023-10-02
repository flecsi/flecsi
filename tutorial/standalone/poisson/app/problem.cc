#include "problem.hh"
#include "poisson.hh"
#include "state.hh"
#include "tasks/init.hh"
#include "tasks/io.hh"

#include <flecsi/execution.hh>

using namespace flecsi;

void
poisson::action::problem(control_policy & cp) {
  util::annotation::rguard<problem_region> guard;
  execute<task::eggcarton, default_accelerator>(
    cp.m, ud(cp.m), fd(cp.m), sd(cp.m), Aud(cp.m));
  execute<task::io, flecsi::mpi>(cp.m, ud(cp.m), "init");
  execute<task::io, flecsi::mpi>(cp.m, sd(cp.m), "actual");

  // This can be used for debugging
#if 0
  execute<task::redblack>(cp.m, test(cp.m));
  execute<task::print>(cp.m, test(cp.m));
#endif

  flog::flush();
} // problem
