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
  auto & s = cp.scheduler();
  s.execute<task::eggcarton>(
    exec::on, *cp.m, ud(*cp.m), fd(*cp.m), sd(*cp.m), Aud(*cp.m));
  execute<task::io, flecsi::mpi>(exec::on, *cp.m, ud(*cp.m), "init");
  execute<task::io, flecsi::mpi>(exec::on, *cp.m, sd(*cp.m), "actual");

  flog::flush();
} // problem
