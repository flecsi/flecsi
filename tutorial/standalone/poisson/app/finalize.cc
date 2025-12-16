#include "finalize.hh"
#include "state.hh"
#include "tasks/io.hh"

using namespace flecsi;

void
poisson::action::finalize(control_policy & cp) {
  cp.scheduler().execute<task::io>(exec::on, *cp.m, ud(*cp.m), "solution");
} // finalize
