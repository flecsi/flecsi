#include "finalize.hh"
#include "state.hh"
#include "tasks/io.hh"

using namespace flecsi;

void
poisson::action::finalize(control_policy & cp) {
  execute<task::io, mpi>(cp.m, ud(cp.m), "solution");
} // finalize
