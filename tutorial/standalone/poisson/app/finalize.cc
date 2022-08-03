#include "finalize.hh"
#include "state.hh"
#include "tasks/io.hh"

using namespace flecsi;

int
poisson::action::finalize() {
  execute<task::io, mpi>(m, sd(m));
  return 0;
} // finalize
