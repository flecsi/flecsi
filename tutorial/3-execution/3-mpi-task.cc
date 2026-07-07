#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

using namespace flecsi;

void
task(exec::cpu s, flecsi::comm::ref c) noexcept {
  MPI_Barrier(c);
  flog(info) << "Hello World from process: " << s.launch().index << std::endl;
}

// Advance control point.

void
advance(control_policy & p) {
  p.scheduler().execute<task>(exec::on, flecsi::comm::world());
} // advance()
control::action<advance, cp::advance> advance_action;
