#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

using namespace flecsi;

// Task with no arguments.

void
task(exec::cpu s) {
  flog(info) << "Hello World from process: " << s.launch().index << std::endl;
}

// Advance control point.

void
advance(control_policy &) {
  execute<task, mpi>(exec::on);
} // advance()
control::action<advance, cp::advance> advance_action;
