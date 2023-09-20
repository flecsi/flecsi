#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

using namespace flecsi;

// Task with no arguments.

void
task(exec::launch_domain) {
  flog(info) << "Hello World from color " << color() << " of " << colors()
             << std::endl;
}

// Advance control point.

void
advance(control_policy &) {
  exec::launch_domain ld{4};

  execute<task>(ld);
} // advance()
control::action<advance, cp::advance> advance_action;
