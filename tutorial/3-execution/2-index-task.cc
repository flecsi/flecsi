#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

using namespace flecsi;

// Task with special arguments.

void
task(exec::cpu s, exec::launch_domain) noexcept {
  flog(info) << "Hello World from point task " << s.launch().index << " of "
             << s.launch().size << std::endl;
}

// Advance control point.

void
advance(control_policy & p) {
  exec::launch_domain ld{4};

  p.scheduler().execute<task>(exec::on, ld);
} // advance()
control::action<advance, cp::advance> advance_action;
