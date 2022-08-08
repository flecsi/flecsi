// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

using namespace flecsi;

/*
  Task with no arguments.
 */

void
task(exec::launch_domain) {
  flog(info) << "Hello World from color " << color() << " of " << colors()
             << std::endl;
}

/*
  Advance control point.
 */

int
advance(control_policy &) {
  exec::launch_domain ld{4};

  execute<task>(ld);

  return 0;
}
control::action<advance, cp::advance> advance_action;
