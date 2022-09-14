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
task() {
  flog(info) << "Hello World from process: " << process() << std::endl;
}

/*
  Advance control point.
 */

void
advance(control_policy &) {
  execute<task, mpi>();
}
control::action<advance, cp::advance> advance_action;
