// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_TEST_PACKAGE_A_HH
#define FLECSI_RUN_TEST_PACKAGE_A_HH

#include "cycle.hh"

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

namespace package_a {

//----------------------------------------------------------------------------//
// Internal
//----------------------------------------------------------------------------//

inline void
internal_init(control_policy &) {
  flog(info) << "A internal init" << std::endl;
}

inline control::meta<internal_init, cp::init_internal> internal_action;

//----------------------------------------------------------------------------//
// Initialization
//----------------------------------------------------------------------------//

inline void
init(control_policy &) {
  flog(info) << "A init" << std::endl;
}

inline control::action<init, cp::initialization> init_action;

//----------------------------------------------------------------------------//
// Advance
//----------------------------------------------------------------------------//

inline void
task1() {
  flog(trace) << "A task 1" << std::endl;
}

inline void
task2() {
  flog(trace) << "A task 2" << std::endl;
}

inline void
internal_advance(control_policy &) {
  flog(info) << "A advance internal" << std::endl;
}

inline control::meta<internal_advance, cp::advance_internal>
  advance_internal_action;

inline void
advance(control_policy &) {
  flog(info) << "A advance" << std::endl;
  flecsi::execute<task1>();
  flecsi::execute<task2>();
}

inline control::action<advance, cp::advance> advance_action;

//----------------------------------------------------------------------------//
// Subcycle
//----------------------------------------------------------------------------//

inline void
subcycle_task() {
  flog(trace) << "A subcycle task" << std::endl;
}

inline void
subcycle(control_policy &) {
  flecsi::execute<subcycle_task>();
}

inline control::action<subcycle, cp::advance_subcycle> subcycle_action;

//----------------------------------------------------------------------------//
// Analysis
//----------------------------------------------------------------------------//

inline void
task3() {
  flog(trace) << "A task 3" << std::endl;
}

inline void
task4() {
  flog(trace) << "A task 4" << std::endl;
}

inline void
analyze(control_policy &) {
  flog(info) << "A analyze" << std::endl;
  flecsi::execute<task3>();
  flecsi::execute<task4>();
}

inline control::action<analyze, cp::analyze> analyze_action;

//----------------------------------------------------------------------------//
// I/O
//----------------------------------------------------------------------------//

inline void
io(control_policy &) {
  flog(info) << "A I/0" << std::endl;
}

inline control::action<io, cp::io> io_action;

//----------------------------------------------------------------------------//
// Mesh
//----------------------------------------------------------------------------//

inline void
mesh(control_policy &) {
  flog(info) << "A mesh" << std::endl;
}

inline control::action<mesh, cp::mesh> mesh_action;

//----------------------------------------------------------------------------//
// Finalize
//----------------------------------------------------------------------------//

inline void
finalize(control_policy &) {
  flog(info) << "A finalize" << std::endl;
}

inline control::action<finalize, cp::finalization> finalize_action;

} // namespace package_a

#endif
