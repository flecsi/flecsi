// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_RUN_TEST_PACKAGE_B_HH
#define FLECSI_RUN_TEST_PACKAGE_B_HH

#include "cycle.hh"
#include "package_a.hh"

#include "flecsi/flog.hh"

namespace package_b {

//----------------------------------------------------------------------------//
// Advance
//----------------------------------------------------------------------------//

inline void
advance(control_policy &) {
  flog(info) << "B advance" << std::endl;
} // advance

inline control::action<advance, cp::advance> advance_action;
inline const auto dep_a = advance_action.add(package_a::advance_action);

//----------------------------------------------------------------------------//
// Subcycle
//----------------------------------------------------------------------------//

inline void
subcycle_task() {
  flog(trace) << "B subcycle task" << std::endl;
}

inline void
subcycle(control_policy &) {
  flecsi::execute<subcycle_task>();
}

inline control::action<subcycle, cp::advance_subcycle> subcycle_action;
inline const auto dep_a_sub = subcycle_action.add(package_a::subcycle_action);

//----------------------------------------------------------------------------//
// Analyze
//----------------------------------------------------------------------------//

inline void
analyze(control_policy &) {
  flog(info) << "B analyze" << std::endl;
} // analyze

inline control::action<analyze, cp::analyze> analyze_action;

} // namespace package_b

#endif
