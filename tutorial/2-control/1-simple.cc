#include "1-simple.hh"

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

using namespace simple;

// Function definition of an initialize action.

void
initialize(control_policy &) {
  flog(info) << "initialize" << std::endl;
}

// Register the initialize action under the 'initialize' control point.

control::action<initialize, cp::initialize> initialize_action;

// Function definition of an advance action.

void
advance(control_policy &) {
  flog(info) << "advance" << std::endl;
}

// Register the advance action under the 'advance' control point.

control::action<advance, cp::advance> advance_action;

// Function definition of a finalize action.

void
finalize(control_policy &) {
  flog(info) << "finalize" << std::endl;
}

// Register the finalize action under the 'finalize' control point.

control::action<finalize, cp::finalize> finalize_action;

// The main function is similar to previous examples, but with the addition of
// logic to check control-model options.

int
main() {
  const flecsi::run::dependencies_guard dg;
  const flecsi::runtime run;
  flecsi::flog::add_output_stream("clog", std::clog, true);
  // Run the control model.  control::invoke will, in turn,
  // execute all of the cycles, and actions of the control model.
  return run.control<control>();
} // main
