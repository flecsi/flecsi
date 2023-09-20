#include "4-state.hh"

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

using namespace state;

void
allocate(control_policy & policy) {
  flog(info) << "allocate" << std::endl;

  // Call a method of the control policy to allocate an array of size 10.

  policy.allocate_values(10);
}
control::action<allocate, cp::allocate> allocate_action;

void
initialize(control_policy & policy) {
  flog(info) << "initialize" << std::endl;

  // Access the array through the 'values()' method, and initialize.

  control_policy::int_custom & values = policy.values();

  for(std::size_t i{0}; i < 10; ++i) {
    values[i] = 20 - i;
  } // for

  policy.steps() = 5;
}
control::action<initialize, cp::initialize> initialize_action;

void
advance(control_policy & policy) {
  std::stringstream ss;

  ss << "advance " << policy.step() << std::endl;

  // Access the array through the 'values()' method, and modify.

  control_policy::int_custom & values = policy.values();

  for(std::size_t i{0}; i < 10; ++i) {
    ss << values[i] << " ";
    values[i] = values[i] + 1;
  } // for

  ss << std::endl;

  flog(info) << ss.str();
}
control::action<advance, cp::advance> advance_action;

void
finalize(control_policy & policy) {
  flog(info) << "finalize" << std::endl;

  // Deallocate the array using the control policy interface.

  policy.deallocate_values();
}
control::action<finalize, cp::finalize> finalize_action;

int
main(int argc, char ** argv) {
  flecsi::run::arguments args(argc, argv);
  const flecsi::run::dependencies_guard dg(args.dep);
  const flecsi::runtime run(args.cfg);
  flecsi::flog::add_output_stream("clog", std::clog, true);
  return run.main<control>(args.act);
} // main
