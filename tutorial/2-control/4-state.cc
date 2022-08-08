// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "4-state.hh"

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

using namespace state;

int
allocate(control_policy & policy) {
  flog(info) << "allocate" << std::endl;

  /*
    Call a method of the control policy to allocate an array of size 10.
   */

  policy.allocate_values(10);

  return 0;
}
control::action<allocate, cp::allocate> allocate_action;

int
initialize(control_policy & policy) {
  flog(info) << "initialize" << std::endl;

  /*
    Access the array through the 'values()' method, and initialize.
   */

  control_policy::int_custom & values = policy.values();

  for(std::size_t i{0}; i < 10; ++i) {
    values[i] = 20 - i;
  } // for

  policy.steps() = 5;

  return 0;
}
control::action<initialize, cp::initialize> initialize_action;

int
advance(control_policy & policy) {
  std::stringstream ss;

  ss << "advance " << policy.step() << std::endl;

  /*
    Access the array through the 'values()' method, and modify.
   */

  control_policy::int_custom & values = policy.values();

  for(std::size_t i{0}; i < 10; ++i) {
    ss << values[i] << " ";
    values[i] = values[i] + 1;
  } // for

  ss << std::endl;

  flog(info) << ss.str();
  return 0;
}
control::action<advance, cp::advance> advance_action;

int
finalize(control_policy & policy) {
  flog(info) << "finalize" << std::endl;

  /*
    Deallocate the array using the control policy interface.
   */

  policy.deallocate_values();

  return 0;
}
control::action<finalize, cp::finalize> finalize_action;

int
main(int argc, char ** argv) {
  auto status = flecsi::initialize(argc, argv);
  status = control::check_status(status);

  if(status != flecsi::run::status::success) {
    return status < flecsi::run::status::clean ? 0 : status;
  }

  flecsi::flog::add_output_stream("clog", std::clog, true);

  status = flecsi::start(control::execute);

  flecsi::finalize();

  return status;
} // main
