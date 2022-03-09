// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#if defined(SUBCYCLE)
#include "2-subcycle.hh"
#else
#include "2-cycle.hh"
#endif

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

using namespace cycle;

int
initialize() {
  flog(info) << "initialize" << std::endl;
  return 0;
}
control::action<initialize, cp::initialize> initialize_action;

int
advance() {
  flog(info) << "advance" << std::endl;
  return 0;
}
control::action<advance, cp::advance> advance_action;

#if defined(SUBCYCLE)
int
advance2() {
  flog(info) << "advance2" << std::endl;
  return 0;
}
control::action<advance2, cp::advance2> advance2_action;

#endif

int
analyze() {
  flog(info) << "analyze" << std::endl;
  return 0;
}
control::action<analyze, cp::analyze> analyze_action;

int
finalize() {
  flog(info) << "finalize" << std::endl;
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

  flecsi::log::add_output_stream("clog", std::clog, true);

  status = flecsi::start(control::execute);

  flecsi::finalize();

  return status;
} // main
