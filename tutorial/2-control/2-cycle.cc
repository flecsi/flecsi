#if defined(SUBCYCLE)
#include "2-subcycle.hh"
#else
#include "2-cycle.hh"
#endif

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

using namespace cycle;

void
initialize(control_policy &) {
  flog(info) << "initialize" << std::endl;
}
control::action<initialize, cp::initialize> initialize_action;

void
advance(control_policy &) {
  flog(info) << "advance" << std::endl;
}
control::action<advance, cp::advance> advance_action;

#if defined(SUBCYCLE)
void
advance2(control_policy &) {
  flog(info) << "advance2" << std::endl;
}
control::action<advance2, cp::advance2> advance2_action;

#endif

void
analyze(control_policy &) {
  flog(info) << "analyze" << std::endl;
}
control::action<analyze, cp::analyze> analyze_action;

void
finalize(control_policy &) {
  flog(info) << "finalize" << std::endl;
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
