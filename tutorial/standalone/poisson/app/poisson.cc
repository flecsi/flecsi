#include "poisson.hh"
#include "analyze.hh"
#include "finalize.hh"
#include "initialize.hh"
#include "options.hh"
#include "problem.hh"
#include "solve.hh"
#include "specialization/control.hh"
#include "state.hh"

#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

int
main(int argc, char ** argv) {
  flecsi::util::annotation::rguard<main_region> main_guard;

  flecsi::run::arguments args(argc, argv);
  const flecsi::run::dependencies_guard dg(args.dep);
  const flecsi::runtime run(args.cfg);
  flecsi::flog::add_output_stream("clog", std::clog, true);
  return run.main<poisson::control>(args.act);
} // main
