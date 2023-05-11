#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

int
main(int argc, char ** argv) {
  flecsi::run::arguments args(argc, argv);
  const flecsi::run::dependencies_guard dg(args.dep);
  const flecsi::runtime run(args.cfg);
  flecsi::flog::add_output_stream("clog", std::clog, true);
  return run.main<control>(args.act);
} // main
