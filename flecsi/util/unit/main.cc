#include <flecsi/execution.hh>
#include <flecsi/util/unit.hh>

using namespace flecsi;
using flecsi::util::unit::control;

int
main(int argc, char ** argv) {
  run::arguments args(argc, argv);
  const run::dependencies_guard dg(args.dep);
  const runtime run(args.cfg);
  flecsi::flog::add_output_stream("flog", std::clog, true);
  return run.main<control>(args.act);
} // main
