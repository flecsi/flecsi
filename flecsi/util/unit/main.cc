#include <flecsi/runtime.hh>
#include <flecsi/util/unit.hh>

using namespace flecsi;
using flecsi::util::unit::control;

namespace {
const control::action<util::unit::control_policy::exit,
  util::unit::test_control_points::exit>
  exit;

program_option<int>
  flog_process("Testing", "flog-process", "Process which should log.");
} // namespace

namespace flecsi::util::unit {
task_local<test_output_t> test_output_t::current_output;
}

int
main(int argc, char ** argv) {
  flecsi::getopt()(argc, argv);
  const run::dependencies_guard dg;
  run::config cfg;
  util::unit::accelerator_config(cfg);
  if(flog_process.has_value())
    cfg.flog.process = flog_process;
  const runtime run(cfg);
  flecsi::flog::add_output_stream("flog", std::clog, true);
  return run.control<control>();
} // main
