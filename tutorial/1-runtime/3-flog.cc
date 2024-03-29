#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/runtime.hh>

#include <fstream>

using namespace flecsi;

// Create some tags to control output.

flog::tag tag1("tag1");
flog::tag tag2("tag2");

flecsi::program_option<std::string> tag("Logging",
  "flog",
  "Specify the flog tag to enable.",
  {{flecsi::option_default, "all"}});

int
top_level_action() {

  // This output will always be generated because it is not scoped within a tag
  // guard.

  flog(trace) << "Trace level output" << std::endl;
  flog(info) << "Info level output" << std::endl;
  flog(warn) << "Warn level output" << std::endl;
  flog(error) << "Error level output" << std::endl;

  // This output will appear only if 'tag1' is enabled.

  {
    flog::guard guard(tag1);
    flog(trace) << "Trace level output (in tag1 guard)" << std::endl;
    flog(info) << "Info level output (in tag1 guard)" << std::endl;
    flog(warn) << "Warn level output (in tag1 guard)" << std::endl;
    flog(error) << "Error level output (in tag1 guard)" << std::endl;
  } // scope

  // This output will be generated only if 'tag2' is enabled.

  {
    flog::guard guard(tag2);
    flog(trace) << "Trace level output (in tag2 guard)" << std::endl;
    flog(info) << "Info level output (in tag2 guard)" << std::endl;
    flog(warn) << "Warn level output (in tag2 guard)" << std::endl;
    flog(error) << "Error level output (in tag2 guard)" << std::endl;
  } // scope

  return 0;
} // top_level_action

int
main(int argc, char ** argv) {
  std::ofstream log_file; // to outlive runtime
  flecsi::getopt()(argc, argv);
  const run::dependencies_guard dg;
  // If FLECSI_ENABLE_FLOG is enabled, FLOG will automatically be initialized
  // when the runtime is created.
  run::config cfg;
  cfg.flog.tags = {tag};
  const runtime run(cfg);

  // In order to see or capture any output from FLOG, the user must add at least
  // one output stream. The function flog::add_output_stream provides an
  // interface for adding output streams to FLOG. The FleCSI runtime must have
  // been initialized before this function can be invoked.

  // Add the standard log descriptor to FLOG's buffers.

  flog::add_output_stream("clog", std::clog, true);

  // Add an output file to FLOG's buffers.

  log_file.open("output.txt");
  flog::add_output_stream("log file", log_file);

  return run.control<run::call>(top_level_action);
} // main
