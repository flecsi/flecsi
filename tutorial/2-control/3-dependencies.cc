#include "3-dependencies.hh"
#include "3-actions.hh"

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"

using namespace dependencies;

// Add dependencies a -> c, and c -> d. These dependencies are added here to
// demonstrate that action relationships do not have to be defined in a single
// source file.

const auto dep_ca = package_c_action.add(package_a_action);
const auto dep_dc = package_d_action.add(package_c_action);

int
main() {
  const flecsi::run::dependencies_guard dg;
  const flecsi::runtime run;
  flecsi::flog::add_output_stream("clog", std::clog, true);
  return run.control<control>();
} // main
