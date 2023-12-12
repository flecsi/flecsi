#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

int
main() {
  const flecsi::run::dependencies_guard dg;
  const flecsi::runtime run;
  flecsi::flog::add_output_stream("clog", std::clog, true);
  return run.control<control>();
} // main
