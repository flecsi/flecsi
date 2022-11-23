#include <flecsi/execution.hh>
#include <flecsi/util/unit.hh>

using namespace flecsi;
using flecsi::util::unit::control;

int
main(int argc, char ** argv) {
  auto status = flecsi::initialize(argc, argv);
  status = control::check_status(status);

  if(status != flecsi::run::status::success) {
    return status < flecsi::run::status::clean ? 0 : status;
  } // if

  log::state::instance->config_stream().add_buffer("flog", std::clog, true);

  status = flecsi::start(control::execute);

  flecsi::finalize();

  return status;
} // main
