#include <flecsi/execution.hh>
#include <flecsi/util/unit.hh>

using flecsi::util::unit::control;

int
main(int argc, char ** argv) {

  auto status = flecsi::initialize(argc, argv);
  status = control::check_status(status);

  if(status != flecsi::run::status::success) {
    return status < flecsi::run::status::clean ? 0 : status;
  } // if

  status = flecsi::start(control::execute);

  flecsi::finalize();

  return status;
} // main
