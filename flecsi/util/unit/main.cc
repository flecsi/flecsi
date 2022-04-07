// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.
#include <flecsi/execution.hh>
#include <flecsi/util/unit.hh>

int
main(int argc, char ** argv) {

  auto status = flecsi::initialize(argc, argv);
  status = flecsi::unit::control::check_status(status);

  if(status != flecsi::run::status::success) {
    return status < flecsi::run::status::clean ? 0 : status;
  } // if

  status = flecsi::start(flecsi::unit::control::execute);

  flecsi::finalize();

  return status;
} // main
