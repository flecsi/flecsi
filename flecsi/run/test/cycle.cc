#include "package_a.hh"
#include "package_b.hh"
#include "package_c.hh"

#include "flecsi/runtime.hh"
#include "flecsi/util/unit.hh"

int
main() {
  flecsi::run::argv argv{""}; // Boost doesn't like argc==0
#if FLECSI_BACKEND == FLECSI_BACKEND_legion
  {
    flecsi::run::config cfg;
    flecsi::util::unit::accelerator_config(cfg);
    if(auto i = cfg.legion.begin(), e = cfg.legion.end(); i != e)
      while(++i != e) {
        argv.push_back("--Xbackend");
        argv.push_back(std::move(*i));
      }
  }
#endif

  auto status =
    flecsi::initialize(argv.size(), flecsi::run::pointers(argv).data());
  status = control::check_status(status);

  if(status != flecsi::run::status::success) {
    return status < flecsi::run::status::clean ? 0 : status;
  } // if

  flecsi::flog::add_output_stream("clog", std::clog, true);

  status = flecsi::start(control::execute);

  flecsi::finalize();

  return status;
} // main
