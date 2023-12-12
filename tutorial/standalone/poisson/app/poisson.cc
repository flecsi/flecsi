#include "poisson.hh"
#include "analyze.hh"
#include "finalize.hh"
#include "initialize.hh"
#include "options.hh"
#include "problem.hh"
#include "solve.hh"
#include "specialization/control.hh"
#include "state.hh"

#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

int
main(int argc, char ** argv) {
  flecsi::util::annotation::rguard<main_region> main_guard;

  flecsi::getopt()(argc, argv);
  const flecsi::run::dependencies_guard dg;
  flecsi::run::config cfg{};
#if FLECSI_BACKEND == FLECSI_BACKEND_legion &&                                 \
  (defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP))
  cfg.legion = {"", "-ll:gpu", "1"};
#endif
  const flecsi::runtime run(cfg);
  flecsi::flog::add_output_stream("clog", std::clog, true);
  return run.control<poisson::control>();
} // main
