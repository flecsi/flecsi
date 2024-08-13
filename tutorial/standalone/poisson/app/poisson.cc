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
  flecsi::run::config cfg;
#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  cfg.legion = {"", "-ll:gpu", "1"};
#elif defined(KOKKOS_ENABLE_OPENMP)
  cfg.legion = {"", "-ll:ocpu", "1", "-ll:onuma", "0"};
#endif
#endif
  const flecsi::runtime run(cfg);
  flecsi::flog::add_output_stream("clog", std::clog, true);
  return run.control<poisson::control>();
} // main
