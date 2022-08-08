/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#include "solve.hh"
#include "options.hh"
#include "poisson.hh"
#include "state.hh"
#include "tasks/norm.hh"
#include "tasks/smooth.hh"

#include <flecsi/execution.hh>
#include <flecsi/flog.hh>
#include <flecsi/util/annotation.hh>

using namespace flecsi;

int
poisson::action::solve(control_policy &) {
  annotation::rguard<solve_region> guard;
  double err{std::numeric_limits<double>::max()};

  std::size_t sub{100};
  std::size_t ita{0};

  // The tracing utility traces and optimizes loops during a Legion run. In this
  // case a do-while loop will be analysed at every calls to the solve action.
  // The trace is created before the loop; note that the object is static so the
  // identifier of the trace remains the same across the calls to the solver
  // action. The following call to the skip method ensure that the first loop of
  // the do-while loop will not be traced which is required for specific Legion
  // implementation. Inside the loop a guard is created. This creation starts
  // the tracing and its destruction at the end of the do-while loop stops the
  // trace.
  static exec::trace t;
  t.skip();
  do {
    auto g = t.make_guard();
    // Annotation to time each cycle of the poisson solve
    annotation::guard<annotation::execution, annotation::detail::low> aguard(
      "poisson-cycle");
    for(std::size_t i{0}; i < sub; ++i) {
      execute<task::red, default_accelerator>(m, ud(m), fd(m));
      execute<task::black, default_accelerator>(m, ud(m), fd(m));
    } // for
    ita += sub;

    execute<task::discrete_operator>(m, ud(m), Aud(m));
    auto residual = reduce<task::diff, exec::fold::sum>(m, fd(m), Aud(m));
    err = std::sqrt(residual.get());
    flog(info) << "residual: " << err << " (" << ita << " iterations)"
               << std::endl;
    flog::flush();
  } while(err > error_tol.value() && ita < max_iterations.value());

  return 0;
} // solve
