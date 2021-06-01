/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#include "poisson.hh"
#include "solve.hh"
#include "state.hh"
#include "tasks/norm.hh"
#include "tasks/smooth.hh"

#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

using namespace flecsi;

struct solve_region : annotation::region<user_execution> {
  inline static const std::string name{"solve"};
};

int
poisson::action::solve() {
  annotation::rguard<solve_region> guard;
  double err{std::numeric_limits<double>::max()};

  std::size_t sub{500};
  std::size_t ita{0};
  do {
    for(std::size_t i{0}; i < sub; ++i) {
      execute<task::smooth>(m, ud(m), fd(m));
    } // for
    ita += sub;

    execute<task::discrete_operator>(m, ud(m), Aud(m));
    auto residual = reduce<task::diff, exec::fold::sum>(m, fd(m), Aud(m));
    err = std::sqrt(residual.get());
    flog(info) << "residual: " << err << " (" << ita << " iterations)"
               << std::endl;
    log::flush();
  } while(err > 10e-05);
  return 0;
} // solve
