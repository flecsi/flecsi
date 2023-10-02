#include "analyze.hh"
#include "poisson.hh"
#include "state.hh"
#include "tasks/norm.hh"

#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include <cmath>

using namespace flecsi;

void
poisson::action::analyze(control_policy & cp) {
  util::annotation::rguard<analyze_region> guard;
  double sum =
    reduce<task::diff, exec::fold::sum>(cp.m, ud(cp.m), sd(cp.m)).get();
  sum = execute<task::scale>(cp.m, sum).get();
  const double l2 = sqrt(sum);
  flog(info) << "l2 error: " << l2 << std::endl;
} // analyze
