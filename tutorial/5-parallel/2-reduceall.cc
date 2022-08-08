// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "canonical.hh"
#include "control.hh"

// this tutorial is based on a 04-data/3-dence.cc tutorial example
// here we will add several forall / parallel_for interfaces

using namespace flecsi;

canon::slot canonical;
canon::cslot coloring;

const field<double>::definition<canon, canon::cells> pressure;

void
init(canon::accessor<wo> t, field<double>::accessor<wo> p) {
  std::size_t off{0};
  for(const auto c : t.cells()) {
    p[c] = (off++) * 2.0;
  } // for
} // init

void
reduce1(canon::accessor<ro> t, field<double>::accessor<ro> p) {
  auto res = reduceall(c, up, t.cells(), exec::fold::max, double, "reduce1") {
    up(p[c]);
  }; // forall

  flog_assert(res == 6.0, res << " != 6.0");

} // reduce1

void
reduce2(canon::accessor<ro> t, field<double>::accessor<ro> p) {
  auto res = flecsi::exec::parallel_reduce<exec::fold::max, double>(
    t.cells(),
    FLECSI_LAMBDA(auto c, auto up) { up(p[c]); },
    std::string("reduce2"));

  flog_assert(res == 6.0, res << " != 6.0");

} // reduce2

void
print(canon::accessor<ro> t, field<double>::accessor<ro> p) {
  std::size_t off{0};
  for(auto c : t.cells()) {
    flog(info) << "cell " << off++ << " has pressure " << p[c] << std::endl;
  } // for
} // print

int
advance(control_policy &) {
  coloring.allocate("test.txt");
  canonical.allocate(coloring.get());

  auto pf = pressure(canonical);

  // cpu task, default
  execute<init>(canonical, pf);
  execute<reduce1, default_accelerator>(canonical, pf);
  execute<reduce2, default_accelerator>(canonical, pf);
  // cpu_task
  execute<print>(canonical, pf);

  return 0;
}
control::action<advance, cp::advance> advance_action;
