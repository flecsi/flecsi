// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "canonical.hh"
#include "control.hh"

using namespace flecsi;

canon::slot canonical, cp;
canon::cslot coloring;

const field<double>::definition<canon, canon::cells> pressure;

void
init(canon::accessor<ro> t, field<double>::accessor<wo> p) {
  std::size_t off{0};
  for(const auto c : t.cells()) {
    p[c] = (off++) * 2.0;
  } // for
} // init

void
copy(field<double>::accessor<ro> src, field<double>::accessor<rw> dest) {
  auto s = src.span();
  std::copy(s.begin(), s.end(), dest.span().begin());
}

void
print(canon::accessor<ro> t, field<double>::accessor<ro> p) {
  std::size_t off{0};
  for(auto c : t.cells()) {
    flog(info) << "cell " << off++ << " has pressure " << p[c] << std::endl;
  } // for
} // print

void
advance(control_policy &) {
  coloring.allocate("test.txt");
  canonical.allocate(coloring.get());
  cp.allocate(coloring.get());

  auto pf = pressure(canonical), pf2 = pressure(cp);

  execute<init>(canonical, pf);
  execute<copy>(pf, pf2);
  execute<print>(cp, pf2);
}
control::action<advance, cp::advance> advance_action;
