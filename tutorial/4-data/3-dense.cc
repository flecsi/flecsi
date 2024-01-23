#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "../3-execution/control.hh"
#include "canonical.hh"

using namespace flecsi;

const field<double>::definition<canon, canon::cells> pressure;

void
init(canon::accessor<ro> t, field<double>::accessor<wo> p) {
  std::size_t off{0};
  for(const auto c : t.cells()) {
    p[c] = (off++) * 2.0;
  } // for
} // init

void
copy(field<double>::accessor<ro> src, field<double>::accessor<wo> dest) {
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
  canon::slot canonical, cp;
  canon::mpi_coloring c("test.txt");
  canonical.allocate(c);
  cp.allocate(c);

  auto pf = pressure(canonical), pf2 = pressure(cp);

  execute<init>(canonical, pf);
  execute<copy>(pf, pf2);
  execute<print>(cp, pf2);
} // advance()
control::action<advance, cp::advance> advance_action;
