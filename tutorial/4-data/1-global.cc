#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "../3-execution/control.hh"

using namespace flecsi;
using namespace flecsi::topo;

template<typename T>
using single = field<T, data::single>;
const single<double>::definition<global> gfield;

void
init(double v, single<double>::accessor<wo> gv) noexcept {
  gv = v;
}

void
print(single<double>::accessor<ro> gv) noexcept {
  flog(trace) << "global value: " << gv << std::endl;
}

void
advance(control_policy & p) {
  auto & s = p.scheduler();
  topo::global::topology gtopo(s, 1);
  const auto v = gfield(gtopo);
  s.execute<init>(42.0, v);
  s.execute<print>(v);
} // advance()
control::action<advance, cp::advance> advance_action;
