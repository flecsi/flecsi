#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

using namespace flecsi;
using namespace flecsi::topo;

template<typename T>
using single = field<T, data::single>;
const single<double>::definition<global> gfield;

void
init(double v, single<double>::accessor<wo> gv) {
  gv = v;
}

void
print(single<double>::accessor<ro> gv) {
  flog(trace) << "global value: " << gv << std::endl;
}

void
advance(control_policy &) {
  topo::global::slot gtopo;
  gtopo.allocate(1);
  const auto v = gfield(gtopo);
  execute<init>(42.0, v);
  execute<print>(v);
} // advance()
control::action<advance, cp::advance> advance_action;
