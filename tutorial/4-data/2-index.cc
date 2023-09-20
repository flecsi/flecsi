#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "control.hh"

using namespace flecsi;

template<typename T>
using single = field<T, data::single>;
const single<std::size_t>::definition<topo::index> ifield;

void
init(single<std::size_t>::accessor<wo> iv) {
  flog(trace) << "initializing value on color " << color() << " of " << colors()
              << std::endl;
  iv = color();
}

void
print(single<std::size_t>::accessor<ro> iv) {
  flog(trace) << "index value: " << iv << " (color " << color() << " of "
              << colors() << ")" << std::endl;
}

void
advance(control_policy &) {
  topo::index::slot custom_topology;
  custom_topology.allocate(4);

  execute<init>(ifield(custom_topology));
  execute<print>(ifield(custom_topology));
} // advance()
control::action<advance, cp::advance> advance_action;
