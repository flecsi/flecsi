#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "../3-execution/control.hh"

using namespace flecsi;

template<typename T>
using single = field<T, data::single>;
const single<std::size_t>::definition<topo::index> ifield;

void
init(exec::cpu s, single<std::size_t>::accessor<wo> iv) {
  flog(trace) << "initializing value on color " << s.launch().index << " of "
              << s.launch().size << std::endl;
  iv = s.launch().index;
}

void
print(exec::cpu s, single<std::size_t>::accessor<ro> iv) {
  flog(trace) << "index value: " << iv << " (color " << s.launch().index
              << " of " << s.launch().size << ")" << std::endl;
}

void
advance(control_policy &) {
  topo::index::slot custom_topology;
  custom_topology.allocate(4);

  execute<init>(exec::on, ifield(custom_topology));
  execute<print>(exec::on, ifield(custom_topology));
} // advance()
control::action<advance, cp::advance> advance_action;
