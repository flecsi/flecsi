#include <flecsi/data.hh>
#include <flecsi/execution.hh>
#include <flecsi/flog.hh>

#include "../3-execution/control.hh"
#include "../4-data/canonical.hh"

// this tutorial is based on a 4-data/3-dense.cc tutorial example

using namespace flecsi;

const field<double>::definition<canon, canon::cells> pressure;

void
init(canon::accessor<ro> t, field<double>::accessor<wo> p) noexcept {
  std::size_t off{0};
  for(const auto c : t.cells()) {
    p[c] = (off++) * 2.0;
  } // for
} // init

void
modify(exec::accelerator s,
  canon::accessor<ro> t,
  field<double>::accessor<rw> p) noexcept {
  s.executor().forall(c, t.cells()) {
    p[c] += 1;
  };
}

void
print(canon::accessor<ro> t, field<double>::accessor<ro> p) noexcept {
  std::size_t off{0};
  for(auto c : t.cells()) {
    flog(info) << "cell " << off++ << " has pressure " << p[c] << std::endl;
  } // for
} // print

void
advance(control_policy & p) {
  auto & s = p.scheduler();

  canon::slot canonical;
  canonical.allocate(s, canon::mpi_coloring(s, "test.txt"));

  auto pf = pressure(canonical);

  // cpu task, default
  s.execute<init>(canonical, pf);
  // Automatically select an execution space based on Kokkos configuration.
  // The runtime moves data between the host and device.
  s.execute<modify>(exec::on, canonical, pf);
  // cpu_task
  s.execute<print>(canonical, pf);
}
control::action<advance, cp::advance> advance_action;
