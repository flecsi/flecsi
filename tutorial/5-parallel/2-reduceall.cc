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
reduce1(exec::accelerator s,
  canon::accessor<ro> t,
  field<double>::accessor<ro> p) noexcept {
  auto res = s.executor().named("reduce1").reduceall(
    c, up, t.cells(), exec::fold::max, double) {
    up(p[c]);
  }; // forall

  flog_assert(res == 6.0, res << " != 6.0");
}

struct reduce2 {
  template<class S>
  static void
  task(S s, canon::accessor<ro> t, field<double>::accessor<ro> p) noexcept {
    auto res = s.executor().template reduce<exec::fold::max, double>(
      t.cells(), FLECSI_LAMBDA(auto c, auto up) { up(p[c]); });

    flog_assert(res == 6.0, res << " != 6.0");
  }
};
template<>
void reduce2::task(exec::gpu,
  canon::accessor<ro>,
  field<double>::accessor<ro>) noexcept = delete;

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

  canon::topology canonical(s, canon::mpi_coloring(s, "test.txt"));

  auto pf = pressure(canonical);

  // cpu task, default
  s.execute<init>(canonical, pf);
  s.execute<reduce1>(exec::on, canonical, pf);
  s.execute<reduce2>(exec::on, canonical, pf);
  // cpu_task
  s.execute<print>(canonical, pf);
}
control::action<advance, cp::advance> advance_action;
