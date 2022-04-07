// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

using double_field = field<double>;

const double_field::definition<topo::global> energy_field;

namespace future_test {

double
init(double a, double_field::accessor<wo> ga) {
  ga[0] = a;
  ga[1] = -a;
  return a + 1;
}

int
check(future<double> x, double_field::accessor<ro> ga) {
  UNIT {
    static_assert(std::is_same_v<decltype(ga[0]), const double &>);
    EXPECT_EQ(x.get(), ga[0] + 1 + color());
    EXPECT_EQ(ga[0], -ga[1]);
  };
}

double
index_init(double a, exec::launch_domain) {
  return a + color();
}

void
void_task() {
  flog(info) << "this is a void task" << std::endl;
}

void
index_void_task(exec::launch_domain) {
  flog(info) << "this is an index void task" << std::endl;
}
} // namespace future_test

int
reduction_task(int a, exec::launch_domain) {
  return a + color();
}

int
future_driver() {
  UNIT {
    using namespace future_test;

    topo::global::cslot g2c;
    g2c.allocate(2);
    topo::global::slot g2;
    g2.allocate(g2c.get());
    const auto energy = energy_field(g2);

    // single future
    auto f = execute<init>(3.1, energy);

    EXPECT_EQ(test<check>(f, energy), 0);
    EXPECT_EQ(f.get(), 3.1 + 1);

    // future map
    const exec::launch_domain ld{run::context::instance().processes()};
    auto fm = execute<index_init>(f.get(), ld);
    EXPECT_EQ(fm.get(0, false), f.get());

    // For all values because it's an index future:
    EXPECT_EQ(test<check>(fm, energy), 0);

    auto fv = execute<void_task>();

    fv.wait();
    fv.get();

    auto fv2 = execute<index_void_task>(ld);

    fv2.wait();
    fv2.get();

    int a = 7;
    // checking reduction operations
    auto fmin = reduce<reduction_task, exec::fold::min>(a, ld);
    EXPECT_EQ(fmin.get(), a);

    auto fmax = reduce<reduction_task, exec::fold::max>(a, ld);
    EXPECT_EQ(fmax.get(), int(a + run::context::instance().processes() - 1));

    auto fsum = reduce<reduction_task, exec::fold::sum>(a, ld);
    int sum = 0;
    for(Color i = 0; i < run::context::instance().processes(); i++)
      sum += a + i;
    EXPECT_EQ(fsum.get(), sum);
  };
} // future

flecsi::unit::driver<future_driver> driver;
