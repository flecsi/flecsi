#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

using double_field = field<double>;

const double_field::definition<topo::global> energy_field;

namespace {

double
init(double a, double_field::accessor<wo> ga) noexcept {
  ga[0] = a;
  ga[1] = -a;
  return a + 1;
}

int
check(exec::cpu s, future<double> x, double_field::accessor<ro> ga) noexcept {
  UNIT("TASK") {
    static_assert(std::is_same_v<decltype(ga[0]), const double &>);
    EXPECT_EQ(x.get(), ga[0] + 1 + s.launch().index);
    EXPECT_EQ(ga[0], -ga[1]);
  };
}

double
index_init(exec::cpu s, double a, exec::launch_domain) noexcept {
  return a + s.launch().index;
}

void
void_task() noexcept {
  flog(info) << "this is a void task" << std::endl;
}

void
index_void_task(exec::launch_domain) noexcept {
  flog(info) << "this is an index void task" << std::endl;
}
} // namespace

int
reduction_task(exec::cpu s, int a, exec::launch_domain) noexcept {
  return a + s.launch().index;
}

int
future_driver(scheduler & s) {
  UNIT() {
    double d = 3.1;
    topo::global::slot g2;
    g2.allocate(s, 2);
    const auto energy = energy_field(g2);

    // single future
    auto f = s.execute<init>(d, energy);

    EXPECT_EQ(s.test<check>(exec::on, f, energy), 0);
    EXPECT_EQ(f.get(), ++d);

    // future map
    const exec::launch_domain ld{s.runtime().processes()};
    auto fm = s.execute<index_init>(exec::on, d, ld);
    for(auto v : fm.all())
      EXPECT_EQ(v, d++);

    // For all values because it's an index future:
    EXPECT_EQ(s.test<check>(exec::on, fm, energy), 0);

    auto fv = s.execute<void_task>();

    fv.wait();
    fv.get();

    auto fv2 = s.execute<index_void_task>(ld);

    fv2.wait();

    int a = 7;
    // checking reduction operations
    auto fmin = s.reduce<reduction_task, exec::fold::min>(exec::on, a, ld);
    EXPECT_EQ(fmin.get(), a);

    auto fmax = s.reduce<reduction_task, exec::fold::max>(exec::on, a, ld);
    EXPECT_EQ(fmax.get(), int(a + s.runtime().processes() - 1));

    auto fsum = s.reduce<reduction_task, exec::fold::sum>(exec::on, a, ld);
    int sum = 0;
    for(Color i = 0; i < s.runtime().processes(); i++)
      sum += a + i;
    EXPECT_EQ(fsum.get(), sum);
  };
} // future

util::unit::driver<future_driver> driver;
