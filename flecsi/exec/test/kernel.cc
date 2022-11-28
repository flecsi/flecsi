#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;
using namespace flecsi::data;

using intN = field<std::array<size_t, 10>, single>;
const intN::definition<topo::index> array_field;

void
modify(intN::accessor<wo> a) {
  forall(i, util::span(*a), "modify") { i = 3; };
}

int
check(intN::accessor<ro> a) {
  UNIT {
    for(auto i : util::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}

int
reduce_vec(intN::accessor<ro> a) {
  UNIT {
    size_t res =
      reduceall(i, up, util::span(*a), exec::fold::sum, size_t, "reduce") {
      up(i);
    };
    EXPECT_EQ(res, 3 * a.get().size());
  };
}

int
kernel_driver() {
  UNIT {
    const auto ar = array_field(process_topology);
    execute<modify, default_accelerator>(ar);
    EXPECT_EQ(test<check>(ar), 0);
    EXPECT_EQ((test<reduce_vec, default_accelerator>(ar)), 0);
  };
} // kernel_driver

util::unit::driver<kernel_driver> driver;
