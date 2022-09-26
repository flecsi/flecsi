#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;
using namespace flecsi::data;
using namespace flecsi::exec;

using intN = field<std::array<size_t, 10>, single>;
const intN::definition<topo::index> array_field;

void
modify(intN::accessor<wo> a) {
  forall(i, util::span(*a), "modify") { i = 3; };
}

int
check(intN::accessor<ro> a) {
  UNIT() {
    for(auto i : util::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}

void
modify_policy(intN::accessor<wo> a) {
  forall(i, flecsi::exec::range_policy(util::span(*a)), "modify_policy") {
    i = 3;
  };
}

int
check_policy(intN::accessor<ro> a) {
  UNIT() {
    for(auto i : util::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}

int
reduce_vec(intN::accessor<ro> a) {
  UNIT() {
    size_t res =
      reduceall(i, up, util::span(*a), exec::fold::sum, size_t, "reduce") {
      up(i);
    };
    EXPECT_EQ(res, 3 * a.get().size());
  };
}

void
modify_bound(intN::accessor<wo> a) {
  forall(j, util::substring_view(util::span(*a), 0, 5), "modify_first") {
    j = 2;
  };
  forall(j, util::substring_view(util::span(*a), 5, 5), "modify_last") {
    j = 5;
  };
}

int
check_bound(intN::accessor<ro> a) {
  UNIT() {
    for(auto j : util::span(*a).subspan(0, 5)) {
      EXPECT_EQ(j, 2);
    }
    for(auto j : util::span(*a).subspan(5, 5)) {
      EXPECT_EQ(j, 5);
    }
  };
}

void
mdrange_init(intN::accessor<wo> a) {
  auto ar = util::span(*a);
  util::mdspan<std::size_t, 2> md_ar(ar.data(), {5, 2});
  forall(
    mi, (mdiota_view(md_ar, full_range(), prefix_range{2})), "mdrange_test") {
    auto [i, j] = mi;
    md_ar[j][i] = 3;
  };
}

int
check_mdrange(intN::accessor<ro> a) {
  UNIT() {
    for(auto i : util::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}

int
reduce_vec_bound(intN::accessor<ro> a) {
  UNIT() {
    size_t res_first = reduceall(j,
      up,
      util::substring_view(util::span(*a), 0, 5),
      exec::fold::sum,
      size_t,
      "reduce_first") {
      up(j);
    };
    EXPECT_EQ(res_first, 2 * 5);

    size_t res_last = reduceall(j,
      up,
      util::substring_view(util::span(*a), 5, 5),
      exec::fold::sum,
      size_t,
      "reduce_last") {
      up(j);
    };
    EXPECT_EQ(res_last, 5 * 5);
  };
}

int
reduce_mdrange_vec(intN::accessor<rw> a) {
  UNIT() {
    auto ar = util::span(*a);
    util::mdspan<std::size_t, 2> md_ar(ar.data(), {5, 2});
    size_t res = reduceall(mi,
      up,
      mdiota_view(md_ar, full_range(), prefix_range{2}),
      exec::fold::sum,
      size_t,
      "mdrange_reduce") {
      auto [i, j] = mi;
      up(md_ar[j][i]);
    };
    EXPECT_EQ(res, 3 * a.get().size());
  };
}

int
kernel_driver() {
  UNIT() {
    const auto ar = array_field(process_topology);
    execute<modify, default_accelerator>(ar);
    EXPECT_EQ(test<check>(ar), 0);
    execute<modify_policy, default_accelerator>(ar);
    EXPECT_EQ(test<check_policy>(ar), 0);
    execute<mdrange_init, default_accelerator>(ar);
    EXPECT_EQ((test<check_mdrange>(ar)), 0);
    EXPECT_EQ((test<reduce_vec, default_accelerator>(ar)), 0);
    EXPECT_EQ((test<reduce_mdrange_vec, default_accelerator>(ar)), 0);
    execute<modify_bound, default_accelerator>(ar);
    EXPECT_EQ(test<check_bound>(ar), 0);
    EXPECT_EQ((test<reduce_vec_bound, default_accelerator>(ar)), 0);
  };
} // kernel_driver

flecsi::unit::driver<kernel_driver> driver;
