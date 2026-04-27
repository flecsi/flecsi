#include "flecsi/data.hh"
#include "flecsi/execution.hh"
#include "flecsi/topology.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;
using namespace flecsi::data;
using namespace flecsi::exec;

using intN = field<std::array<size_t, 10>, single>;
const intN::definition<topo::index> array_field;

void
modify(accelerator s, intN::accessor<wo> a) noexcept {
  s.executor().forall(i, std::span(*a)) {
    i = 3;
  };
}

int
check(intN::accessor<ro> a) noexcept {
  UNIT() {
    for(auto i : std::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}

void
modify_policy(accelerator s, intN::accessor<wo> a) noexcept {
  s.executor().threads<64, 1>().forall(i, std::span(*a)) {
    i = 3;
  };
}

int
check_policy(intN::accessor<ro> a) noexcept {
  UNIT() {
    for(auto i : std::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}

struct I {
  int i;
  FLECSI_INLINE_TARGET I operator+(I o) const {
    return {i + o.i};
  }
};
template<>
constexpr I flecsi::exec::fold::sum::identity<I>{};

int
reduce_vec(accelerator s, intN::accessor<ro> a) noexcept {
  UNIT() {
    size_t res =
      s.executor().reduceall(i, up, std::span(*a), exec::fold::sum, size_t) {
      up(i);
    };
    EXPECT_EQ(res, 3 * a.get().size());
    EXPECT_EQ((s.executor().reduceall(
                 i, up, util::iota_view(0, 4), exec::fold::sum, I) {
      up({i});
    }).i,
      6);
  };
}

void
mdrange_init(accelerator s, intN::accessor<wo> a) noexcept {
  auto ar = std::span(*a);
  util::mdspan<std::size_t, 2> md_ar(ar.data(), {5, 2});
  s.executor().forall(mi, (mdiota_view(md_ar, full_range(), prefix_range{2}))) {
    auto [i, j] = mi;
    md_ar[j][i] = 3;
  };
}

int
check_mdrange(intN::accessor<ro> a) noexcept {
  UNIT() {
    for(auto i : std::span(*a)) {
      EXPECT_EQ(i, 3);
    }
  };
}

int
reduce_mdrange_vec(accelerator s, intN::accessor<rw> a) noexcept {
  UNIT() {
    auto ar = std::span(*a);
    util::mdspan<std::size_t, 2> md_ar(ar.data(), {5, 2});
    size_t res = s.executor().reduceall(mi,
      up,
      mdiota_view(md_ar, full_range(), prefix_range{2}),
      exec::fold::sum,
      size_t) {
      auto [i, j] = mi;
      up(md_ar[j][i]);
    };
    EXPECT_EQ(res, 3 * a.get().size());
  };
}

int
kernel_driver(scheduler & s) {
  UNIT() {
    topo::index::topology pt(s, s.runtime().processes());
    const auto ar = array_field(pt);
    s.execute<modify>(on, ar);
    EXPECT_EQ(s.test<check>(ar), 0);
    s.execute<modify_policy>(on, ar);
    EXPECT_EQ(s.test<check_policy>(ar), 0);
    s.execute<mdrange_init>(on, ar);
    EXPECT_EQ((s.test<check_mdrange>(ar)), 0);
    EXPECT_EQ((s.test<reduce_vec>(on, ar)), 0);
    EXPECT_EQ((s.test<reduce_mdrange_vec>(on, ar)), 0);
  };
} // kernel_driver

util::unit::driver<kernel_driver> driver;
