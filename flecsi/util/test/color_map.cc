#include "flecsi/util/color_map.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/unit.hh"

using namespace flecsi;

namespace {
struct single {
  single() = default; // no serialization needed
  single(int r) : r(r) {
    if(!r && std::exchange(alive, true))
      flog_fatal("two single objects");
  }
  ~single() {
    alive = false;
  }

  static inline bool alive;

  Color r;
};
} // namespace

int
interface() {
  UNIT("TASK") {
    // Assumed that the test is run with 3 threads and 8 colors
    ASSERT_EQ(processes(), 3lu);

    {
      static constexpr util::equal_map em(10, 4);
      constexpr auto chk = [](Color c, std::size_t a, std::size_t sz) {
        const auto r = em[c];
        for(auto i : r)
          if(em.bin(i) != c)
            return false;
        return r.begin() == a && r.size() == sz;
      };
      static_assert(chk(0, 0, 3));
      static_assert(chk(1, 3, 3));
      static_assert(chk(2, 6, 2));
      static_assert(chk(3, 8, 2));

      util::offsets eo(em), off({1, 2, 5});
      EXPECT_EQ(eo[3].size(), em[3].size());
      EXPECT_EQ(off[0].size(), 1);
      EXPECT_EQ(off[1].size(), 1);
      EXPECT_EQ(off[2].size(), 3);
    }

    EXPECT_EQ(util::mpi::one_to_alli([](int r, int) { return single{r}; }, 0).r,
      process());
  };
}

int
color_map() {
  UNIT() { EXPECT_EQ((test<interface, mpi>()), 0); };
}

util::unit::driver<color_map> color_map_driver;
