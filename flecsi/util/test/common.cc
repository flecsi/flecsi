#include "flecsi/util/common.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/function_traits.hh"
#include "flecsi/util/unit.hh"

#include <random>

using namespace flecsi;

float MyFun(double, int, long);
float MyFunNoexcept(double, int, long) noexcept;

template<class A, class B>
constexpr const bool & eq = std::is_same_v<A, B>;

template<auto & F, class R, class A, bool N>
constexpr bool
test() {
  using tf = util::function_t<F>;
  static_assert(tf::nonthrowing == N);
  static_assert(eq<typename tf::return_type, R>);
  static_assert(eq<typename tf::arguments_type, A>);
  return true;
}

static_assert(test<MyFun, float, std::tuple<double, int, long>, false>());
static_assert(
  test<MyFunNoexcept, float, std::tuple<double, int, long>, true>());

// ---------------
using c31 = util::constants<3, 1>;
static_assert(c31::size == 2);
static_assert(c31::index<1> == 1);
static_assert(c31::index<3> == 0);
static_assert(c31::first == 3);
using c4 = util::constants<4>;
static_assert(c4::value == 4);
static_assert(c4::first == 4);
static_assert(!util::constants<>::size);

// ---------------
int
common() {
  UNIT() {
    // square
    UNIT_CAPTURE() << flecsi::util::square(10) << std::endl;
    UNIT_CAPTURE() << flecsi::util::square(20.0) << std::endl;
    UNIT_CAPTURE() << std::endl;

    {
      util::counter<2> a(0);
      EXPECT_EQ(a(), 1);
      EXPECT_EQ(a(), 2);
    }

    {
      constexpr util::key_array<int, c31> m{};
      static_assert(&m.get<3>() == &m[0]);
      static_assert(&m.get<1>() == &m[1]);
    }

    {
      constexpr util::key_tuple<util::key_type<2, short>,
        util::key_type<8, void *>>
        p{1, nullptr};
      static_assert(p.get<2>() == 1);
      static_assert(p.get<8>() == nullptr);
    }

    {
      // demangle, type
      // The results depend on #ifdef __GNUG__, so we'll just exercise
      // these functions, without checking for particular results.
      EXPECT_NE(flecsi::util::demangle("foo"), "");

      auto str_demangle = UNIT_TTYPE(int);
      auto str_type = flecsi::util::type<int>();

      EXPECT_NE(str_demangle, "");
      EXPECT_NE(str_type, "");
      EXPECT_EQ(str_demangle, str_type);

      const auto sym = flecsi::util::symbol<common>();
#ifdef __GNUG__
      EXPECT_EQ(sym, "common()");
#else
      EXPECT_NE(sym, "");
#endif
    }

    // ------------------------
    // Compare
    // ------------------------

#ifdef __GNUG__

#if defined(__PPC64__) && !defined(_ARCH_PWR9)
    EXPECT_TRUE(UNIT_EQUAL_BLESSED("common.blessed.ppc"));
#else
    EXPECT_TRUE(UNIT_EQUAL_BLESSED("common.blessed.gnug"));
#endif
#elif defined(_MSC_VER)
    EXPECT_TRUE(UNIT_EQUAL_BLESSED("common.blessed.msvc"));
#else
    EXPECT_TRUE(UNIT_EQUAL_BLESSED("common.blessed"));
#endif
  };
} // common

util::unit::driver<common> driver;
