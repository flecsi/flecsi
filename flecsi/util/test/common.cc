/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */

#include "flecsi/util/common.hh"
#include "flecsi/util/constant.hh"
#include "flecsi/util/demangle.hh"
#include "flecsi/util/function_traits.hh"
#include "flecsi/util/unit.hh"

#include <random>

using namespace flecsi;

struct MyClass {
  int operator()(float, double, long double) const {
    return 0;
  }

  void mem(char, int) {}
  void memc(char, int) const {}
  void memv(char, int) volatile {}
  void memcv(char, int) const volatile {}
};

inline float
MyFun(double, int, long) {
  return float(0);
}

template<class T>
using ft = util::function_traits<T>;

template<class A, class B>
constexpr const bool & eq = std::is_same_v<A, B>;

template<class T>
using ret = typename ft<T>::return_type;
template<class T>
using args = typename ft<T>::arguments_type;
template<class... TT>
using tup = std::tuple<TT...>;

template<class T, class R, class A>
constexpr bool
test() {
  static_assert(eq<typename ft<T>::return_type, R>);
  static_assert(eq<typename ft<T>::arguments_type, A>);
  return true;
}
template<class T, class... TT>
constexpr bool
same() {
  return (
    test<TT, typename ft<T>::return_type, typename ft<T>::arguments_type>() &&
    ...);
}
template<auto M>
constexpr bool
pmf() {
  using T = decltype(M);
  static_assert(eq<typename ft<T>::owner_type, MyClass>);
  return test<T, void, tup<char, int>>();
}

static_assert(pmf<&MyClass::mem>());
static_assert(pmf<&MyClass::memc>());
static_assert(pmf<&MyClass::memv>());
static_assert(pmf<&MyClass::memcv>());

static_assert(test<MyClass, int, tup<float, double, long double>>());
static_assert(test<decltype(MyFun), float, tup<double, int, long>>());
static_assert(same<decltype(MyFun),
  decltype(&MyFun),
  decltype(*MyFun),
  std::function<decltype(MyFun)>>());
static_assert(same<MyClass,
  MyClass &,
  const MyClass &,
  volatile MyClass &,
  const volatile MyClass &,
  MyClass &&,
  const MyClass &&,
  volatile MyClass &&,
  const volatile MyClass &&>());

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
  UNIT {
    // types
    UNIT_CAPTURE() << UNIT_TTYPE(FLECSI_COUNTER_TYPE) << std::endl;
    UNIT_CAPTURE() << UNIT_TTYPE(flecsi::util::counter_t) << std::endl;
    UNIT_CAPTURE() << std::endl;

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

flecsi::unit::driver<common> driver;
