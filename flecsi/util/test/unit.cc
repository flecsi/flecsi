#include "flecsi/execution.hh"
#include <flecsi/util/unit.hh>

using namespace flecsi;

/*----------------------------------------------------------------------------*
  These test the unit test control model, as well as the different
  trace levels.

  The control model has control points: "initialization", "driver", and
  "finalization".
 *----------------------------------------------------------------------------*/

int
log_driver() {
  UNIT() {
    {
      std::vector<std::size_t> v;
      for(std::size_t i{0}; i < 10; ++i) {
        v.emplace_back(i);
      }

      EXPECT_EQ(flog::to_string(v), "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
    }

    {
      std::vector<std::vector<std::size_t>> v;
      for(std::size_t i{0}; i < 10; ++i) {
        v.push_back({0, 1, 2});
      }

      EXPECT_EQ(flog::to_string(v),
        "[[0, 1, 2],\n [0, 1, 2],\n [0, 1, 2],\n [0, 1, 2],\n [0, 1, 2],\n [0, "
        "1, 2],\n [0, 1, 2],\n [0, 1, 2],\n [0, 1, 2],\n [0, 1, 2]]");
    }

    {
      std::map<std::size_t, std::size_t> m;
      for(std::size_t i{0}; i < 10; ++i) {
        m[i] = i;
      }

      EXPECT_EQ(flog::to_string(m),
        "{0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}");
    }

    {
      std::map<std::size_t, std::vector<std::size_t>> m;
      for(std::size_t i{0}; i < 10; ++i) {
        m[i] = {0, 1, 2};
      }

      EXPECT_EQ(flog::to_string(m),
        "{0:\n [0, 1, 2],\n 1:\n [0, 1, 2],\n 2:\n [0, 1, 2],\n 3:\n [0, 1, "
        "2],\n 4:\n [0, 1, 2],\n 5:\n [0, 1, 2],\n 6:\n [0, 1, 2],\n 7:\n [0, "
        "1, 2],\n 8:\n [0, 1, 2],\n 9:\n [0, 1, 2]}");
    }
  };
} // flog

util::unit::driver<log_driver> log_test_driver;

int
init_a() {
  flog(info) << "init" << std::endl;
  auto &sl = flog::state::instance().strip_level(), current = sl;

  for(int i = 0; i < 5; ++i) {
    sl = i;

    flog(trace) << "trace (strip level " << i << ")" << std::endl;
    flog(info) << "info (strip level " << i << ")" << std::endl;
    flog(warn) << "warn (strip level " << i << ")" << std::endl;
    flog(error) << "error (strip level " << i << ")" << std::endl;
  }

  sl = current;

  bool color = flog::state::color_output();
  flog(info) << "COLOR OUTPUT: " << (color ? "true" : "false") << std::endl;
  color = !color;
  flog(info) << "COLOR OUTPUT: " << (!color ? "true" : "false") << std::endl;
  color = !color;

  return 0;
}

util::unit::initialization<init_a> ia_action;

int
test1() {
  UNIT() {

    ASSERT_EQ(0, 0);
    EXPECT_EQ(0, 0);

    flog(info) << "THIS IS SOME LOG INFO FOR A TEST" << std::endl;
  };
}

util::unit::driver<test1> test1_driver;

int
test2() {
  UNIT() {

    ASSERT_EQ(0, 0);
    int v{0};
    ASSERT_EQ(v, 0);
  };
}

util::unit::driver<test2> test2_driver;

int
finalization() {
  flog(info) << "finalize" << std::endl;
  return 0;
}

util::unit::finalization<finalization> f_action;

/*----------------------------------------------------------------------------*
  This tests task execution in the unit test framework.
 *----------------------------------------------------------------------------*/

int
task_pass() {
  UNIT("TASK") {
    flog(info) << "this test passes" << std::endl;
    ASSERT_EQ(1, 1);
  }; // UNIT
}

int
task_assert_fail() {
  UNIT("TASK") {
    flog(info) << "this test fails an assertion" << std::endl;
    ASSERT_EQ(+0, 1); // + should not appear in output
  }; // UNIT
}

int
task_expect_fail() {
  UNIT("TASK") {
    flog(info) << "this test fails an expectation" << std::endl;
    EXPECT_EQ(+0, 1);
  }; // UNIT
}

program_option<bool> fail("Test Options",
  "fail,f",
  "Force test failure.",
  {{flecsi::option_implicit, true}, {flecsi::option_zero}});

int
dag() {
  UNIT() {
    ASSERT_EQ(test<task_pass>(), 0);
    ASSERT_NE(test<task_assert_fail>(), 0);
    ASSERT_NE(test<task_expect_fail>(), 0);

    flog(info) << "output from driver" << std::endl;

    EXPECT_EQ(test<task_pass>(), 0);
    EXPECT_NE(test<task_assert_fail>(), 0);
    EXPECT_NE(test<task_expect_fail>(), 0);

    // These show what happens during actual failure
    if(fail.has_value()) {
      EXPECT_EQ(test<task_expect_fail>(), 0);
      ASSERT_EQ(test<task_assert_fail>(), 0);
    } // if
  }; // UNIT
} // dag

FLECSI_TARGET int
no_failures() {
  UNIT() {
    ASSERT_GT(42, 3.14);
    ASSERT_GE('x', 'x');
    EXPECT_LE(1 + 2 + 3, 7);
    EXPECT_LT(12, 20);
    ASSERT_TRUE(5 < 6);
    char str[] = "FleCSI";
    EXPECT_STREQ(str, "FleCSI");
    ASSERT_STRNE(str, "flecsi");
  };
}

FLECSI_TARGET int
expect_failure(int & counter) {
  UNIT() {
    EXPECT_GT(1.0, 1.0);
    ++counter; // ensure the expect didn't exit early
  };
}

FLECSI_TARGET int
assert_failure(int & counter) {
  UNIT() {
    ASSERT_GT('a', 'a' + 1);
    ++counter; // ensure the assert actually exited early
  };
}

FLECSI_TARGET int
gpu_unit_test() {
  UNIT() {
    EXPECT_EQ(no_failures(), 0);
    int counter{0};
    EXPECT_EQ(assert_failure(counter), 1);
    EXPECT_EQ(counter, 0);
    EXPECT_EQ(expect_failure(counter), 1);
    EXPECT_EQ(counter, 1);
  };
}

int
launch_gpu_unit() {
  using flecsi::exec::parallel_reduce;
  using flecsi::exec::fold::sum;
  using flecsi::util::iota_view;
  const int num_threads{1};
  return parallel_reduce<sum, int>(
    iota_view<int>(0, num_threads),
    FLECSI_LAMBDA(auto &&, auto op) { op(gpu_unit_test()); },
    "run all tests");
}

/*----------------------------------------------------------------------------*
  Driver for tests of the unit testing framework
 *----------------------------------------------------------------------------*/

int
unit_test_framework() {
  UNIT() {
    EXPECT_EQ((test<launch_gpu_unit, default_accelerator>()), 0);
    EXPECT_EQ(dag(), 0);
  };
}

util::unit::driver<unit_test_framework> driver;
