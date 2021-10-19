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

#include <flecsi/util/unit.hh>

using namespace flecsi;

/*----------------------------------------------------------------------------*
  These test the unit test control model, as well as the different
  trace levels. To actually test the trace levels, you must do several
  cmake configurations that change the strip level (FLOG_STRIP_LEVEL).

  The control model has control points: "initialization", "driver", and
  "finalization".
 *----------------------------------------------------------------------------*/

int
init_a() {
  flog::devel_guard guard(unit_tag);
  flog(info) << "init a" << std::endl;

  flog(trace) << "trace (strip level " << FLOG_STRIP_LEVEL << ")" << std::endl;
  flog(info) << "info (strip level " << FLOG_STRIP_LEVEL << ")" << std::endl;
  flog(warn) << "warn (strip level " << FLOG_STRIP_LEVEL << ")" << std::endl;
  flog(error) << "error (strip level " << FLOG_STRIP_LEVEL << ")" << std::endl;
  return 0;
}

unit::initialization<init_a> ia_action;

int
init_b() {
  flog::devel_guard guard(unit_tag);
  flog(info) << "init b" << std::endl;
  return 0;
}

unit::initialization<init_b> ib_action;
const auto ab = ib_action.add(ia_action);

int
test1() {
  UNIT() {

    ASSERT_EQ(0, 0);
    EXPECT_EQ(0, 0);

    flog::devel_guard guard(unit_tag);
    flog(info) << "THIS IS SOME LOG INFO FOR A TEST" << std::endl;
  };
}

unit::driver<test1> test1_driver;

int
test2() {
  UNIT() {

    ASSERT_EQ(0, 0);
    int v{0};
    ASSERT_EQ(v, 0);
  };
}

unit::driver<test2> test2_driver;

int
finalization() {
  flog::devel_guard guard(unit_tag);
  flog(info) << "finalize" << std::endl;
  return 0;
}

unit::finalization<finalization> f_action;

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

flecsi::unit::driver<dag> driver;
