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
#pragma once

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"
#include "flecsi/util/unit/types.hh"

#include <tuple>

/// \cond core
namespace flecsi {
namespace unit {
/// \defgroup unit Unit Testing
/// Unit test framework much like Google Test but with task support.
/// \ingroup utils
/// \{

enum class test_control_points {
  initialization,
  driver,
  finalization
}; // enum test_control_points

inline const char *
operator*(test_control_points cp) {
  switch(cp) {
    case test_control_points::initialization:
      return "initialization";
    case test_control_points::driver:
      return "driver";
    case test_control_points::finalization:
      return "finalization";
  }
  flog_fatal("invalid unit test control point");
}

struct control_policy {

  using control_points_enum = test_control_points;

  struct node_policy {};

  template<auto CP>
  using control_point = flecsi::run::control_point<CP>;

  using control_points =
    std::tuple<control_point<control_points_enum::initialization>,
      control_point<control_points_enum::driver>,
      control_point<control_points_enum::finalization>>;

}; // struct control_policy

using control = flecsi::run::control<flecsi::unit::control_policy>;

/// A test initialization registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the function to call
template<control::target_type Target>
using initialization =
  control::action<Target, test_control_points::initialization>;

/// A test registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the test function to call
template<control::target_type Target>
using driver = control::action<Target, test_control_points::driver>;

/// A test finalization registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the function to call
template<control::target_type Target>
using finalization = control::action<Target, test_control_points::finalization>;

/// \}
} // namespace unit
} // namespace flecsi
//// \endcore
