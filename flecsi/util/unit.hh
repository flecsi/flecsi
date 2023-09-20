// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_UNIT_HH
#define FLECSI_UTIL_UNIT_HH

#include "flecsi/execution.hh"
#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"
#include "flecsi/util/unit/types.hh"

#include <tuple>

namespace flecsi::util::unit {
/// \defgroup unit Unit Testing
/// Unit test framework much like Google Test but with task support.
/// Output is via \ref flog.
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

struct control_policy : flecsi::run::control_base {

  control_policy() : status(0x0) {}

  ~control_policy() noexcept(false) {
    throw exception{status};
  }

  using control_points_enum = test_control_points;

  struct node_policy {};

  using control_points = list<point<control_points_enum::initialization>,
    point<control_points_enum::driver>,
    point<control_points_enum::finalization>>;

  int status;
}; // struct control_policy

using control = flecsi::run::control<control_policy>;

template<int (&F)(), test_control_points cp>
class action
{
private:
  static void wrap(control_policy & p) {
    p.status |= F();
  }
  control::action<wrap, cp> act;
};

using target_type = int (&)();
/// A test initialization registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the function to call
template<target_type Target>
using initialization = action<Target, test_control_points::initialization>;

/// A test registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the test function to call
template<target_type Target>
using driver = action<Target, test_control_points::driver>;

/// A test finalization registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the function to call
template<target_type Target>
using finalization = action<Target, test_control_points::finalization>;

/// \}
} // namespace flecsi::util::unit

#endif
