// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_UNIT_HH
#define FLECSI_UTIL_UNIT_HH

#include "flecsi/flog.hh"
#include "flecsi/runtime.hh"
#include "flecsi/util/unit/types.hh"

#include <tuple>

namespace flecsi::util::unit {
/// \defgroup unit Unit Testing
/// Unit test framework much like Google Test but with task support.
/// Each \a Target is a function with signature `int(flecsi::scheduler&)` or
/// (\b deprecated) `int()`; if any returns a
/// non-zero value, so does the process built with \c flecsi_add_test.
/// Output is via \ref flog.
///
/// \ns{util::unit}.
/// \ingroup utils
/// \{

enum class test_control_points {
  initialization,
  driver,
  finalization,
  exit
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
    case test_control_points::exit:
      return "exit";
  }
  flog_fatal("invalid unit test control point");
}

struct control_policy : flecsi::run::control_base {

  control_policy() : status(0x0) {}

  using control_points_enum = test_control_points;

  using control_points = list<point<control_points_enum::initialization>,
    point<control_points_enum::driver>,
    point<control_points_enum::finalization>,
    point<control_points_enum::exit>>;

  static void exit(control_policy & p) {
    throw exception{p.status};
  }

  int status;
}; // struct control_policy

using control = flecsi::run::control<control_policy>;

using target_type = int (&)(scheduler &);
template<target_type F, test_control_points cp>
class action
{
private:
  static void wrap(control_policy & p) {
    p.status |= F(p.scheduler());
  }
  control::action<wrap, cp> act;
};

namespace detail {
template<target_type F>
constexpr target_type
adapt() {
  return F;
}
template<int (&F)()>
[[deprecated("accept a scheduler")]] constexpr target_type
adapt() {
  return *[](scheduler &) { return F(); };
}
} // namespace detail

/// A test initialization registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the function to call
template<auto & Target>
using initialization =
  action<detail::adapt<Target>(), test_control_points::initialization>;

/// A test registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the test function to call
template<auto & Target>
using driver = action<detail::adapt<Target>(), test_control_points::driver>;

/// A test finalization registration.
/// Declare a non-local variable of this type for each function.
/// \tparam Target the function to call
template<auto & Target>
using finalization =
  action<detail::adapt<Target>(), test_control_points::finalization>;

inline void
accelerator_config([[maybe_unused]] run::config & c) {
#if FLECSI_BACKEND == FLECSI_BACKEND_legion
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  c.legion = {"", "-ll:gpu", "1"};
#elif defined(REALM_USE_OPENMP)
  c.legion = {"", "-ll:ocpu", "1", "-ll:onuma", "0"};
#endif
#elif FLECSI_BACKEND == FLECSI_BACKEND_hpx
  // Setting `hpx.ignore_batch_env=1` instructs HPX not to rely on
  // environment information passed to the application by Slurm (or
  // another) batch scheduler.  This setting allows all tests to be run
  // from a single batch job even if different tests involve, e.g.,
  // different process counts.  If this setting is not used, conflicting
  // configuration information is passed to HPX, causing possible hangs
  // during the execution of the tests.
  //
  // Setting `hpx.os_threads=4` specifies that HPX should allocate four OS
  // threads, which are used for running FleCSI tasks.  This configuration
  // setting avoids oversubscription of the test environment with multiple
  // processes on the same node.
  c.hpx = {"hpx.ignore_batch_env!=1", "hpx.os_threads!=4"};
#endif
}

/// \}
} // namespace flecsi::util::unit

#endif
