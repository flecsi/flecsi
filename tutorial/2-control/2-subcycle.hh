// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef TUTORIAL_2_CONTROL_2_SUBCYCLE_HH
#define TUTORIAL_2_CONTROL_2_SUBCYCLE_HH

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

namespace cycle {

enum class cp { initialize, advance, advance2, analyze, finalize };

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::initialize:
      return "initialize";
    case cp::advance:
      return "advance";
    case cp::advance2:
      return "advance2";
    case cp::analyze:
      return "analyze";
    case cp::finalize:
      return "finalize";
  }
  flog_fatal("invalid control point");
}

struct control_policy : flecsi::run::control_base {

  using control_points_enum = cp;
  struct node_policy {};

  using control = flecsi::run::control<control_policy>;

  /*
    Define a function to access the substep_ data member;
   */

  size_t & substep() {
    return substep_;
  }

  size_t & step() {
    return step_;
  }

  /*
    Define a subcycle control function.
   */

  static bool subcycle_control(control_policy & policy) {
    return policy.substep()++ % 3 < 2;
  }

  static bool cycle_control(control_policy & policy) {
    return policy.step()++ < 5;
  }

  /*
    Define a subcycle type.
   */

  using subcycle =
    cycle<subcycle_control, point<cp::advance>, point<cp::advance2>>;

  using main_cycle =
    flecsi::run::cycle<cycle_control, subcycle, point<cp::analyze>>;

  using control_points =
    list<point<cp::initialize>, main_cycle, point<cp::finalize>>;

private:
  size_t substep_{0};
  size_t step_{0};
};

using control = flecsi::run::control<control_policy>;

} // namespace cycle

#endif
