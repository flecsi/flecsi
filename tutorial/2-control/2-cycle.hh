#ifndef TUTORIAL_2_CONTROL_2_CYCLE_HH
#define TUTORIAL_2_CONTROL_2_CYCLE_HH

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

namespace cycle {

enum class cp { initialize, advance, analyze, finalize };

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::initialize:
      return "initialize";
    case cp::advance:
      return "advance";
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

  // Define a function to access the step_ data member.

  size_t & step() {
    return step_;
  }

  // Cycle predicates are passed the policy object.

  static bool cycle_control(control_policy & policy) {
    return policy.step()++ < 5;
  }

  // A cycle type. Cycles are similar to the control_points tuple, with the
  // addition of a predicate function that controls termination of the cycle.

  using main_cycle =
    cycle<cycle_control, point<cp::advance>, point<cp::analyze>>;

  // The control_points list type takes the cycle as one of its
  // elements. Valid types for the control_points tuple are, therefore,
  // either typeified enumeration values, or cycles.

  using control_points =
    list<point<cp::initialize>, main_cycle, point<cp::finalize>>;

private:
  size_t step_{0};
};

using control = flecsi::run::control<control_policy>;

} // namespace cycle

#endif
