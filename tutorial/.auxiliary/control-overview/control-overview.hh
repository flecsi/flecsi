#ifndef TUTORIAL_AUXILIARY_CONTROL_OVERVIEW_CONTROL_OVERVIEW_HH
#define TUTORIAL_AUXILIARY_CONTROL_OVERVIEW_CONTROL_OVERVIEW_HH

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

namespace example {

enum class cp { one, two, three, four };

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::one:
      return "Control Point 1";
    case cp::two:
      return "Control Point 2";
    case cp::three:
      return "Control Point 3";
    case cp::four:
      return "Control Point 4";
  }
  flog_fatal("invalid control point");
}

struct control_policy : flecsi::run::control_base {
  using control_points_enum = cp;
  struct node_policy {};

  using control = flecsi::run::control<control_policy>;

  size_t & step() {
    return step_;
  }

  static bool step_control() {
    return control::policy().step()++ < 5;
  }

  using main_cycle = cycle<step_control, point<cp::two>, point<cp::three>>;

  using control_points = list<point<cp::one>, main_cycle, point<cp::four>>;

private:
  size_t step_{0};
};

using control = flecsi::run::control<control_policy>;

} // namespace example

#endif
