#ifndef FLECSI_RUN_TEST_CYCLE_CONTROL_HH
#define FLECSI_RUN_TEST_CYCLE_CONTROL_HH

#include "flecsi/flog.hh"
#include "flecsi/runtime.hh"

namespace example {

enum class cp { spec_init, one, two, three, four };

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::spec_init:
      return "Meta control point";
    case cp::one:
      return "Control point 1";
    case cp::two:
      return "Control point 2";
    case cp::three:
      return "Control point 3";
    case cp::four:
      return "Control point 4";
  }
  flog_fatal("invalid control point");
}

struct control_policy {
  using control_points_enum = cp;
  struct node_policy {};

  using control = flecsi::run::control<control_policy>;

  std::size_t & step() {
    return step_;
  }

  static bool step_control() {
    return control::state().step()++ < 5;
  }

  template<cp P>
  using p = flecsi::run::control_point<P>;

  using main_cycle = flecsi::run::cycle<step_control, p<cp::two>, p<cp::three>>;

  using control_points =
    flecsi::util::types<flecsi::run::control_base::meta<cp::spec_init>,
      p<cp::one>,
      main_cycle,
      p<cp::four>>;

private:
  std::size_t step_{0};
};

using control = flecsi::run::control<control_policy>;

} // namespace example

#endif
