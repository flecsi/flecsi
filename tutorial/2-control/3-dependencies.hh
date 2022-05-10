// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef TUTORIAL_2_CONTROL_3_DEPENDENCIES_HH
#define TUTORIAL_2_CONTROL_3_DEPENDENCIES_HH

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

namespace dependencies {

enum class cp { cp1, cp2 };

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::cp1:
      return "Control Point 1";
    case cp::cp2:
      return "Control Point 2";
  }
  flog_fatal("invalid control point");
}

struct control_policy : flecsi::run::control_base {

  using control_points_enum = cp;
  struct node_policy {};

  using control = flecsi::run::control<control_policy>;

  using control_points = list<point<cp::cp1>, point<cp::cp2>>;
};

using control = flecsi::run::control<control_policy>;

} // namespace dependencies

#endif
