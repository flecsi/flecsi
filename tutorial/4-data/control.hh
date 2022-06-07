// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.
#ifndef CONTROL_HH
#define CONTROL_HH

#include <flecsi/flog.hh>
#include <flecsi/run/control.hh>

enum class cp { advance };

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::advance:
      return "advance";
  }
  flog_fatal("invalid control point");
}

struct control_policy : flecsi::run::control_base {

  using control_points_enum = cp;
  struct node_policy {};

  using control = flecsi::run::control<control_policy>;

  using control_points = list<point<cp::advance>>;
};

using control = flecsi::run::control<control_policy>;
#endif
