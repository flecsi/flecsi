// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef TUTORIAL_2_CONTROL_4_STATE_HH
#define TUTORIAL_2_CONTROL_4_STATE_HH

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

namespace state {

enum class cp { allocate, initialize, advance, finalize };

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::allocate:
      return "allocate";
    case cp::initialize:
      return "initialize";
    case cp::advance:
      return "advance";
    case cp::finalize:
      return "finalize";
  }
  flog_fatal("invalid control point");
}

struct control_policy : flecsi::run::control_base {

  using control_points_enum = cp;
  struct node_policy {};

  using control = flecsi::run::control<control_policy>;

  static bool cycle_control() {
    return control::instance().step()++ < control::instance().steps();
  }

  using cycle = flecsi::run::cycle<cycle_control, point<cp::advance>>;

  using control_points = list<point<cp::allocate>,
    point<cp::initialize>,
    cycle,
    point<cp::finalize>>;

  /*--------------------------------------------------------------------------*
    State interface
   *--------------------------------------------------------------------------*/

  size_t & step() {
    return step_;
  }

  size_t & steps() {
    return steps_;
  }

  void allocate_values(size_t size) {
    values_ = new size_t[size];
  }

  void deallocate_values() {
    delete[] values_;
  }

  size_t * values() const {
    return values_;
  }

private:
  /*--------------------------------------------------------------------------*
    State members
   *--------------------------------------------------------------------------*/

  size_t step_{0};
  size_t steps_{0};
  size_t * values_;
};

using control = flecsi::run::control<control_policy>;

} // namespace state

#endif
