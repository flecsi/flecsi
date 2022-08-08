// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef TUTORIAL_2_CONTROL_4_STATE_HH
#define TUTORIAL_2_CONTROL_4_STATE_HH

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

namespace placeholder {

// The following class exists solely for pedagogical purposes.  It
// demonstrates that values of arbitrary type can be manipulated by a
// FleCSI control model, even if said type is not itself "well-behaved"
// or FleCSI-aware.  Think of custom as a std::vector or some
// Trilinos data structure, for example.
template<typename T>
class custom
{
private:
  std::unique_ptr<T[]> data;

public:
  explicit custom(std::size_t sz) : data(std::make_unique<T[]>(sz)) {}

  T & operator[](std::size_t idx) {
    return data[idx];
  }

  const T & operator[](std::size_t idx) const {
    return data[idx];
  }
};

} // namespace placeholder

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

  static bool cycle_control(control_policy & policy) {
    return policy.step()++ < policy.steps();
  }

  using main_cycle = cycle<cycle_control, point<cp::advance>>;

  using control_points = list<point<cp::allocate>,
    point<cp::initialize>,
    main_cycle,
    point<cp::finalize>>;

  /*--------------------------------------------------------------------------*
    State interface
   *--------------------------------------------------------------------------*/

  using int_custom = placeholder::custom<int>;

  std::size_t & step() {
    return step_;
  }

  std::size_t & steps() {
    return steps_;
  }

  void allocate_values(std::size_t size) {
    // Instead of new we use make_unique to allocate a unique pointer.
    values_ = std::make_unique<int_custom>(size);
  }

  void deallocate_values() {
    // Instead of delete[] we reset the unique pointer to nullptr, which
    // frees the custom object.
    values_.reset();
  }

  int_custom & values() {
    return *values_;
  }

private:
  /*--------------------------------------------------------------------------*
    State members
   *--------------------------------------------------------------------------*/

  std::size_t step_{0};
  std::size_t steps_{0};
  std::unique_ptr<int_custom> values_;
};

using control = flecsi::run::control<control_policy>;

} // namespace state

#endif
