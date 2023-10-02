#ifndef TUTORIAL_2_CONTROL_1_SIMPLE_HH
#define TUTORIAL_2_CONTROL_1_SIMPLE_HH

#include "flecsi/flog.hh"
#include "flecsi/run/control.hh"

namespace simple {

// Enumeration defining the control point identifiers. This will be used to
// specialize the core control type.

enum class cp { initialize, advance, finalize };

// Define labels for the control points.  Using a function allows error
// checking.

inline const char *
operator*(cp control_point) {
  switch(control_point) {
    case cp::initialize:
      return "initialize";
    case cp::advance:
      return "advance";
    case cp::finalize:
      return "finalize";
  }
  flog_fatal("invalid control point");
}

// Control policy for this example. The control policy primarily defines types
// that are used by the core control type to define the control-flow model for
// the program. For this simple example, the policy captures the user-defined
// enumeration of control-points identifiers, defines an empty node policy, and
// defines the order of control point execution using a list.

struct control_policy : flecsi::run::control_base {

  // Capture the control points enumeration type. This type is used in the
  // control policy interface whenever a control point is required.

  using control_points_enum = cp;

  // The actions that are added under each control point are used to form a
  // directed acyclic graph (DAG). The node_policy type allows the user to add
  // interfaces and data that are available from, and are carried with the
  // action. A more complex example will demonstrate this capability.

  struct node_policy {};

  // The control_points list defines the actual control points as typeified
  // integers derived from the control point identifiers from the user-defined
  // enumeration.

  using control_points =
    list<point<cp::initialize>, point<cp::advance>, point<cp::finalize>>;
}; // struct control_policy

// Define a fully-qualified control type for the end user.

using control = flecsi::run::control<control_policy>;

} // namespace simple

#endif
