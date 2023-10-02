#ifndef POISSON_INITIALIZE_HH
#define POISSON_INITIALIZE_HH

#include "specialization/control.hh"

namespace poisson {
namespace action {

void init_mesh(control_policy &);
inline control::action<init_mesh, cp::initialize> init_mesh_action;

} // namespace action
} // namespace poisson

#endif
