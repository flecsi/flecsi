#ifndef POISSON_PROBLEM_HH
#define POISSON_PROBLEM_HH

#include "initialize.hh"
#include "specialization/control.hh"

namespace poisson {
namespace action {

int problem();
inline control::action<problem, cp::initialize> problem_action;
inline auto const problem_dep = problem_action.add(init_mesh_action);

} // namespace action
} // namespace poisson

#endif
