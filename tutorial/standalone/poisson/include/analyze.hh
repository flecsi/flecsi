#ifndef POISSON_ANALYZE_HH
#define POISSON_ANALYZE_HH

#include "specialization/control.hh"

namespace poisson {
namespace action {

int analyze();
inline control::action<analyze, cp::analyze> analyze_action;

} // namespace action
} // namespace poisson

#endif
