/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#ifndef POISSON_ANALYZE_HH
#define POISSON_ANALYZE_HH

#include "specialization/control.hh"

namespace poisson {
namespace action {

int analyze(control_policy &);
inline control::action<analyze, cp::analyze> analyze_action;

} // namespace action
} // namespace poisson

#endif
