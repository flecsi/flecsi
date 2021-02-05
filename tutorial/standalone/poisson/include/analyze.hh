/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#pragma once

#include "specialization/control.hh"

namespace poisson {
namespace action {

int analyze();
inline control::action<analyze, cp::analyze> analyze_action;

} // namespace action
} // namespace poisson
