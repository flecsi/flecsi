/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#ifndef POISSON_FINALIZE_HH
#define POISSON_FINALIZE_HH

#include "specialization/control.hh"

namespace poisson {
namespace action {

int finalize();
inline control::action<finalize, cp::finalize> finalize_action;

} // namespace action
} // namespace poisson

#endif
