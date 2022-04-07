/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#ifndef POISSON_OPTIONS_HH
#define POISSON_OPTIONS_HH

#include "flecsi/execution.hh"

#include <limits>

namespace poisson {

inline flecsi::program_option<std::size_t>
  x_extents("x-extents", "The x extents of the mesh.", 1);
inline flecsi::program_option<std::size_t>
  y_extents("y-extents", "The y extents of the mesh.", 1);
inline flecsi::program_option<std::size_t> max_iterations("max-iterations",
  "max_iterations,m",
  "The maximum number of solver iterations.",
  {{flecsi::option_default, std::numeric_limits<std::size_t>::max()}});
inline flecsi::program_option<double> error_tol("error-tolerance",
  "tolerance,t",
  "The maximum absolute residual tolerance.",
  {{flecsi::option_default, 1.0e-4}});
} // namespace poisson

#endif
