/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#include "tasks/smooth.hh"

using namespace flecsi;

void
poisson::task::smooth(mesh::accessor<ro> m,
  field<double>::accessor<rw, ro> ua,
  field<double>::accessor<ro, ro> fa) {
  auto u = m.mdspan<mesh::vertices>(ua);
  auto f = m.mdspan<mesh::vertices>(fa);
  const auto dsqr = pow(m.delta(), 2);

  // clang-format off
  for(auto j : m.vertices<mesh::y_axis>()) {
    for(auto i : m.vertices<mesh::x_axis>()) {
      u[i][j] = 0.25 * (dsqr * f[i][j] +
        u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1]);
    } // for
  } // for
  // clang format on
} // smooth
