/*----------------------------------------------------------------------------*
  Copyright (c) 2020 Triad National Security, LLC
  All rights reserved
 *----------------------------------------------------------------------------*/

#include "tasks/io.hh"

#include <sstream>

using namespace flecsi;

void
poisson::task::io(mesh::accessor<ro> m,
  field<double>::accessor<ro, ro> ua,
  std::string filebase) {
  auto u = m.mdspan<mesh::vertices>(ua);

  std::stringstream ss;
  ss << filebase;
  if(processes() == 1) {
    ss << ".dat";
  }
  else {
    ss << "-" << process() << ".dat";
  } // if

  std::ofstream solution(ss.str(), std::ofstream::out);

  for(auto j : m.vertices<mesh::y_axis, mesh::logical>()) {
    const double y = m.value<mesh::y_axis>(j);
    for(auto i : m.vertices<mesh::x_axis, mesh::logical>()) {
      const double x = m.value<mesh::x_axis>(i);
      solution << x << " " << y << " " << u[j][i] << std::endl;
    } // for
  } // for
} // io
