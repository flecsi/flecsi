#include "tasks/io.hh"

#include <fstream>
#include <sstream>

using namespace flecsi;

void
poisson::task::io(exec::cpu s,
  mesh::accessor<ro> m,
  field<double>::accessor<ro, na> ua,
  std::string filebase) {
  auto u = m.mdspan<mesh::vertices>(ua);

  std::stringstream ss;
  ss << filebase;
  if(s.launch().size == 1) {
    ss << ".dat";
  }
  else {
    ss << "-" << s.launch().size << ".dat";
  } // if

  std::ofstream solution(ss.str(), std::ofstream::out);

  for(auto j : m.axis<mesh::y_axis>().layout.logical()) {
    const double y = m.value<mesh::y_axis>(j);
    for(auto i : m.axis<mesh::x_axis>().layout.logical()) {
      const double x = m.value<mesh::x_axis>(i);
      solution << x << " " << y << " " << u[j][i] << std::endl;
    } // for
  } // for
} // io
