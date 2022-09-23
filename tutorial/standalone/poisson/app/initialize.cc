#include "initialize.hh"
#include "options.hh"
#include "specialization/control.hh"
#include "state.hh"

#include <flecsi/flog.hh>

using namespace flecsi;

void
poisson::action::init_mesh(control_policy &) {
  flog(info) << "Initializing " << x_extents.value() << "x" << y_extents.value()
             << " mesh" << std::endl;
  flecsi::flog::flush();

  std::vector<std::size_t> axis_extents{x_extents.value(), y_extents.value()};

  // Distribute the number of processes over the axis colors.
  auto axis_colors = mesh::distribute(flecsi::processes(), axis_extents);

  coloring.allocate(axis_colors, axis_extents);

  mesh::grect geometry;
  geometry[0][0] = 0.0;
  geometry[0][1] = 1.0;
  geometry[1] = geometry[0];

  m.allocate(coloring.get(), geometry);
} // init_mesh
