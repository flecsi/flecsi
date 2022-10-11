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

  mesh::gcoord axis_extents{x_extents.value(), y_extents.value()};

  coloring.allocate(flecsi::processes(), axis_extents);

  mesh::grect geometry;
  geometry[0][0] = 0.0;
  geometry[0][1] = 1.0;
  geometry[1] = geometry[0];

  m.allocate(coloring.get(), geometry);
} // init_mesh
