#include "tasks/norm.hh"

using namespace flecsi;

double
poisson::task::diff(mesh::accessor<ro> m,
  field<double>::accessor<ro, ro> aa,
  field<double>::accessor<ro, ro> ba) {
  auto a = m.mdspan<mesh::vertices>(aa);
  auto b = m.mdspan<mesh::vertices>(ba);

  double sum{0};
  for(auto j : m.vertices<mesh::y_axis>()) {
    for(auto i : m.vertices<mesh::x_axis>()) {
      sum += pow(a[j][i] - b[j][i], 2);
    } // for
  } // for

  return sum;
} // diff

double
poisson::task::scale(mesh::accessor<ro> m, double sum) {
  return m.dxdy() * sum;
} // scale

void
poisson::task::discrete_operator(mesh::accessor<ro> m,
  field<double>::accessor<ro, ro> ua,
  field<double>::accessor<rw, ro> Aua) {
  auto u = m.mdspan<mesh::vertices>(ua);
  auto Au = m.mdspan<mesh::vertices>(Aua);

  const double w = 1.0 / m.dxdy();
  const auto dx_over_dy = m.xdelta() / m.ydelta();
  const auto dy_over_dx = m.ydelta() / m.xdelta();

  // clang-format off
  for(auto j : m.vertices<mesh::y_axis>()) {
    for(auto i : m.vertices<mesh::x_axis>()) {
      Au[j][i] = w * (2.0 * (dx_over_dy + dy_over_dx) * u[j][i] -
                      dy_over_dx * (u[j][i + 1] + u[j][i - 1]) -
                      dx_over_dy * (u[j + 1][i] + u[j - 1][i]));
    } // for
  } // for
  // clang-format on
} // residual
