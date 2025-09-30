#include "tasks/smooth.hh"

using namespace flecsi;

void
poisson::task::red(exec::accelerator s,
  mesh::accessor<ro> m,
  field<double>::accessor<rw, ro> ua,
  field<double>::accessor<ro, na> fa) noexcept {
  auto u = m.mdspan<mesh::vertices>(ua);
  auto f = m.mdspan<mesh::vertices>(fa);
  const auto dxdy = m.dxdy();
  const auto dx_over_dy = m.xdelta() / m.ydelta();
  const auto dy_over_dx = m.ydelta() / m.xdelta();
  const auto factor = 1.0 / (2 * (dx_over_dy + dy_over_dx));

  // clang-format off
  s.executor().named("red").forall(j, m.vertices<mesh::y_axis>()) {
    for(auto i : m.red<mesh::x_axis>(j)) {
      u[j][i] = factor *
                (dxdy * f[j][i] +
                 dy_over_dx * (u[j][i + 1] + u[j][i - 1]) +
                 dx_over_dy * (u[j + 1][i] + u[j - 1][i]));
    } // for
  }; // forall
  // clang-format on
} // smooth

void
poisson::task::black(exec::accelerator s,
  mesh::accessor<ro> m,
  field<double>::accessor<rw, ro> ua,
  field<double>::accessor<ro, na> fa) noexcept {
  auto u = m.mdspan<mesh::vertices>(ua);
  auto f = m.mdspan<mesh::vertices>(fa);
  const auto dxdy = m.dxdy();
  const auto dx_over_dy = m.xdelta() / m.ydelta();
  const auto dy_over_dx = m.ydelta() / m.xdelta();
  const auto factor = 1.0 / (2 * (dx_over_dy + dy_over_dx));

  // clang-format off
  s.executor().named("black").forall(j, m.vertices<mesh::y_axis>()) {
    for(auto i: m.black<mesh::x_axis>(j)) {
      u[j][i] = factor *
                (dxdy * f[j][i] +
                 dy_over_dx * (u[j][i + 1] + u[j][i - 1]) +
                 dx_over_dy * (u[j + 1][i] + u[j - 1][i]));
    } // for
  }; // forall
  // clang-format on
} // smooth
