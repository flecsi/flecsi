#ifndef TUTORIAL_6_TOPOLOGY_SPH_PHYSICS_HH
#define TUTORIAL_6_TOPOLOGY_SPH_PHYSICS_HH

#include <flecsi/execution.hh>

namespace sph {

namespace constants {
const inline double sigma = 4. / 3.;
const inline double alpha = 1.;
const inline double beta = 2.;
const inline double gamma = 5. / 3.;
const inline double dt = 1.e-3;
const inline double minh = 0.001;
const inline double epsilon = 0.001 * minh * minh;
const inline double rho_l = 0.125;
const inline double rho_h = 1.;
} // namespace constants

FLECSI_INLINE_TARGET double
kernel(double r, double h) {
  const auto q = r / h;
  const double sigma = constants::sigma / h;
  double result = 0.;
  if(q < 0)
    ;
  else if(q < 1)
    result = sigma * (1. - 1.5 * q * q + .75 * q * q * q);
  else if(q <= 2)
    result = sigma * (.25 * (2 - q) * (2 - q) * (2 - q));
  assert(result >= 0.);
  return result;
}

FLECSI_INLINE_TARGET double
grad_kernel(double rij, double h) {
  const auto r = std::abs(rij);
  const auto q = r / h;
  const double sigma = constants::sigma / (h * h);
  double result = 0.;
  if(q < 0)
    ;
  else if(q < 1)
    result = sigma * (-3 * q + 9. / 4. * q * q) * rij;
  else if(q <= 2)
    result = sigma * (-.75 * (2 - q) * (2 - q)) * rij;
  return result;
}

FLECSI_INLINE_TARGET double
mu(double r, double v, double hab) {
  const double eps = constants::epsilon * hab * hab;
  return hab * r * v / (r * r + eps);
}

FLECSI_INLINE_TARGET double
sound_speed(double u) {
  return std::sqrt(constants::gamma * (constants::gamma - 1.) * u);
}

FLECSI_INLINE_TARGET double
viscosity(double r, double v, double rho_ab, double hab, double cs) {
  double visc = 0.;
  if(r * v < 0) {
    auto m = mu(r, v, hab);
    visc = (-constants::alpha * cs * m + constants::beta * m * m) / rho_ab;
  }
  return visc;
}

template<typename T>
void
eos(T & t,
  flecsi::util::span<const double> rho,
  flecsi::util::span<const double> u,
  flecsi::util::span<double> p) {
  forall(e, t.entities(), "compute EOS") {
    p[e] = (constants::gamma - 1) * rho[e] * u[e];
  };
}

template<typename T>
void
init_base(T & e, std::size_t global_nents, std::size_t offset) {
  const double h = 1. / static_cast<double>(global_nents);
  for(std::size_t i = 0; i < e.size(); ++i) {
    e[i].radius_ = 2 * h + h / 4.;
    e[i].coordinates_ = h * (offset + i);
    e[i].id_ = offset + i;
    e[i].mass_ =
      e[i].coordinates_[0] < 0.5 ? constants::rho_h * h : constants::rho_l * h;
  }
}

template<typename T>
void
init_physics(T & t,
  flecsi::util::span<double> v,
  flecsi::util::span<double> rho,
  flecsi::util::span<double> p,
  flecsi::util::span<double> u,
  flecsi::util::span<bool> is_w) {
  forall(a, t.entities(), "Init Physics") {
    v[a] = 0.;
    if(t.e_i[a].coordinates[0] < 0.5) {
      rho[a] = 1.0;
      p[a] = 1.0;
    }
    else {
      rho[a] = 0.125;
      p[a] = 0.1;
    }
    u[a] = p[a] / ((constants::gamma - 1) * rho[a]);
    is_w[a] =
      !((t.e_i(a).coordinates[0] > 0.05) && (t.e_i(a).coordinates[0] < 0.95));
  };
}

// Compute density
template<typename T>
void
density(T && t, flecsi::util::span<double> rho) {
  forall(e, t.entities(), "density") {
    rho[e] = 0;
    for(auto n : t.neighbors(e)) {
      const double h = (t.e_i[e].radius + t.e_i[n].radius) * 0.5;
      const double x =
        std::abs(t.e_i[e].coordinates[0] - t.e_i[n].coordinates[0]);
      rho[e] += t.e_i[n].mass * kernel(x, h);
    }
  };
} // density

template<typename T>
void
acceleration(T & t,
  flecsi::util::span<const double> v,
  flecsi::util::span<const double> rho,
  flecsi::util::span<const double> p,
  flecsi::util::span<const double> u,
  flecsi::util::span<double> dvdt) {
  for(auto e : t.entities()) {
    dvdt[e] = 0;
    const double h = t.e_i[e].radius;
    auto pp = p[e] / (rho[e] * rho[e]);
    for(auto n : t.neighbors(e)) {
      if(n == e)
        continue;
      const auto h_ab = (h + t.e_i[n].radius) / 2.;
      const auto ppb = p[n] / (rho[n] * rho[n]);
      const auto r_ab = t.e_i[e].coordinates[0] - t.e_i[n].coordinates[0];
      const auto vel_ab = v[e] - v[n];
      const auto rho_ab = (rho[e] + rho[n]) / 2.;
      const auto cs_ab = (sound_speed(u[e]) + sound_speed(u[n])) / 2.;
      const auto visc = viscosity(r_ab, vel_ab, rho_ab, h_ab, cs_ab);
      dvdt[e] += -t.e_i[n].mass * (pp + ppb + visc) * grad_kernel(r_ab, h_ab);
    }
  }
}

template<typename T>
void
dudt(T & t,
  flecsi::util::span<const double> v,
  flecsi::util::span<const double> rho,
  flecsi::util::span<const double> p,
  flecsi::util::span<const double> u,
  flecsi::util::span<const bool> is_w,
  flecsi::util::span<double> dudt) {
  for(auto e : t.entities()) {
    const double h = t.e_i[e].radius;
    dudt[e] = 0.;
    if(is_w[e])
      continue;
    auto pp = p[e] / (rho[e] * rho[e]);
    for(auto n : t.neighbors(e)) {
      if(n == e)
        continue;
      const auto h_ab = (h + t.e_i[n].radius) / 2.;
      const auto r_ab = t.e_i[e].coordinates[0] - t.e_i[n].coordinates[0];
      const auto vel_ab = v[e] - v[n];
      const auto cs_ab = (sound_speed(u[e]) + sound_speed(u[n])) / 2.;
      const auto rho_ab = (rho[e] + rho[n]) / 2.;
      const auto visc = viscosity(r_ab, vel_ab, rho_ab, h_ab, cs_ab);
      dudt[e] +=
        t.e_i[n].mass * (pp + visc / 2.) * vel_ab * grad_kernel(r_ab, h_ab);
    }
  }
}

template<typename T>
void
advance(T & t,
  flecsi::util::span<double> v,
  flecsi::util::span<const double> acc,
  flecsi::util::span<double> ie,
  flecsi::util::span<const double> die,
  flecsi::util::span<const bool> is_w) {
  for(auto e : t.entities()) {
    if(is_w[e])
      continue;
    v[e] += constants::dt * acc[e];
    t.e_i[e].coordinates[0] += constants::dt * v[e];
    ie[e] += constants::dt * die[e];
  }
}

} // namespace sph

#endif
