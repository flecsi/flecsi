#include "sph_physics.hh"

namespace sph {

double
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

double
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

double
mu(double r, double v, double hab) {
  const double eps = constants::epsilon * hab * hab;
  return hab * r * v / (r * r + eps);
}

double
sound_speed(double u) {
  return std::sqrt(constants::gamma * (constants::gamma - 1.) * u);
}

double
viscosity(double r, double v, double rho_ab, double hab, double cs) {
  double visc = 0.;
  if(r * v < 0) {
    auto m = mu(r, v, hab);
    visc = (-constants::alpha * cs * m + constants::beta * m * m) / rho_ab;
  }
  return visc;
}

} // namespace sph
