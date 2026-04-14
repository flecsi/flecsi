#ifndef POISSON_STATE_HH
#define POISSON_STATE_HH

#include "specialization/mesh.hh"

namespace poisson {

inline const flecsi::field<double>::definition<mesh, mesh::vertices> ud, fd, sd,
  Aud;

} // namespace poisson

#endif
