#ifndef POISSON_STATE_HH
#define POISSON_STATE_HH

#include "specialization/mesh.hh"

namespace poisson {

inline const field<double>::definition<mesh, mesh::vertices> ud;
inline const field<double>::definition<mesh, mesh::vertices> fd;
inline const field<double>::definition<mesh, mesh::vertices> sd;
inline const field<double>::definition<mesh, mesh::vertices> Aud;

inline mesh::slot m;
inline mesh::cslot coloring;

} // namespace poisson

#endif
