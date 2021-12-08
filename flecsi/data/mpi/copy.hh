// High-level topology type implementation.

#ifndef FLECSI_DATA_MPI_COPY_HH
#define FLECSI_DATA_MPI_COPY_HH

#include "flecsi/topo/color.hh"

namespace flecsi::data {
namespace mpi {
// This differs from topo::claims only in that the field is dense for
// compatibility with leg::halves.
struct claims : topo::specialization<topo::column, claims> {
  using Field = flecsi::field<borrow::Value>;
  static const Field::definition<claims> field;
};
inline const claims::Field::definition<claims> claims::field;
} // namespace mpi

struct pointers : mpi::claims::core {
  using Value = std::size_t;
  static constexpr auto & field = mpi::claims::field;

  // execution.hh is available here, but not accessor.hh.
  pointers(prefixes &, topo::claims::core & src);

  auto operator*() {
    return field(*this);
  }

private:
  static void expand(topo::claims::Field::accessor<ro>,
    std::size_t w,
    mpi::claims::Field::accessor<wo>);
};
} // namespace flecsi::data

#endif
