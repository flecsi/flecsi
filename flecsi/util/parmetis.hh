// Copyright (C) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_PARMETIS_HH
#define FLECSI_UTIL_PARMETIS_HH

#include <flecsi-config.h>

#include "flecsi/util/color_map.hh"
#include "flecsi/util/crs.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/types.hh"

#include <parmetis.h>

#include <vector>

/// \cond core
namespace flecsi {
namespace util {
namespace parmetis {
/// \addtogroup utils
/// \{

inline auto
with_zero(const util::offsets & o) {
  std::vector<idx_t> ret;
  ret.reserve(o.size() + 1);
  ret.push_back(0);
  auto & v = o.ends();
  ret.insert(ret.end(), v.begin(), v.end());
  return ret;
}

/// Generate a coloring of the given naive graph partition into \em colors
/// colors.  This function uses \c ParMETIS_V3_PartKway.  Each
/// process in the comm must participate.
/// \param dist distribution of entities over ranks
/// \param graph local connectivity graph
/// \param colors The number of partitions to create.
/// \param comm   An MPI_Comm object that defines the number of processes.
inline std::vector<Color>
color(const util::offsets & dist,
  const crs & graph,
  idx_t colors,
  MPI_Comm comm = MPI_COMM_WORLD) {

  auto [rank, size] = util::mpi::info(comm);

  flog_assert(dist.size() == size_t(size),
    "distribution size (" << colors << ") must equal comm size(" << size
                          << ")");

  idx_t wgtflag = 0;
  idx_t numflag = 0;
  idx_t ncon = 1;
  std::vector<real_t> tpwgts(ncon * colors, 1.0 / colors);

  // We may need to expose some of the ParMETIS configuration options.
  std::vector<real_t> ubvec(ncon, 1.05);
  idx_t options[3]{};
  idx_t edgecut;

  std::vector<idx_t> part(graph.size());

  std::vector<idx_t> vtxdist = with_zero(dist), xadj = with_zero(graph.offsets);
  std::vector<idx_t> adjncy = as<idx_t>(graph.values);
  // ParMETIS rejects certain trivial cases and nullptr+[0,0) ranges.
  if(adjncy.empty())
    adjncy.emplace_back();

  auto sub = util::mpi::comm::split(comm, part.empty() ? MPI_UNDEFINED : 0);

  if(sub) {
    // clang-format off
    int result = ParMETIS_V3_PartKway(&vtxdist[0], &xadj[0], &adjncy[0],
        nullptr, nullptr, &wgtflag, &numflag, &ncon, &colors, &tpwgts[0],
        ubvec.data(), options, &edgecut, part.data(), &sub.c);
    // clang-format on

    flog_assert(result == METIS_OK, "ParMETIS_V3_PartKway returned " << result);
  }

  return {part.begin(), part.end()};
} // color

/// \}
} // namespace parmetis
} // namespace util
} // namespace flecsi
/// \endcond

#endif
