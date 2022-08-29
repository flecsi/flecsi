// Copyright (c) 2016, Los Alamos National Security, LLC
// All rights reserved.

#ifndef FLECSI_UTIL_PARMETIS_HH
#define FLECSI_UTIL_PARMETIS_HH

#include <flecsi-config.h>

#include "flecsi/util/color_map.hh"
#include "flecsi/util/crs.hh"
#include "flecsi/util/mpi.hh"
#include "flecsi/util/types.hh"

#if !defined(FLECSI_ENABLE_PARMETIS)
#error FLECSI_ENABLE_PARMETIS not defined! This file depends on ParMETIS!
#endif

#include <parmetis.h>

#include <vector>

namespace flecsi {
namespace util {
namespace parmetis {
/// \addtogroup utils
/// \{

/// Generate a coloring of the given naive graph partition defined by
/// \em naive into \em colors colors. This function uses ParMETIS' \em
/// ParMETIS_V3_PartKway interface to perform a k-way partitioning
/// of the graph over the number of processes defined in \em comm. Each
/// process in the comm must participate.
/// \param naive  A distributed compressed-row-storage representation
///               of the connectivity graph.
/// \param colors The number of partitions to create.
/// \param comm   An MPI_Comm object that defines the number of processes.
static std::vector<Color>
color(dcrs const & naive, idx_t colors, MPI_Comm comm = MPI_COMM_WORLD) {

  auto [rank, size] = util::mpi::info(comm);

  flog_assert(naive.distribution.size() == size_t(size),
    "invalid naive coloring! naive.colors("
      << colors << ") must equal comm size(" << size << ")");

  idx_t wgtflag = 0;
  idx_t numflag = 0;
  idx_t ncon = 1;
  std::vector<real_t> tpwgts(ncon * colors, 1.0 / colors);

  // We may need to expose some of the ParMETIS configuration options.
  std::vector<real_t> ubvec(ncon, 1.05);
  idx_t options[4] = {0, 0, 0, 0};
  idx_t edgecut;

  std::vector<idx_t> part(naive.size());

  if(size != colors) {
    options[0] = 1;
    options[3] = PARMETIS_PSR_UNCOUPLED;

    const equal_map cm(naive.distribution.total(), colors);

    for(size_t i{0}; i < naive.size(); ++i) {
      part[i] = cm.bin(naive.distribution(rank) + i);
    } // for
  } // if

  std::stringstream ss;
  ss << "part size: " << part.size() << std::endl;
  for(auto i : part) {
    ss << i << " ";
  }
  ss << std::endl;
  flog_devel(info) << ss.str() << std::endl;

  std::vector<idx_t> vtxdist(1);
  {
    auto & v = naive.distribution.ends();
    vtxdist.insert(vtxdist.end(), v.begin(), v.end());
  }
  std::vector<idx_t> xadj = as<idx_t>(naive.offsets);
  std::vector<idx_t> adjncy = as<idx_t>(naive.indices);

  // clang-format off
  int result = ParMETIS_V3_PartKway(&vtxdist[0], &xadj[0], &adjncy[0],
    nullptr, nullptr, &wgtflag, &numflag, &ncon, &colors, &tpwgts[0],
    ubvec.data(), options, &edgecut, &part[0], &comm);
  // clang-format on

  flog_assert(result == METIS_OK, "ParMETIS_V3_PartKway returned " << result);

  return {part.begin(), part.end()};
} // color

/// \}
} // namespace parmetis
} // namespace util
} // namespace flecsi

#endif
