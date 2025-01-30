// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_COPY_HH
#define FLECSI_DATA_LOCAL_COPY_HH

// High-level topology type implementation.

#include "flecsi/data/local/storage.hh"

namespace flecsi::data {
namespace local {

struct copy_base {
  using index_type = std::size_t;

  using Point = std::pair<index_type, index_type>; // (rank, index)
  static Point point(std::size_t r, std::size_t i) {
    return {r, i};
  }
};

struct copy_engine : copy_base {
  template<typename AllToAll>
  copy_engine(const prefixes & src,
    const data::intervals & intervals,
    field_id_t fid,
    AllToAll && all_to_all)
    : source(&src->get_region()), destination(intervals.share()) {
    // Make sure the task that is writing to the field has finished running
    (*destination)[fid].synchronize();
    // The input comprises the color and index of shared elements stored at
    // each ghost element; reverse those pointers to know what to send where.

    auto remote_sources = destination->get_storage<Point, ro>(fid);

    // Calculate the memory needed up front for the ghost_entities
    std::map<Color, std::size_t> mem_size;
    for(const auto & [begin, end] : destination->ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        mem_size[shared.first]++;
      }
    }

    for(auto & p : mem_size)
      ghost_entities[p.first].resize(std::exchange(p.second, 0));

    // Essentially a GroupByKey of remote_sources, keys are the remote source
    // ranks and values are vectors of remote source indices.
    std::map<Color, std::vector<index_type>> remote_shared_entities;
    for(const auto & [begin, end] : destination->ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        remote_shared_entities[shared.first].emplace_back(shared.second);
        // We also group local ghost entities into
        // (src rank, { local ghost ids})

        // GCC 12.2.0 thinks this is dependent:
        ghost_entities[shared.first]
          .template data<rw>()[mem_size[shared.first]++] = ghost_idx;
      }
    }

    // Create the inverse mapping of remote_shared_entities. This creates a map
    // from remote destination rank to a vector of *local* source indices. This
    // information is later used by MPI_Send().
    {
      std::size_t r = 0;
      for(auto & v : all_to_all(remote_shared_entities)) {
        if(!v.empty()) {
          shared_entities[r].resize(v.size());
          std::uninitialized_copy(
            v.begin(), v.end(), shared_entities[r].data<rw>().data());
        }
        ++r;
      }
    }

    // We need to figure out the max local source index in order to give correct
    // nelems when calling region::get_storage().
    for(const auto & [rank, indices] : shared_entities) {
      auto indices_view = indices.data();
      max_local_source_idx = std::max(max_local_source_idx,
        *std::max_element(
          indices_view.data(), indices_view.data() + indices_view.size()));
      max_shared_indices_size =
        std::max(max_shared_indices_size, indices_view.size());
    }
    max_local_source_idx += 1;
  }

  // (remote rank, { local indices })
  using SendPoints = std::map<Color, local::detail::storage<index_type>>;

  region_impl * source; // kept alive by subsequent tasks
  intervals::ref destination;
  SendPoints ghost_entities; // (src rank,  { local ghost indices})
  SendPoints shared_entities; // (dest rank, { local shared indices})
  std::size_t max_local_source_idx = 0, max_shared_indices_size = 0;
};

} // namespace local
} // namespace flecsi::data

#endif // FLECSI_DATA_LOCAL_COPY_HH
