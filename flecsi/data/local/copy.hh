// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_COPY_HH
#define FLECSI_DATA_LOCAL_COPY_HH

// High-level topology type implementation.

#include "flecsi/data/local/storage.hh"

namespace flecsi::data {
namespace local {

struct copy_engine {
  using index_type = std::size_t;

  // One copy engine for each entity type i.e. vertex, cell, edge.
  template<typename AllToAll>
  copy_engine(const data::points & pts,
    const data::intervals & intervals,
    field_id_t meta_fid /* for remote shared entities */,
    AllToAll && all_to_all)
    : source(pts.r), destination(intervals.share()) {
    // Make sure the task that is writing to the field has finished running
    (*destination)[meta_fid].synchronize();
    // There is no information about the indices of local shared entities,
    // ranks and indices of the destination of copy i.e. (local source
    // index, {(remote dest rank, remote dest index)}). We need to do a shuffle
    // operation to reconstruct this info from {(local ghost index, remote
    // source rank, remote source index)}.
    auto remote_sources =
      destination->get_storage<data::points::Value, ro>(meta_fid);

    // Calculate the memory needed up front for the ghost_entities
    std::map<Color, std::size_t> mem_size;
    for(const auto & [begin, end] : destination->ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        mem_size[shared.first]++;
      }
    }

    // allocate the memory needed
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
          .template data<exec::task_processor_type_t::loc,
            flecsi::rw>()[mem_size[shared.first]++] = ghost_idx;
      }
    }

    // Create the inverse mapping of group_shared_entities. This creates a map
    // from remote destination rank to a vector of *local* source indices. This
    // information is later used by MPI_Send().
    {
      std::size_t r = 0;
      for(auto & v : all_to_all(remote_shared_entities)) {
        if(!v.empty()) {
          shared_entities[r].resize(v.size());
          std::uninitialized_copy(v.begin(),
            v.end(),
            shared_entities[r]
              .data<exec::task_processor_type_t::loc, rw>()
              .data());
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
