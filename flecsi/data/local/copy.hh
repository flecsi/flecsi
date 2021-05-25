// Copyright (c) 2016, Triad National Security, LLC
// All rights reserved.

#ifndef FLECSI_DATA_LOCAL_COPY_HH
#define FLECSI_DATA_LOCAL_COPY_HH

// High-level topology type implementation.

#include "flecsi/topo/color.hh"

namespace flecsi::data {
namespace local {
// This differs from topo::claims only in that the field is dense for
// compatibility with leg::halves.
struct claims : topo::specialization<topo::column, claims> {
  using Field = flecsi::field<borrow::Value>;
  static const Field::definition<claims> field;
};
inline const claims::Field::definition<claims> claims::field;
} // namespace local

struct pointers : local::claims::core {
  using Value = std::size_t;
  static constexpr auto & field = local::claims::field;

  // execution.hh is available here, but not accessor.hh.
  pointers(prefixes &, topo::claims::core & src);

  auto operator*() {
    return field(*this);
  }

private:
  static void expand(topo::claims::Field::accessor<ro>,
    std::size_t w,
    local::claims::Field::accessor<wo>);
};

namespace local {

struct copy_engine {
  // One copy engine for each entity type i.e. vertex, cell, edge.
  copy_engine(const data::points & pts,
    const data::intervals & intervals,
    field_id_t meta_fid /* for remote shared entities */)
    : source(pts), destination(intervals), meta_fid(meta_fid) {}

  // The initialization of the copy_engine may have to be delayed to make sure
  // the calls to get_storage can be safely executed.
  template<typename AllToAll>
  void init_copy_engine(AllToAll && all_to_all) {
    // Make sure the task that is writing to the field has finished running
    destination[meta_fid].synchronize();
    // There is no information about the indices of local shared entities,
    // ranks and indices of the destination of copy i.e. (local source
    // index, {(remote dest rank, remote dest index)}). We need to do a shuffle
    // operation to reconstruct this info from {(local ghost index, remote
    // source rank, remote source index)}.
    auto remote_sources =
      destination.get_storage<const data::points::Value>(meta_fid);
    // Essentially a GroupByKey of remote_sources, keys are the remote source
    // ranks and values are vectors of remote source indices.
    SendPoints remote_shared_entities;
    for(const auto & [begin, end] : destination.ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        remote_shared_entities[shared.first].emplace_back(shared.second);
        // We also group local ghost entities into
        // (src rank, { local ghost ids})
        ghost_entities[shared.first].emplace_back(ghost_idx);
      }
    }

    // Create the inverse mapping of group_shared_entities. This creates a map
    // from remote destination rank to a vector of *local* source indices. This
    // information is later used by MPI_Send().
    {
      std::size_t r = 0;
      for(auto & v : all_to_all(remote_shared_entities)) {
        if(!v.empty())
          shared_entities.try_emplace(r, std::move(v));
        ++r;
      }
    }

    // We need to figure out the max local source index in order to give correct
    // nelems when calling region::get_storage().
    for(const auto & [rank, indices] : shared_entities) {
      max_local_source_idx = std::max(max_local_source_idx,
        *std::max_element(indices.begin(), indices.end()));
    }
    max_local_source_idx += 1;
  }

  // the operator()() is implemented in the derived copy_engine instance

protected:
  // (remote rank, { local indices })
  using SendPoints = std::map<Color, std::vector<std::size_t>>;

  const data::points & source;
  const data::intervals & destination;
  field_id_t meta_fid;
  SendPoints ghost_entities; // (src rank,  { local ghost indices})
  SendPoints shared_entities; // (dest rank, { local shared indices})
  std::size_t max_local_source_idx = 0;
};

} // namespace local
} // namespace flecsi::data

#endif // FLECSI_DATA_LOCAL_COPY_HH
