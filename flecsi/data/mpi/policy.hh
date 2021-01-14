/*
    @@@@@@@@  @@           @@@@@@   @@@@@@@@ @@
   /@@/////  /@@          @@////@@ @@////// /@@
   /@@       /@@  @@@@@  @@    // /@@       /@@
   /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@
   /@@////   /@@/@@@@@@@/@@       ////////@@/@@
   /@@       /@@/@@//// //@@    @@       /@@/@@
   /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@
   //       ///  //////   //////  ////////  //

   Copyright (c) 2016, Triad National Security, LLC
   All rights reserved.
                                                                              */
#pragma once

#include "flecsi/data/field_info.hh"
#include "flecsi/topo/core.hh" // single_space
#include "flecsi/util/array_ref.hh"
#include "flecsi/util/mpi.hh"

#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flecsi {
namespace data {

namespace mpi {
struct region {
  // The constructor is collectively called on all ranks with the same s,
  // and fs. s.first is number of rows while s.second is number of columns.
  // MPI may assume "number of rows" == number of ranks. The number of columns
  // is a placeholder for the number of data points to be stored in a
  // particular row (aka rank), it could be exact or in the case when we don't
  // know the exact number yet, a large number (flecsi::data::logical_size) is
  // used.
  //
  // Generic frontend code supplies `s` and `fs` (for information about fields),
  // requesting memory to be (notionally) reserved from backend. Here we only
  // create the underlying std::vector<> without actually allocating any memory.
  // The client code (mostly exec::task_prologue) will call .get_storage() with
  // a field id and the number of elements on this rank, as determined by
  // partitioning of the field. We will then call .resize() on the
  // std::vector<>.
  region(size2 s, const fields & fs) : s(std::move(s)), fs(fs) {
    for(auto f : fs) {
      storages.emplace(f->fid, 0);
    }
  }

  size2 size() const {
    return s;
  }

  // The span is safe because it is used only within a user task while the
  // vectors are resized or destroyed only outside user tasks (though perhaps
  // during execute).
  template<class T>
  util::span<T> get_storage(field_id_t fid) {
    return get_storage<T>(fid, s.second);
  }

  template<class T>
  util::span<T> get_storage(field_id_t fid, std::size_t nelems) {
    auto & v = storages.at(fid);
    std::size_t nbytes = nelems * sizeof(T);
    if(nbytes > v.size())
      v.resize(nbytes);
    return {reinterpret_cast<T *>(v.data()), nelems};
  }

  auto get_field_info(field_id_t fid) const {
    for(size_t i = 0; i < fs.size(); ++i) {
      if(fs[i]->fid == fid)
        return fs[i];
    }
    throw std::runtime_error("can not find field");
  }

protected:
  void vacuous(field_id_t) {
    /* Nothing to do for  MPI backend */
  }

private:
  size2 s; // (nranks, nelems)
  fields fs; // fs[].fid is only unique within a region, i.e. r0.fs[].fid is
             // unrelated to r1.fs[].fid even if they have the same value.

  std::unordered_map<field_id_t, std::vector<std::byte>> storages;
};

struct partition {
  using row = std::size_t;
  static row make_row(std::size_t, std::size_t n) {
    return n;
  }
  static std::size_t row_size(const row & r) {
    return r;
  }

  explicit partition(region & r) : r(&r) {
    // This constructor is usually (almost always) called when r.s.second != a
    // large number, meaning it has the actual value. In this case, r.s.second
    // is the number of elements in the partition on this rank (the same value
    // on all ranks though).
    nelems = r.size().second;
  }

  partition(region & r,
    const partition & other,
    field_id_t fid,
    completeness = incomplete)
    : r(&r) {
    // Constructor for the case when how the data is partitioned is stored
    // as a field in another region referenced by the "other' partition.
    // Delegate to update().
    update(other, fid);
  }

  std::size_t colors() const {
    // number of rows, essentially the number of MPI ranks.
    return r->size().first;
  }

  template<typename T>
  auto get_storage(field_id_t fid) const {
    return r->get_storage<T>(fid, nelems);
  }

  void
  update(const partition & other, field_id_t fid, completeness = incomplete) {
    // The number of elements for each ranks is stored as a field of the
    // partition::row data type on the `other` partition.
    const auto s = other.get_storage<row>(fid); // non-owning span
    flog_assert(
      s.size() == 1, "underlying partition must have size 1, not " << s.size());
    nelems = s[0];
  }

  template<topo::single_space>
  const partition & get_partition(field_id_t) const {
    return *this;
  }

  size_t size() const {
    return nelems;
  }

private:
  region * r;
  // number of elements in this partition on this particular rank.
  size_t nelems = 0;
};
} // namespace mpi

// For backend-agnostic interface:
using region_base = mpi::region;
using mpi::partition;

struct intervals {
  using Value = subrow; // [begin, end)
  static Value make(subrow r, std::size_t = 0) {
    return r;
  }

  intervals(region_base & r,
    const partition & p,
    field_id_t fid, // The field id for the metadata in the region in p.
    completeness = incomplete)
    : r(r) {
    // Called by upper layer, supplied with a region and a partition. There are
    // two regions involved. The region `r` has the storage for real field data
    // (e.g. density, pressure etc.) as the destination of the ghost copy. It
    // also contains the pairs of (rank, index) of shared entities on remote
    // peers. The region and associated storage in the partition `p` contains
    // metadata on which entities are local ghosts (destination of copy) on the
    // current rank. The metadata is in the form of [beginning index, ending
    // index), type aliased as Value, into the index space of the entity (e.g.
    // vertex, edge, cell). We thus need to use the p.get_storage() to get the
    // metadata, not the region.get_storage() which gives the real data. This
    // works the same way as how one partition stores the number of elements in
    // a particular 'shard' on a rank for another partition. In addition, we
    // also need to copy Values from the partition and save them locally. User
    // code might change it after this constructor returns. We can not use a
    // copy assignment directly here since metadata is an util::span while
    // ghost_ranges is a std::vector<>.
    ghost_ranges = to_vector(p.get_storage<Value>(fid));
    // Get The largest value of `end index` in ghost_ranges (i.e. the upper
    // bound). This tells how much memory needs to be allocated for ghost
    // entities.
    max_end = std::max_element(
      ghost_ranges.begin(), ghost_ranges.end(), [](Value x, Value y) {
        return x.second < y.second;
      })->second;
  }

private:
  // This member function is only called by copy_engine.
  friend struct copy_engine;

  template<typename T>
  auto get_storage(field_id_t fid) const {
    return r.get_storage<T>(fid, max_end);
  }

  // We use a reference to region instead of pointer since it is not nullable
  // nor needs to be redirected after initialization.
  region_base & r;

  // Locally cached metadata on ranges of ghost index.
  std::vector<Value> ghost_ranges;
  std::size_t max_end;
};

struct points {
  using Value = std::pair<std::size_t, std::size_t>; // (rank, index)
  static Value make(std::size_t r, std::size_t i) {
    return {r, i};
  }

  points(region_base & r,
    const intervals &,
    field_id_t,
    completeness = incomplete)
    : r(r) {}

private:
  // The region `r` contains field data of shared entities on this rank as
  // source to be copied to remote peers. We make copy_engine a friend to allow
  // direct access to the region.
  friend struct copy_engine;

  region_base & r;
};

struct copy_engine {
  // One copy engine for each entity type i.e. vertex, cell, edge.
  copy_engine(const points & points,
    const intervals & intervals,
    field_id_t meta_fid /* for remote shared entities */)
    : source(points), destination(intervals), meta_fid(meta_fid) {
    // There is no information about the indices of local shared entities,
    // ranks and indices of the destination of copy i.e. (local source
    // index, {(remote dest rank, remote dest index)}). We need to do a shuffle
    // operation to reconstruct this info from {(local ghost index, remote
    // source rank, remote source index)}.
    auto remote_sources =
      destination.get_storage<const points::Value>(meta_fid);
    // Essentially a GroupByKey of remote_sources, keys are the remote source
    // ranks and values are vectors of remote source indices.
    SendPoints grouped_shared_entities;
    for(const auto & [begin, end] : destination.ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        const auto & shared = remote_sources[ghost_idx];
        grouped_shared_entities[shared.first].emplace_back(shared.second);
      }
    }

    // Do a limited, ragged Alltoall communication (like util::mpi::alltoallv
    // but without the serialization overhead) to create the inverse mapping of
    // group_shared_entities. This creates a map from remote destination rank to
    // a vector of *local* source indices. This information is later used by
    // MPI_Send().
    remote_ghost_entities = shuffle(grouped_shared_entities);

    // We need to figure out the max local source index in order to give correct
    // nelems when calling region::get_storage().
    for(const auto & [rank, indices] : remote_ghost_entities) {
      max_local_source_idx = std::max(max_local_source_idx,
        *std::max_element(indices.begin(), indices.end()));
    }
    max_local_source_idx += 1;

    // We need to reserve enough memory for MPI requests used in operator().
    // This only need to be calculated once.
    auto nrecvs = std::transform_reduce(destination.ghost_ranges.begin(),
      destination.ghost_ranges.end(),
      0,
      std::plus<>(), // this uses C++14 std::plus<void> where T is deduced.
      [](const auto& p) { return p.second - p.first; });
    auto nsends = std::transform_reduce(remote_ghost_entities.begin(),
      remote_ghost_entities.end(),
      0,
      std::plus<>(), // this uses C++14 std::plus<void> where T is deduced.
      [](const auto& p) { return p.second.size(); });

    nreqs = nrecvs + nsends;
  }

  // called with each field (and field_id_t) on the entity, for example, one
  // for pressure, temperature, density etc.
  void operator()(field_id_t data_fid) const {
    auto source_storage =
      source.r.get_storage<std::byte>(data_fid, max_local_source_idx);
    auto destination_storage = destination.get_storage<std::byte>(data_fid);

    // FIXME: should we assert(source_type_size == dest_type_size)?
    auto type_size = source.r.get_field_info(data_fid)->type_size;

    std::vector<MPI_Request> requests;
    requests.reserve(nreqs);

    auto remote_sources =
      destination.get_storage<const points::Value>(meta_fid);

    for(const auto & [begin, end] : destination.ghost_ranges) {
      for(auto ghost_idx = begin; ghost_idx < end; ++ghost_idx) {
        auto source_rank = remote_sources[ghost_idx].first;
        requests.resize(requests.size() + 1);
        MPI_Irecv(destination_storage.data() + ghost_idx * type_size,
          type_size,
          MPI_BYTE,
          source_rank,
          0,
          MPI_COMM_WORLD,
          &requests.back());
      }
    }

    for(const auto & [dest_rank, local_indices] : remote_ghost_entities) {
      for(auto shared_idx : local_indices) {
        requests.resize(requests.size() + 1);
        MPI_Isend(source_storage.data() + shared_idx * type_size,
          type_size,
          MPI_BYTE,
          int(dest_rank),
          0,
          MPI_COMM_WORLD,
          &requests.back());
      }
    }

    std::vector<MPI_Status> status(requests.size());
    MPI_Waitall(requests.size(), requests.data(), status.data());
  }

private:
  // (rank, { indices })
  using SendPoints = std::map<std::size_t, std::vector<std::size_t>>;

  static SendPoints shuffle(SendPoints & shared_entities) {
    auto [not_used, nranks] = util::mpi::info(MPI_COMM_WORLD);

    std::vector<std::size_t> send_counts(nranks);
    for(const auto & [rank, indices] : shared_entities) {
      send_counts[rank] = indices.size();
    }

    std::vector<std::size_t> recv_counts(nranks);
    MPI_Alltoall(send_counts.data(),
      1,
      util::mpi::type<std::size_t>(),
      recv_counts.data(),
      1,
      util::mpi::type<std::size_t>(),
      MPI_COMM_WORLD);

    std::vector<MPI_Request> requests;
    // We need to reserve enough memory for request, otherwise, .resize()
    // will invalidate &requests.back(). Since we send and receive at most
    // one message to each peer, the total number of requests is upper bounded
    // by 2 * nranks.
    requests.reserve(2 * nranks);
    SendPoints results;
    for(int i{0}; i < nranks; ++i) {
      if(recv_counts[i] > 0) {
        results.emplace(i, recv_counts[i]);
        requests.resize(requests.size() + 1);
        MPI_Irecv(results[i].data(),
          recv_counts[i],
          util::mpi::type<std::size_t>(),
          i,
          0,
          MPI_COMM_WORLD,
          &requests.back());
      }
    }

    for(int i{0}; i < nranks; ++i) {
      if(send_counts[i] > 0) {
        requests.resize(requests.size() + 1);
        MPI_Isend(shared_entities[i].data(),
          send_counts[i],
          util::mpi::type<std::size_t>(),
          i,
          0,
          MPI_COMM_WORLD,
          &requests.back());
      }
    }

    std::vector<MPI_Status> status(requests.size());
    MPI_Waitall(requests.size(), requests.data(), status.data());
    return results;
  }

  const points & source;
  const intervals & destination;
  field_id_t meta_fid;
  SendPoints remote_ghost_entities;
  std::size_t max_local_source_idx = 0;
  std::size_t nreqs = 0;
};
} // namespace data
} // namespace flecsi
