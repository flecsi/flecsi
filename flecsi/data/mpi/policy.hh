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

#include <cstddef>
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
    // copy assignment here since metadata is an util::span while ghost_ranges
    // is a std::vector<>.
    auto metadata = p.get_storage<Value>(fid);
    std::copy(
      metadata.begin(), metadata.end(), std::back_inserter(ghost_ranges));
    // Get The largest value of `end index` in ghost_ranges (i.e. the upper
    // bound). This tells how much memory needs to be allocated for ghost
    // entities.
    max_end = std::max_element(
      ghost_ranges.begin(), ghost_ranges.end(), [](Value x, Value y) {
        return x.second < y.second;
      })->second;
  }

private:
  // This member function is only called by either points or copy_engine.
  friend struct points;
  friend struct copy_engine;

  template<typename T>
  auto get_storage(field_id_t fid) {
    return r.get_storage<T>(fid, max_end);
  }

  // FIXME: who actually populates the storage?
  // We use a reference to region instead of pointer since it is not nullable
  // nor needs to be redirected after initialization.
  region_base & r;

  // Locally cached metadata on ranges of ghost index.
  // TODO: who need to have access to ghost_ranges?
  std::vector<Value> ghost_ranges;
  std::size_t max_end;
};

struct points {
  using Value = std::pair<std::size_t, std::size_t>; // (rank, index)
  static Value make(std::size_t r, std::size_t i) {
    return {r, i};
  }

  // FIXHIM: what Davis wrote in topology.hh regarding points might not be
  // correct.
  points(const region_base & r,
    intervals & intervals,
    field_id_t fid,
    completeness = incomplete)
    : source(r) {
    // Called by upper layer. Created from an intervals. The region `r` contains
    // field data of shared entities on this rank as source to be copied to
    // remote peers. The region and the field selected by `fid` in the
    // `intervals` contains metadata of index of shared entities on remote
    // peers and their ranks, as source to be copy from remote peers.
    const auto & remote_shared = intervals.get_storage<Value>(fid);
    for(auto & [begin, end] : intervals.ghost_ranges) {
      std::cout << "my rank: " << flecsi::run::context::instance().process()
                << std::endl;

      for(auto ghost = begin; ghost < end; ++ghost) {
        std::cout << ", ghost index: " << ghost
                  << ", source rank: " << remote_shared[ghost].first
                  << ", source index: " << remote_shared[ghost].second;
        std::cout << std::endl;
      }
    }

    // TODO: there is no information about the indices of local shared entities
    //  and ranks of the destination of copy (local source index, {remote
    //  destination ranks}). I need to do a shuffle operation to reconstruct
    //  this info from (local ghost index, remote source rank, remote source
    //  index).
  }

  const region_base & source;
};

struct copy_engine {
  // Upper layer supplies point which has information about shared entities
  // (individual indices), intervals which have information about ghost entities
  // (index_begin, index_end), the real data is stored in mpi::region r
  // with data_fid. I need to push the bits over the wire.
  // Question: How can I find T here? It is NOT available anywhere (except
  // implicitly in private region::fs which has sizeof(t)). Davis suggested
  // going back to std::byte land, blah, blah.
  // auto storage = r.get_storage<T>();
  // (MPI_Send/Receive r.storage()[]).

  // FIXME: where should I store "states" like MPI_Window, MPI_Comm etc.? How
  //  should I clean up those resources? If we manage resource here, should
  //  copy_engine copyable or move only (or share_ptr<> to resources)?
  // ANS: it will be the states of copy_engine.
  // One copy engine for each entity type, vertex, cell, edge, each supplied
  // with the same field_id_t.
  copy_engine(const points & shared,
    const intervals & ghosts,
    field_id_t /* metadata field of WHAT? */)
    : shared(shared), ghosts(ghosts) {}

  // called with each field (and field_id_t) on the entity, for example, one
  // for pressure, temperature, density etc.
  void operator()(field_id_t /* data field to be copied */) const {
    // To be defined....
    //    const auto & shared = intervals.get_storage<Value>(fid);
    //    for(auto & [begin, end] : intervals.ghost_ranges) {
    //      std::cout << "my rank: " <<
    //      flecsi::run::context::instance().process();
    //
    //      for(auto ghost = begin; ghost < end; ++ghost) {
    //        std::cout << ", ghost index: " << ghost
    //                  << ", peer rank: " << shared[ghost].first
    //                  << ", shared index: " << shared[ghost].second;
    //        std::cout << std::endl;
    //      }
  }

private:
  const points & shared;
  const intervals & ghosts;
};
} // namespace data
} // namespace flecsi
