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
  // know the exact number yet, a large number is used.
  //
  // Generic frontend code supplies `s` and `fs` (for information about fields),
  // requesting memory to be (notionally) reserved from backend. Here we only
  // create the underlying std::vector<> without actually allocating any memory.
  // The client code (mostly mpi::task_prologue) will call .get_storage() with
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
  void vacuous(field_id_t) {}

private:
  size2 s;
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
    // This constructor is usually (almost always) called when r.s.second != 4G
    // meaning it has the actual value. In this case, r.s.second is the number
    // of elements in the partition on this rank (the same value on all ranks
    // though).
    nelems = r.size().second;
  }

  partition(region & r,
    const partition & other,
    field_id_t fid,
    completeness = incomplete)
    : r(&r) {
    // Constructor for the case when how the data is partitioned is stored
    // as a field in another region. Delegate to update()
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
    // The number of elements for each ranks is stored as a field of the `row`
    // data type on the `other` partition.
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

  intervals(const region_base &,
    const partition &,
    field_id_t /* fid */,
    completeness = incomplete) {
    // Called by upper layer. Created from a partition. Tells me which cells
    // are ghost (destination) on which rank.
    // Use the partition.get_storage() to get the storage, not the
    // region.get_storage(). The same thing as one partition store the..
    //
    // The element type stored in partition.get_storage() is Value defined
    // above. This works in a similar way as partition.update(), see the
    // comment over there. Note: we need to copy the Value from the
    // storage and save locally. User code might change it after this
    // constructor returns.
    //
    // auto storage = parition.get_storage<Value>(fid); // gets the meta-data.
    // for (auto x : xs)
    //  saved.push_back(storage[i]);
    // private:
    //  std::vector<Value> saved;
    //  region *r;
    //  \strikeout{partition * p;}
    // public:
    //
    //  template <typename T>
    //  auto get_storage() { return r->get_storage<T>(fid); } // get the read
    //  data
    //
    //  \strikeout{region& get_region() { return *r; }}
    //  \strikeout{partition& get_partition(fid) { return *p;}}
    // TODO: add information I need to do ghost copy. It is totally up to
    //  me on what needs to be stored.
  }

  // TODO:
  //  private:
  //   reference to region and/or partition if needed.
  //   std::vector<subrow aka Value>
  //   aka std::vector<std::pair<std::size_t,std::size_t>>
};

struct points {
  using Value = std::pair<std::size_t, std::size_t>;
  static Value make(std::size_t r, std::size_t i) {
    return {r, i};
  }

  // FIXME: Just to silence the compiler, I have no idea what I am doing.
  points(const region_base &,
    const intervals &,
    field_id_t /* fid */,
    completeness = incomplete) {
    // Called by upper layer. Create from an intervals. Tells me which cells
    // are shared (source) on which rank.

    // Use the private std::vector<Value> saved in intervals. We also need
    // use intervals::get_region()::get_storage() or better
    // intervals::get_storage(), to get the Value (as typedef above),
    // Value = size_t, size_t, i.e. rank and local index on the rank.

    // for (auto &[b, e] : intervals.saved)
    //   for (auto i=b; i<e; ++i)
    //     \strikeout{use(ivals.get_partition().get_storage<Value>(fid)[i]);}
    //     use(ivals.get_storage<points::Value>(fid)[i]);

    // TODO: add information I need to do ghost copy. It is totally up to
    //  me on what needs to be stored.

    // TODO:
    //  private:
    //   store information about memory locations on the peer so I can do things
    //   like one sided MPI communication (e.g. MPI_Datatype, MPI_Window etc.).
  }
};

struct copy_engine {
  // Upper layer supplies point which has information about shared cells
  // (individual indices), interval which have information about ghost cells
  // (index_begin, index_end), the real data is stored in mpi::region r
  // with data_fid. I need to push the bits over the wire.
  // Question: How can I find T here? It is NOT available anywhere (except
  // implicitly in private region::fs which has sizeof(t)). Davis suggested
  // going back to std::byte land, blah, blah.
  // auto storage = r.get_storage<T>();
  // (MPI_Send/Receive r.storage()[]).
  copy_engine(const points &, const intervals &, field_id_t) {}

  void operator()(field_id_t) const {}
};
} // namespace data
} // namespace flecsi
